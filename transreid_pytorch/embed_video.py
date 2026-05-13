import os
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from config import cfg
from datasets.bases import read_image  # your repo helper
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import make_model
from utils.logger import setup_logger
from tqdm import tqdm
import torch.nn as nn
class MotGtCropDataset(Dataset):
    """
    Iterates over GT detections (rows of gt.txt) and returns person crops.
    """

    def __init__(self, seq_root, transform=None, min_box_area=10, use_conf_only_if_present=False):
        self.seq_root = Path(seq_root)
        self.img_dir = self.seq_root / "img1"
        self.gt_path = self.seq_root / "gt" / "gt.txt"

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Missing img1: {self.img_dir}")
        if not self.gt_path.exists():
            raise FileNotFoundError(f"Missing gt.txt: {self.gt_path}")

        # MOT format usually: frame, id, x, y, w, h, conf, x, y, z
        df = pd.read_csv(
            self.gt_path,
            header=None,
            names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"],
        )

        # If your gt.txt is "pure GT", conf might be 1 always. Keep it simple:
        if use_conf_only_if_present and "conf" in df.columns:
            df = df[df["conf"] > 0]

        # Basic validity
        df = df[(df["bb_width"] > 0) & (df["bb_height"] > 0)]
        df["area"] = df["bb_width"] * df["bb_height"]
        df = df[df["area"] >= float(min_box_area)]

        # Ensure int for indexing
        df["frame"] = df["frame"].astype(int)
        df["id"] = df["id"].astype(int)

        # Build frame -> image path (DanceTrack uses 000001.jpg style)
        # If your naming differs, adjust here.
        self.frame_to_path = {}
        # Pre-scan all images once
        imgs = sorted(self.img_dir.glob("*"))
        if not imgs:
            raise ValueError(f"No images found in {self.img_dir}")

        # Common MOT naming: 000001.jpg ... or 000001.png
        # We'll map by stem int.
        for p in imgs:
            if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
                continue
            try:
                f = int(p.stem)
                self.frame_to_path[f] = p
            except ValueError:
                # if stems not numeric, you need a custom mapping
                pass

        if not self.frame_to_path:
            raise ValueError(
                f"Could not map frames from filenames in {self.img_dir}. "
                f"Expected numeric stems like 000001.jpg"
            )

        # Keep only GT rows for which the frame image exists
        df = df[df["frame"].isin(self.frame_to_path.keys())].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No GT rows match available frames in img1.")

        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame = int(row["frame"])
        tid = int(row["id"])

        img_path = self.frame_to_path[frame]
        # You use read_image; ensure it returns PIL.Image or numpy in RGB.
        # If read_image returns PIL already, great. If numpy, convert.
        img = read_image(str(img_path))
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        W, H = img.size
        x = float(row["bb_left"])
        y = float(row["bb_top"])
        w = float(row["bb_width"])
        h = float(row["bb_height"])

        # Clamp and convert to pixel box
        x1 = int(max(0, np.floor(x)))
        y1 = int(max(0, np.floor(y)))
        x2 = int(min(W, np.ceil(x + w)))
        y2 = int(min(H, np.ceil(y + h)))

        # Handle degenerate boxes after clamp
        if x2 <= x1 or y2 <= y1:
            # Return a tiny black crop (or raise). Black is safer for dataloader.
            raise ValueError("Invalid box")
            # crop = Image.new("RGB", (cfg.INPUT.SIZE_TEST[1], cfg.INPUT.SIZE_TEST[0]), (0, 0, 0))
        else:
            crop = img.crop((x1, y1, x2, y2)).convert("RGB")

        if self.transform is not None:
            crop = self.transform(crop)

        bbox = np.array([x1, y1, x2, y2], dtype=np.int32)
        return crop, tid, frame, bbox, str(img_path)
    
    
def build_test_transform():
    return T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )
    


def load_checkpoint_for_embeddings(model, checkpoint_path, logger):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for candidate_key in ("teacher", "student", "state_dict", "model", "model_state_dict"):
            if candidate_key in checkpoint and isinstance(checkpoint[candidate_key], dict):
                checkpoint = checkpoint[candidate_key]
                break
        for candidate_key in ("state_dict", "model", "model_state_dict"):
            if candidate_key in checkpoint and isinstance(checkpoint[candidate_key], dict):
                checkpoint = checkpoint[candidate_key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format: {}".format(type(checkpoint)))

    model_state = model.state_dict()
    loadable_state = {}
    skipped_keys = []

    for key, value in checkpoint.items():
        clean_key = key.replace("module.", "").replace("backbone.", "base.")
        if "classifier" in clean_key:
            continue
        if clean_key not in model_state:
            continue
        if not hasattr(value, "shape"):
            continue
        if model_state[clean_key].shape != value.shape:
            skipped_keys.append(clean_key)
            continue
        loadable_state[clean_key] = value

    incompatible = model.load_state_dict(loadable_state, strict=False)
    if not loadable_state:
        raise RuntimeError("No compatible tensors were found in {}".format(checkpoint_path))
    logger.info("Loaded %d tensors from %s", len(loadable_state), checkpoint_path)
    if skipped_keys:
        logger.info("Skipped %d tensors with shape mismatches", len(skipped_keys))
    if incompatible.missing_keys:
        logger.info("Missing keys after load: %d", len(incompatible.missing_keys))
    if incompatible.unexpected_keys:
        logger.info("Unexpected keys after load: %d", len(incompatible.unexpected_keys))

def normalize_embeddings(embeddings):
    if str(cfg.TEST.FEAT_NORM).lower() == "yes":
        embeddings = F.normalize(embeddings, dim=1, p=2)
    return embeddings



def extract_and_save_tracks(
    model,
    dataset,
    output_dir,
    batch_size=64,
    num_workers=4,
    device="cuda",
    l2_normalize=True,
):
    output_dir = Path(output_dir)
    track_dir = output_dir / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
        drop_last=False,
    )

    # track_id -> list of (frame, emb, bbox)
    store = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for crops, tids, frames, bboxes, _img_paths in tqdm(loader):
            crops = crops.to(device, non_blocking=True)

            feats = model(crops, cam_label=None, view_label=None)
            if l2_normalize:
                feats = F.normalize(feats, dim=1, p=2)

            feats = feats.detach().cpu().numpy()
            tids = tids.numpy().astype(int)
            frames = frames.numpy().astype(int)
            bboxes = bboxes.numpy().astype(np.int32)

            for i in range(len(tids)):
                store[int(tids[i])].append((int(frames[i]), feats[i], bboxes[i]))

    # Write per-track files
    index_rows = []
    for tid, items in store.items():
        # sort by time
        items.sort(key=lambda x: x[0])
        frames_arr = np.array([it[0] for it in items], dtype=np.int32)
        emb_arr = np.stack([it[1] for it in items], axis=0).astype(np.float32)
        bbox_arr = np.stack([it[2] for it in items], axis=0).astype(np.int32)

        npy_path = track_dir / f"track_{tid:06d}.npy"
        fr_path = track_dir / f"track_{tid:06d}_frames.npy"
        bb_path = track_dir / f"track_{tid:06d}_bboxes.npy"

        np.save(npy_path, emb_arr)
        np.save(fr_path, frames_arr)
        np.save(bb_path, bbox_arr)

        index_rows.append(
            {
                "track_id": tid,
                "num_dets": int(len(items)),
                "emb_path": str(npy_path),
                "frames_path": str(fr_path),
                "bboxes_path": str(bb_path),
            }
        )

    # index.csv
    with open(track_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "num_dets", "emb_path", "frames_path", "bboxes_path"])
        w.writeheader()
        w.writerows(index_rows)

    return track_dir


def pca_2d(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T

def plot_tracks_pca_trajectories(track_dir, out_png, max_tracks=30, min_len=20):
    track_dir = Path(track_dir)
    tracks = []
    for p in sorted(track_dir.glob("track_*.npy")):
        if p.name.endswith("_frames.npy") or p.name.endswith("_bboxes.npy"):
            continue
        tid = int(p.stem.split("_")[1])
        E = np.load(p)
        if E.shape[0] >= min_len:
            Fm = np.load(track_dir / f"track_{tid:06d}_frames.npy")
            tracks.append((tid, Fm, E))
        if len(tracks) >= max_tracks:
            break

    if not tracks:
        raise ValueError("No tracks matching min_len/max_tracks constraints.")

    X = np.concatenate([t[2] for t in tracks], axis=0)
    X2 = pca_2d(X)

    # split back
    idx = 0
    plt.figure(figsize=(10, 8))
    for (tid, Fm, E) in tracks:
        n = E.shape[0]
        pts = X2[idx:idx+n]
        idx += n
        # plot polyline + points (no fixed colors requested; matplotlib will cycle)
        plt.plot(pts[:, 0], pts[:, 1], linewidth=1, alpha=0.9)
        plt.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.7)
        plt.text(pts[0, 0], pts[0, 1], str(tid), fontsize=8)

    plt.title("Track embedding trajectories (PCA2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    
    
def run_on_val_root(val_root, out_root, args, logger):
    val_root = Path(val_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    model = make_model(cfg, num_class=1, camera_num=1, view_num=1)
    load_checkpoint_for_embeddings(model, args.weight, logger)
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    transform = build_test_transform()

    for seq_dir in sorted([p for p in val_root.iterdir() if p.is_dir() and (p / "img1").exists()]):
        seq_out = out_root / seq_dir.name
        seq_out.mkdir(parents=True, exist_ok=True)
        logger.info("Processing %s", seq_dir)

        ds = MotGtCropDataset(seq_dir, transform=transform, min_box_area=50)
        track_dir = extract_and_save_tracks(
            model=model,
            dataset=ds,
            output_dir=seq_out,
            batch_size=args.batch_size,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            device=str(device),
            l2_normalize=(str(cfg.TEST.FEAT_NORM).lower() == "yes"),
        )

        plot_tracks_pca_trajectories(track_dir, seq_out / "tracks_pca_trajectories.png", max_tracks=40, min_len=20)

        logger.info("Saved outputs to %s", seq_out)
def run_on_seq_root(seq_root, out_root, logger, batch_size = 8):
    seq_root = Path(seq_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    model = make_model(cfg, num_class=1, camera_num=1, view_num=1)
    load_checkpoint_for_embeddings(model, cfg.TEST.WEIGHT, logger)
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    transform = build_test_transform()

    seq_out = out_root
    seq_out.mkdir(parents=True, exist_ok=True)
    logger.info("Processing %s", seq_root)

    ds = MotGtCropDataset(seq_root, transform=transform, min_box_area=50)
    track_dir = extract_and_save_tracks(
        model=model,
        dataset=ds,
        output_dir=seq_out,
        batch_size=batch_size,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        device=str(device),
        l2_normalize=(str(cfg.TEST.FEAT_NORM).lower() == "yes"),
    )

    plot_tracks_pca_trajectories(track_dir, seq_out / "tracks_pca_trajectories.png", max_tracks=40, min_len=20)

    logger.info("Saved outputs to %s", seq_out)

def _load_track(track_dir: Path, tid: int):
    E = np.load(track_dir / f"track_{tid:06d}.npy").astype(np.float32)          # (Ti, D)
    Fm = np.load(track_dir / f"track_{tid:06d}_frames.npy").astype(np.int32)   # (Ti,)
    # ensure sorted (should already be)
    order = np.argsort(Fm)
    return Fm[order], E[order]

def _carry_forward_embeddings(frames_src, emb_src, frames_grid):
    """
    For each t in frames_grid, pick embedding at largest frame <= t.
    Returns mask for times where no embedding exists yet.
    """
    # idx = rightmost insertion position - 1
    idx = np.searchsorted(frames_src, frames_grid, side="right") - 1
    valid = idx >= 0
    out = np.zeros((len(frames_grid), emb_src.shape[1]), dtype=np.float32)
    out[valid] = emb_src[idx[valid]]
    return out, valid

def plot_pairwise_cosine_over_time(track_dir, tid_a, tid_b, out_png, smoothing_window=1):
    track_dir = Path(track_dir)
    Fa, Ea = _load_track(track_dir, tid_a)
    Fb, Eb = _load_track(track_dir, tid_b)

    # union time grid (frames where any of the two exists)
    grid = np.unique(np.concatenate([Fa, Fb], axis=0))

    A, valid_a = _carry_forward_embeddings(Fa, Ea, grid)
    B, valid_b = _carry_forward_embeddings(Fb, Eb, grid)
    valid = valid_a & valid_b
    if valid.sum() == 0:
        raise ValueError(f"No overlapping time (after carry-forward) between {tid_a} and {tid_b}")

    # cosine: assume embeddings already L2-normalized; if not, normalize here
    # A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    # B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    sims = (A * B).sum(axis=1)  # (T,)

    # mask invalid times
    sims_plot = np.full_like(sims, np.nan, dtype=np.float32)
    sims_plot[valid] = sims[valid]

    # optional smoothing (moving average over valid positions)
    if smoothing_window and smoothing_window > 1:
        w = int(smoothing_window)
        x = np.copy(sims_plot)
        # simple nan-aware moving average
        sm = np.full_like(x, np.nan)
        for i in range(len(x)):
            lo = max(0, i - w//2)
            hi = min(len(x), i + w//2 + 1)
            seg = x[lo:hi]
            seg = seg[~np.isnan(seg)]
            if len(seg) > 0:
                sm[i] = float(seg.mean())
        sims_plot = sm

    plt.figure(figsize=(12, 3.2))
    plt.plot(grid, sims_plot)
    plt.ylim(-1, 1)
    plt.title(f"Cosine similarity over time: track {tid_a} vs track {tid_b}")
    plt.xlabel("frame")
    plt.ylabel("cosine similarity")
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_all_pairs_over_time(track_dir, out_dir, track_ids=None, smoothing_window=1):
    track_dir = Path(track_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not track_ids:
        track_ids = range(len(pd.read_csv(track_dir/"index.csv")))
    import itertools
    for a, b in itertools.combinations(track_ids, 2):
        out_png = out_dir / f"cosine_time_{a:06d}_vs_{b:06d}.png"
        plot_pairwise_cosine_over_time(
            track_dir=track_dir,
            tid_a=a,
            tid_b=b,
            out_png=out_png,
            smoothing_window=smoothing_window,
        )
        
        
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def load_all_detections_from_tracks(track_dir: Path, min_track_len=20):
    """
    Returns a flat list of detections:
      frames (M,), tids (M,), embs (M,D)
    Assumes embeddings already L2-normalized (as in your pipeline).
    """
    track_dir = Path(track_dir)
    frames_all, tids_all, embs_all = [], [], []

    for p in sorted(track_dir.glob("track_*.npy")):
        if p.name.endswith("_frames.npy") or p.name.endswith("_bboxes.npy"):
            continue
        tid = int(p.stem.split("_")[1])
        E = np.load(p).astype(np.float32)  # (T,D)
        Fm = np.load(track_dir / f"track_{tid:06d}_frames.npy").astype(np.int32)

        if len(Fm) < min_track_len:
            continue

        order = np.argsort(Fm)
        Fm = Fm[order]
        E = E[order]

        frames_all.append(Fm)
        tids_all.append(np.full_like(Fm, tid, dtype=np.int32))
        embs_all.append(E)

    if not embs_all:
        raise ValueError("No tracks loaded (maybe min_track_len too high).")

    frames = np.concatenate(frames_all, axis=0)
    tids = np.concatenate(tids_all, axis=0)
    embs = np.concatenate(embs_all, axis=0)
    return frames, tids, embs

def build_frame_index(frames, tids, embs):
    """
    frame -> (tids_f, embs_f)
    """
    by_frame = defaultdict(list)
    for f, tid, e in zip(frames, tids, embs):
        by_frame[int(f)].append((int(tid), e))
    frame_map = {}
    for f, items in by_frame.items():
        tids_f = np.array([it[0] for it in items], dtype=np.int32)
        embs_f = np.stack([it[1] for it in items], axis=0).astype(np.float32)
        frame_map[f] = (tids_f, embs_f)
    return frame_map

def next_frame_reid_metrics(track_dir, out_png, min_track_len=20, min_pairs_per_frame=5):
    """
    For each frame t, checks whether each detection at t retrieves the correct
    identity among detections at t+1 via cosine NN.

    Produces:
      - accuracy(t)
      - margin(t): mean(best_same - best_other) over matchable detections
    """
    track_dir = Path(track_dir)
    frames, tids, embs = load_all_detections_from_tracks(track_dir, min_track_len=min_track_len)
    frame_map = build_frame_index(frames, tids, embs)

    all_frames = sorted(frame_map.keys())
    # We only evaluate frames where t and t+1 exist
    eval_frames = [f for f in all_frames if (f + 1) in frame_map]

    acc_t = []
    margin_t = []
    n_t = []

    for f in eval_frames:
        tids_a, Ea = frame_map[f]       # (Na,), (Na,D)
        tids_b, Eb = frame_map[f + 1]   # (Nb,), (Nb,D)

        # cosine similarity matrix (Na,Nb) since L2-normalized
        S = Ea @ Eb.T

        # For each i in frame f, best match index in frame f+1
        j_hat = np.argmax(S, axis=1)          # (Na,)
        tid_hat = tids_b[j_hat]               # (Na,)

        # only evaluate queries whose GT id exists in next frame
        next_ids_set = set(map(int, tids_b.tolist()))
        matchable = np.array([int(t) in next_ids_set for t in tids_a], dtype=bool)
        if matchable.sum() < min_pairs_per_frame:
            continue

        correct = (tid_hat == tids_a) & matchable
        acc = correct[matchable].mean().item()

        # margin: best_same - best_other (for matchable queries)
        # For each query i: best similarity to same-id in next frame
        # and best similarity to different-id in next frame.
        margins = []
        for i in np.where(matchable)[0]:
            gt = tids_a[i]
            sims = S[i]  # (Nb,)
            same_mask = (tids_b == gt)
            if not same_mask.any():
                continue
            best_same = np.max(sims[same_mask])
            best_other = np.max(sims[~same_mask]) if (~same_mask).any() else -np.inf
            margins.append(float(best_same - best_other))
        m = float(np.mean(margins)) if margins else np.nan

        acc_t.append(acc)
        margin_t.append(m)
        n_t.append(int(matchable.sum()))

    if len(acc_t) == 0:
        raise ValueError("No frames evaluated (check min_track_len / min_pairs_per_frame).")

    # Plot
    xs = np.arange(len(acc_t))  # index over evaluated frames
    frames_plot = np.array([f for f in eval_frames if (f+1) in frame_map], dtype=int)
    # Note: we skipped some frames by min_pairs_per_frame; align properly:
    # easiest is recompute frames_plot based on collected points:
    # We'll track frames in the loop instead for strict alignment.

def next_frame_reid_metrics_and_plot(track_dir, out_png, min_track_len=20, min_pairs_per_frame=5):
    track_dir = Path(track_dir)
    frames, tids, embs = load_all_detections_from_tracks(track_dir, min_track_len=min_track_len)
    frame_map = build_frame_index(frames, tids, embs)

    all_frames = sorted(frame_map.keys())
    eval_frames = [f for f in all_frames if (f + 1) in frame_map]

    frames_used = []
    acc_t = []
    margin_t = []
    n_t = []

    for f in eval_frames:
        tids_a, Ea = frame_map[f]
        tids_b, Eb = frame_map[f + 1]
        S = Ea @ Eb.T

        j_hat = np.argmax(S, axis=1)
        tid_hat = tids_b[j_hat]

        next_ids_set = set(map(int, tids_b.tolist()))
        matchable = np.array([int(t) in next_ids_set for t in tids_a], dtype=bool)
        if matchable.sum() < min_pairs_per_frame:
            continue

        correct = (tid_hat == tids_a) & matchable
        acc = correct[matchable].mean().item()

        margins = []
        for i in np.where(matchable)[0]:
            gt = tids_a[i]
            sims = S[i]
            same_mask = (tids_b == gt)
            if not same_mask.any():
                continue
            best_same = np.max(sims[same_mask])
            best_other = np.max(sims[~same_mask]) if (~same_mask).any() else -np.inf
            margins.append(float(best_same - best_other))
        m = float(np.mean(margins)) if margins else np.nan

        frames_used.append(f)
        acc_t.append(acc)
        margin_t.append(m)
        n_t.append(int(matchable.sum()))

    if len(acc_t) == 0:
        raise ValueError("No frames evaluated (check min_track_len / min_pairs_per_frame).")

    frames_used = np.array(frames_used, dtype=int)
    acc_t = np.array(acc_t, dtype=np.float32)
    margin_t = np.array(margin_t, dtype=np.float32)
    n_t = np.array(n_t, dtype=np.int32)

    plt.figure(figsize=(12, 4.2))
    plt.plot(frames_used, acc_t, label="next-frame rank-1 accuracy")
    # margin can be outside [-1,1]; keep separate scale? we’ll overlay but it may dwarf.
    # Better: plot margin on same plot but it’s okay if values are modest; otherwise comment out.
    plt.plot(frames_used, margin_t, label="mean margin (best_same - best_other)")
    plt.title("ReID next-frame matching quality over time")
    plt.xlabel("frame t")
    plt.ylabel("value (higher is better)")
    plt.grid(True, linewidth=0.5)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return frames_used, acc_t, margin_t, n_t
if __name__ =="__main__":
    
    cfg.merge_from_file("configs/market/vit_small.yml")
    cfg.MODEL.PRETRAIN_CHOICE = "finetune"
    cfg.TEST.WEIGHT = str(Path(r"..\pretrained\checkpoint0260.pth"))
    cfg.freeze()
    seq_root= Path(r"C:\Users\adamgass\OneDrive - myidemia\Documents\DanceTrack\val\val\dancetrack0094")
    out_root = Path.cwd() / seq_root.stem
    logger = setup_logger("transreid", str(out_root), if_train=False)
    # run_on_seq_root(seq_root, out_root ,logger)
    
    track_dir = out_root / "tracks"
    
    plot_all_pairs_over_time(
    track_dir=track_dir,
    out_dir=Path(track_dir) / "pairwise_cosine_time",
    smoothing_window=11,   # optional, helps reduce noise
)
    
    next_frame_reid_metrics_and_plot(
    track_dir,
    out_png=out_root / "next_frame_reid_over_time.png",
    min_track_len=20,
    min_pairs_per_frame=3,
)