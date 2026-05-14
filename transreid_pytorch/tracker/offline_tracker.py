import os
import sys
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import collections

# Import from parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import cfg
from datasets.bases import read_image
from model import make_model
from utils.logger import setup_logger

from tracker.kalman_filter import KalmanFilter
from tracker.basetrack import BaseTrack, TrackState
from tracker import matching

def xywh_to_xyah(xywh):
    """Convert [x, y, w, h] to [cx, cy, a, h]"""
    ret = np.asarray(xywh).copy()
    ret[0] = ret[0] + ret[2] / 2
    ret[1] = ret[1] + ret[3] / 2
    ret[2] = ret[2] / ret[3]
    return ret

def xyah_to_tlbr(xyah):
    """Convert [cx, cy, a, h] to [x1, y1, x2, y2]"""
    ret = np.asarray(xyah).copy()
    w = ret[2] * ret[3]
    ret[0] = ret[0] - w / 2
    ret[1] = ret[1] - ret[3] / 2
    ret[2] = ret[0] + w
    ret[3] = ret[1] + ret[3]
    return ret

class Tracklet(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, feat, frame_id):
        
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        
        self.score = score
        self.tracklet_len = 0
        
        self.curr_feat = feat
        self.smooth_feat = feat.copy()
        self.features = [feat]
        
        # History for offline tracking
        self.history_tlwh = [self.tlwh.copy()]
        self.history_frames = [frame_id]
        
        # State tracking
        self.frame_id = frame_id
        self.start_frame = frame_id

    @property
    def tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x, max y)`."""
        if self.mean is None:
            ret = self.tlwh.copy()
            ret[2] += ret[0]
            ret[3] += ret[1]
            return ret
        ret = self.mean[:4].copy()
        ret = xyah_to_tlbr(ret)
        return ret

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[4:] = 0 # reset velocity if not tracked
        self.shared_kalman.predict(self.kf)
        self.mean = self.kf.x.flatten()
        self.covariance = self.kf.P

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()
        self.kalman_filter = kalman_filter
        
        # [x,y,w,h] -> [cx, cy, a, h]
        xyah = xywh_to_xyah(self.tlwh)
        self.kf = kalman_filter.initiate(xyah)
        self.mean = self.kf.x.flatten()
        self.covariance = self.kf.P
        
        self.tracklet_len = 1
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: Tracklet
        :type frame_id: int
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self.tlwh = new_tlwh
        self.history_tlwh.append(new_tlwh.copy())
        self.history_frames.append(frame_id)
        
        xyah = xywh_to_xyah(new_tlwh)
        self.shared_kalman.update(self.kf, xyah)
        self.mean = self.kf.x.flatten()
        self.covariance = self.kf.P
        
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        
        # Update feature
        self.curr_feat = new_track.curr_feat
        self.features.append(self.curr_feat)
        # EMA smoothing for the tracklet feature
        alpha = 0.9
        self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * self.curr_feat
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

class MotDetCropDataset(Dataset):
    """
    Iterates over pred detections and returns person crops.
    """
    def __init__(self, seq_root, det_path, transform=None, min_score=0.3):
        self.seq_root = Path(seq_root)
        self.img_dir = self.seq_root / "img1"
        self.det_path = Path(det_path)

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Missing img1: {self.img_dir}")
        if not self.det_path.exists():
            raise FileNotFoundError(f"Missing det.txt: {self.det_path}")

        # format: <frame>, -1, <x1>, <y1>, <w>, <h>, <score>, -1, -1, -1
        df = pd.read_csv(
            self.det_path,
            header=None,
            names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"],
        )

        df = df[df["conf"] >= min_score]
        df = df[(df["bb_width"] > 0) & (df["bb_height"] > 0)]

        df["frame"] = df["frame"].astype(int)

        self.frame_to_path = {}
        imgs = sorted(self.img_dir.glob("*"))
        if not imgs:
            raise ValueError(f"No images found in {self.img_dir}")

        for p in imgs:
            if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
                continue
            try:
                f = int(p.stem)
                self.frame_to_path[f] = p
            except ValueError:
                pass

        # filter df
        df = df[df["frame"].isin(self.frame_to_path.keys())].reset_index(drop=True)
        # we assign a temporary detection id for sorting
        df["det_idx"] = np.arange(len(df))

        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame = int(row["frame"])
        det_idx = int(row["det_idx"])
        score = float(row["conf"])

        img_path = self.frame_to_path[frame]
        img = read_image(str(img_path))
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        W, H = img.size
        x = float(row["bb_left"])
        y = float(row["bb_top"])
        w = float(row["bb_width"])
        h = float(row["bb_height"])

        x1 = int(max(0, np.floor(x)))
        y1 = int(max(0, np.floor(y)))
        x2 = int(min(W, np.ceil(x + w)))
        y2 = int(min(H, np.ceil(y + h)))

        if x2 <= x1 or y2 <= y1:
            crop = Image.new("RGB", (128, 256), (0, 0, 0))
        else:
            crop = img.crop((x1, y1, x2, y2)).convert("RGB")

        if self.transform is not None:
            crop = self.transform(crop)

        tlwh = np.array([x, y, w, h], dtype=np.float32)
        return crop, det_idx, frame, score, tlwh

class OfflineTracker:
    def __init__(self, model, device="cuda", l2_normalize=True):
        self.model = model
        self.device = device
        self.l2_normalize = l2_normalize
        self.kalman_filter = KalmanFilter()
        
        # Local tracking params
        self.match_thresh = 0.8
        self.iou_thresh = 0.5
        # We are strict here. We want to stop tracklets on occlusion.
        
        # Global tracking params
        self.max_time_gap = 60
        self.global_match_thresh = 0.7

    def extract_features(self, dataset, batch_size=64, num_workers=4):
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, pin_memory=(self.device != "cpu")
        )
        
        self.model.eval()
        
        # Store detections by frame
        detections_by_frame = collections.defaultdict(list)
        
        with torch.no_grad():
            for crops, det_idxs, frames, scores, tlwhs in tqdm(loader, desc="Extracting Features"):
                crops = crops.to(self.device, non_blocking=True)
                feats = self.model(crops, cam_label=None, view_label=None)
                if self.l2_normalize:
                    feats = F.normalize(feats, dim=1, p=2)
                feats = feats.cpu().numpy()
                frames = frames.numpy().astype(int)
                scores = scores.numpy().astype(float)
                tlwhs = tlwhs.numpy()
                
                for i in range(len(frames)):
                    frame = frames[i]
                    trk = Tracklet(tlwhs[i], scores[i], feats[i], frame)
                    detections_by_frame[frame].append(trk)
                    
        return detections_by_frame

    def local_association(self, detections_by_frame):
        """
        Phase 1: Strict frame-by-frame matching to build pure tracklets.
        """
        all_tracklets = [] # All tracklets ever created (we never delete them)
        tracked_tracklets = []
        
        frames = sorted(detections_by_frame.keys())
        if not frames:
            return []
            
        for frame in tqdm(frames, desc="Local Association"):
            detections = detections_by_frame[frame]
            
            # Predict
            for track in tracked_tracklets:
                track.predict()
                
            # Cost Matrix
            if len(tracked_tracklets) > 0 and len(detections) > 0:
                # Combining IoU and ReID. Since we want to be strict to avoid identity switches,
                # we penalize high cosine distance heavily.
                iou_dists = matching.iou_distance(tracked_tracklets, detections)
                reid_dists = matching.embedding_distance(tracked_tracklets, detections)
                
                # If they don't overlap, make cost very high to enforce separation
                # (or if you trust ReID completely, you can lower this, but for tracking IoU gating is safe)
                iou_mask = iou_dists > self.iou_thresh
                
                # Combine
                cost_matrix = 0.5 * iou_dists + 0.5 * reid_dists
                cost_matrix[iou_mask] = np.inf
                cost_matrix[reid_dists > 0.4] = np.inf # Strict ReID threshold
                
                matches, u_track, u_det = matching.linear_assignment(cost_matrix, thresh=self.match_thresh)
            else:
                matches = []
                u_track = np.arange(len(tracked_tracklets))
                u_det = np.arange(len(detections))
                
            # Update matches
            for itracked, idet in matches:
                track = tracked_tracklets[itracked]
                det = detections[idet]
                track.update(det, frame)
                
            # Handle unmatched tracks -> mark them lost. We do not keep them active for next frame
            # in the local phase to avoid drift. We'll link them in the global phase.
            for it in u_track:
                track = tracked_tracklets[it]
                track.mark_lost()
                
            # Handle unmatched detections -> start new tracklet
            new_tracklets = []
            for idet in u_det:
                det = detections[idet]
                det.activate(self.kalman_filter, frame)
                new_tracklets.append(det)
                all_tracklets.append(det)
                
            tracked_tracklets = [t for t in tracked_tracklets if t.state == TrackState.Tracked]
            tracked_tracklets.extend(new_tracklets)
            
        return all_tracklets

    def global_association(self, all_tracklets):
        """
        Phase 2: Link broken tracklets using ReID and spatial-temporal constraints.
        """
        # Filter noise tracklets (too short)
        valid_tracklets = [t for t in all_tracklets if t.tracklet_len >= 2 or t.score > 0.6]
        
        # We need to construct a graph of all valid tracklets
        n = len(valid_tracklets)
        cost_matrix = np.full((n, n), np.inf)
        
        for i in tqdm(range(n), desc="Building Global Graph"):
            for j in range(n):
                if i == j: continue
                
                t_i = valid_tracklets[i]
                t_j = valid_tracklets[j]
                
                # t_j must happen after t_i
                time_gap = t_j.start_frame - t_i.end_frame
                
                if 0 < time_gap <= self.max_time_gap:
                    # Spatial feasibility (just simple distance for now, could use KF projection)
                    # Get last box of t_i and first box of t_j
                    box_i = t_i.history_tlwh[-1]
                    box_j = t_j.history_tlwh[0]
                    
                    # Compute centers
                    cx_i, cy_i = box_i[0] + box_i[2]/2, box_i[1] + box_i[3]/2
                    cx_j, cy_j = box_j[0] + box_j[2]/2, box_j[1] + box_j[3]/2
                    
                    # Simple speed limit (pixels per frame)
                    dist = np.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
                    speed = dist / time_gap
                    
                    # Assuming max speed (e.g. 20 pixels per frame on 1080p, depending on resolution)
                    # We can use a relaxed threshold
                    if speed < 50.0:
                        # ReID distance
                        reid_dist = 1.0 - np.dot(t_i.smooth_feat, t_j.smooth_feat)
                        if reid_dist < self.global_match_thresh:
                            # We can also add a small time penalty
                            cost = reid_dist + 0.1 * (time_gap / self.max_time_gap)
                            cost_matrix[i, j] = cost

        matches, _, _ = matching.linear_assignment(cost_matrix, thresh=self.global_match_thresh)
        
        # Merge tracklets
        # We represent merges using a Union-Find or a simple linked list
        next_track = {i: j for i, j in matches}
        
        # Find roots
        has_prev = {j: True for i, j in matches}
        
        final_tracks = []
        global_id = 1
        
        for i in range(n):
            if not has_prev.get(i, False):
                # i is a start of a trajectory
                merged_tlwh = []
                merged_frames = []
                
                curr = i
                while True:
                    t = valid_tracklets[curr]
                    # In a real tracker, we would interpolate between the gap
                    # from t_prev.end_frame to t_curr.start_frame
                    merged_tlwh.extend(t.history_tlwh)
                    merged_frames.extend(t.history_frames)
                    
                    if curr in next_track:
                        curr = next_track[curr]
                    else:
                        break
                        
                # Perform linear interpolation for missing frames
                interp_frames, interp_tlwh = self.interpolate(merged_frames, merged_tlwh)
                
                final_tracks.append((global_id, interp_frames, interp_tlwh))
                global_id += 1
                
        return final_tracks

    def interpolate(self, frames, tlwhs):
        """Linearly interpolate missing frames in a track."""
        if len(frames) == 0:
            return [], []
            
        frames = np.array(frames)
        tlwhs = np.array(tlwhs)
        
        min_f, max_f = frames[0], frames[-1]
        full_frames = np.arange(min_f, max_f + 1)
        
        full_tlwhs = np.zeros((len(full_frames), 4))
        for i in range(4):
            full_tlwhs[:, i] = np.interp(full_frames, frames, tlwhs[:, i])
            
        return full_frames.tolist(), full_tlwhs.tolist()

def write_results(filename, final_tracks):
    """Write tracking results to MOT format"""
    results = []
    for track_id, frames, tlwhs in final_tracks:
        for f, tlwh in zip(frames, tlwhs):
            x, y, w, h = tlwh
            # MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
            results.append(f"{int(f)},{int(track_id)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
            
    with open(filename, 'w') as f:
        f.writelines(results)

def build_test_transform():
    return T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )

def run_offline_tracker(seq_root, det_path, out_path, weight_path, device="cuda"):
    logger = setup_logger("offline_tracker", str(Path(out_path).parent), if_train=False)
    logger.info(f"Running offline tracker on {seq_root}")
    
    cfg.merge_from_file(str(Path(__file__).resolve().parent.parent / "configs/market/vit_small.yml"))
    cfg.MODEL.PRETRAIN_CHOICE = "finetune"
    cfg.TEST.WEIGHT = weight_path
    cfg.freeze()

    device = torch.device(device)
    model = make_model(cfg, num_class=1, camera_num=1, view_num=1)
    
    # Load weights
    from embed_video import load_checkpoint_for_embeddings
    load_checkpoint_for_embeddings(model, cfg.TEST.WEIGHT, logger)
    
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    transform = build_test_transform()
    
    dataset = MotDetCropDataset(seq_root, det_path, transform=transform, min_score=0.3)
    
    tracker = OfflineTracker(model, device=str(device))
    
    logger.info("Extracting features...")
    detections_by_frame = tracker.extract_features(dataset, batch_size=64, num_workers=cfg.DATALOADER.NUM_WORKERS)
    
    logger.info("Phase 1: Local Association...")
    tracklets = tracker.local_association(detections_by_frame)
    
    logger.info("Phase 2: Global Association...")
    final_tracks = tracker.global_association(tracklets)
    
    logger.info(f"Writing results to {out_path}...")
    write_results(out_path, final_tracks)
    logger.info("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_root", type=str, required=True, help="Path to sequence root (contains img1/)")
    parser.add_argument("--det_path", type=str, required=True, help="Path to det.txt")
    parser.add_argument("--out_path", type=str, default="tracking_results.txt", help="Output path")
    parser.add_argument("--weight", type=str, required=True, help="Path to TransReID weight")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    run_offline_tracker(args.seq_root, args.det_path, args.out_path, args.weight, args.device)
