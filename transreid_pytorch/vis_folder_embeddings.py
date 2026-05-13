import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import cfg
from datasets.bases import read_image
from model import make_model
from utils.logger import setup_logger


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class FolderImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        if not self.root_dir.exists():
            raise FileNotFoundError("Image folder does not exist: {}".format(self.root_dir))

        self.paths = [
            path
            for path in sorted(self.root_dir.rglob("*"), key=lambda item: item.as_posix().lower())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not self.paths:
            raise ValueError("No images found in {}".format(self.root_dir))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        image = read_image(str(img_path))
        if self.transform is not None:
            image = self.transform(image)
        return image, str(img_path)


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


def project_to_2d(embeddings):
    if embeddings.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1)
    return coords


def build_records(image_paths, image_root):
    root = Path(image_root).resolve()
    records = []
    for img_path in image_paths:
        path = Path(img_path).resolve()
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError:
            relative_path = path.as_posix()
        records.append(
            {
                "path": str(path),
                "relative_path": relative_path,
                "basename": path.name.lower(),
                "stem": path.stem.lower(),
            }
        )
    return records


def resolve_highlight_indices(records, highlight_tokens):
    selected_indices = []
    seen = set()
    unresolved = []

    for token in highlight_tokens:
        normalized_token = token.replace("\\", "/").lower().strip()
        token_path = Path(normalized_token)
        matches = []

        for index, record in enumerate(records):
            relative_path = record["relative_path"].lower()
            basename = record["basename"]
            stem = record["stem"]

            if normalized_token == relative_path:
                matches.append(index)
                continue
            if token_path.name == basename:
                matches.append(index)
                continue
            if token_path.stem == stem:
                matches.append(index)

        if not matches:
            unresolved.append(token)
            continue

        for match in matches:
            if match not in seen:
                seen.add(match)
                selected_indices.append(match)

    return selected_indices, unresolved


def save_projection_csv(output_path, records, coords, highlight_mask):
    fieldnames = ["image", "relative_path", "x", "y", "highlight"]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record, coord, is_highlight in zip(records, coords, highlight_mask):
            writer.writerow(
                {
                    "image": record["basename"],
                    "relative_path": record["relative_path"],
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "highlight": int(bool(is_highlight)),
                }
            )


def add_thumbnail(ax, image_path, coord, thumb_size, label):
    image = Image.open(image_path).convert("RGB")
    image = image.copy()
    image.thumbnail((thumb_size, thumb_size))
    image_array = np.asarray(image)
    image_box = OffsetImage(image_array, zoom=1.0)
    box = AnnotationBbox(
        image_box,
        coord,
        frameon=True,
        pad=0.18,
        bboxprops={"edgecolor": "black", "linewidth": 0.8, "facecolor": "white"},
    )
    ax.add_artist(box)
    ax.annotate(
        label,
        xy=coord,
        xytext=(0, -(thumb_size * 0.75)),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=6,
        color="black",
        annotation_clip=False,
    )


def visualize_embeddings(records, embeddings, output_dir, highlight_tokens, thumb_size, image_root, logger):
    coords = project_to_2d(embeddings)
    highlight_indices, unresolved = resolve_highlight_indices(records, highlight_tokens)
    highlight_mask = np.zeros(len(records), dtype=bool)
    highlight_mask[highlight_indices] = True

    if unresolved:
        logger.warning("Could not match %d highlight names: %s", len(unresolved), ", ".join(unresolved))

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=18, c="#c7c7c7", alpha=0.65, linewidths=0)

    if highlight_indices:
        ax.scatter(
            coords[highlight_indices, 0],
            coords[highlight_indices, 1],
            s=36,
            c="#f97316",
            edgecolors="black",
            linewidths=0.35,
            zorder=3,
        )

    root = Path(image_root)
    for index in highlight_indices:
        add_thumbnail(ax, records[index]["path"], coords[index], thumb_size, records[index]["basename"])

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_pad = max((x_max - x_min) * 0.08, 1.0)
    y_pad = max((y_max - y_min) * 0.08, 1.0)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("PersonViT embeddings from {} ({} images)".format(root.name, len(records)))
    ax.grid(True, color="#e5e7eb", linewidth=0.7)
    ax.set_facecolor("white")
    fig.tight_layout()

    plot_path = output_dir / "embedding_scatter.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_dir / "embedding_projection.csv"
    save_projection_csv(csv_path, records, coords, highlight_mask)

    return plot_path, csv_path, coords, highlight_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and visualize PersonViT embeddings from a folder of images")
    parser.add_argument("--config_file", default="configs/market/vit_small.yml", help="path to config file", type=str)
    parser.add_argument("--image_dir", required=True, help="folder that contains images to embed", type=str)
    parser.add_argument("--output_dir", default="folder_embedding_vis", help="directory for plots and CSV output", type=str)
    parser.add_argument("--weight", default="../pretrained/checkpoint0260.pth", help="checkpoint path", type=str)
    parser.add_argument("--batch_size", default=32, type=int, help="inference batch size")
    parser.add_argument("--thumb_size", default=56, type=int, help="thumbnail size in pixels for highlighted images")
    parser.add_argument(
        "--highlight",
        action="append",
        default=[],
        help="image name or relative path to emphasize; repeat the flag for multiple images",
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.MODEL.PRETRAIN_CHOICE = "finetune"
    cfg.TEST.WEIGHT = args.weight
    cfg.freeze()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("transreid", str(output_dir), if_train=False)
    logger.info(args)
    logger.info("Running with config:\n%s", cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)

    transform = build_test_transform()
    dataset = FolderImageDataset(args.image_dir, transform=transform)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = make_model(cfg, num_class=1, camera_num=1, view_num=1)
    load_checkpoint_for_embeddings(model, args.weight, logger)
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        logger.info("Using %d GPUs for inference", torch.cuda.device_count())
        model = nn.DataParallel(model)
    model.eval()

    all_embeddings = []
    all_paths = []
    with torch.no_grad():
        for images, paths in data_loader:
            images = images.to(device)
            features = model(images, cam_label=None, view_label=None)
            all_embeddings.append(features.cpu())
            all_paths.extend(paths)

    embeddings = torch.cat(all_embeddings, dim=0)
    embeddings = normalize_embeddings(embeddings)
    embeddings_np = embeddings.numpy()

    np.save(output_dir / "embeddings.npy", embeddings_np)
    records = build_records(all_paths, args.image_dir)
    plot_path, csv_path, coords, highlight_mask = visualize_embeddings(
        records,
        embeddings_np,
        output_dir,
        args.highlight,
        args.thumb_size,
        args.image_dir,
        logger,
    )

    logger.info("Saved embeddings to %s", output_dir / "embeddings.npy")
    logger.info("Saved 2D projection to %s", csv_path)
    logger.info("Saved visualization to %s", plot_path)
    if highlight_mask.any():
        logger.info("Highlighted %d images", int(highlight_mask.sum()))


if __name__ == "__main__":
    main()