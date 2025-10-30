import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

from vbr_dataset import vbrInterpolatedDataset, get_paths_from_scene, load_calibration


def parse_args():
    parser = argparse.ArgumentParser(description="Batch depth map estimation with ZoeDepth")
    parser.add_argument("--vbr_scene", type=str, required=True, help="Name of the scene.")
    parser.add_argument("--vbr_root", type=str, required=True, help="Root directory to dataset.")
    parser.add_argument("--pairs_file", type=str, required=True, help="Path to the pairs CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for depth maps")
    parser.add_argument("--save", type=str, choices=['depth', '3d', 'both'], default='depth', help="What to save: depth maps, 3D points, or both")
    return parser.parse_args()


def depth_to_3dpoints(depth_map, K):
    H, W = depth_map.shape
    i_range = np.arange(W)
    j_range = np.arange(H)
    u, v = np.meshgrid(i_range, j_range)
    pixels_homog = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).T  # 3 x N

    K_inv = np.linalg.inv(K)
    depth_flat = depth_map.flatten()
    pts3d = (K_inv @ (pixels_homog * depth_flat)).T.reshape(H, W, 3)
    return pts3d


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load ZoeDepth model
    conf = get_config("zoedepth", "infer", config_version="kitti")
    zoe = build_model(conf).to(DEVICE).eval()

    # Load your dataset and calibration
    vbr_scene = vbrInterpolatedDataset(args.vbr_root, args.vbr_scene)
    calib_path = get_paths_from_scene(args.vbr_root, args.vbr_scene)[-1]
    calib = load_calibration(calib_path)
    K = calib['cam_l']['K']

    # Read pairs from pairs file: list of (idx1, idx2)
    pairs = []
    with open(args.pairs_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((int(parts[0]), int(parts[1])))

    processed_images = set()

    # Iterate over pairs with progress bar
    for idx1, idx2 in tqdm(pairs, desc="Processing pairs"):
        for idx in (idx1, idx2):
            try:
                img_path = vbr_scene[idx]['image']
            except Exception as e:
                print(f"Warning: cannot get image for index {idx}: {e}")
                continue

            if img_path in processed_images:
                continue
            processed_images.add(img_path)

            # Load image with PIL for ZoeDepth
            pil_img = Image.open(img_path).convert("RGB")

            # Infer depth using ZoeDepth
            with torch.no_grad():
                depth = zoe.infer_pil(pil_img)
            # depth = depth.cpu().numpy()

            basename = os.path.splitext(os.path.basename(img_path))[0]

            if args.save in ['depth', 'both']:
                raw_depth_out_path = os.path.join(args.output_dir, basename + ".npy")
                np.save(raw_depth_out_path, depth)

            if args.save in ['3d', 'both']:
                pts3d = depth_to_3dpoints(depth, K)
                pts3d_out_path = os.path.join(args.output_dir, basename + "_3d.npy")
                np.save(pts3d_out_path, pts3d)


if __name__ == "__main__":
    main()
