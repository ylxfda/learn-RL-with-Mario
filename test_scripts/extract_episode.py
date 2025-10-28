#!/usr/bin/env python3
"""
Extract NPZ episode file content to a folder with visualizations

This script extracts episode data from NPZ files saved by the DreamerV3 agent:
- Creates a subfolder with the same basename as the NPZ file
- Saves observation images as an animated GIF
- Saves other data (rewards, actions, etc.) as CSV files
"""

import argparse
import pathlib
import numpy as np
from PIL import Image
import sys
import csv


def extract_episode(npz_path: pathlib.Path, output_dir: pathlib.Path = None):
    """
    Extract episode data from NPZ file

    Args:
        npz_path: Path to the NPZ file
        output_dir: Output directory (defaults to folder next to NPZ file)
    """
    npz_path = pathlib.Path(npz_path)

    if not npz_path.exists():
        print(f"Error: File not found: {npz_path}")
        return False

    # Load NPZ file
    print(f"Loading {npz_path}...")
    with np.load(npz_path) as data:
        episode = {k: data[k] for k in data.keys()}

    # Create output directory (same basename as NPZ file)
    if output_dir is None:
        output_dir = npz_path.parent / npz_path.stem
    else:
        output_dir = pathlib.Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting to {output_dir}/")

    # Print episode info
    print(f"\nEpisode info:")
    print(f"  Keys: {list(episode.keys())}")
    if 'reward' in episode:
        print(f"  Length: {len(episode['reward'])} steps")
        print(f"  Total reward: {episode['reward'].sum():.2f}")

    # Process each key
    for key, value in episode.items():
        print(f"\nProcessing '{key}': shape={value.shape}, dtype={value.dtype}")

        # Handle image data (3D or 4D arrays with values in [0, 255])
        if key == 'image' and len(value.shape) >= 3:
            # Image data: create GIF
            print(f"  Creating GIF animation...")
            num_frames = len(value)
            create_gif(value, output_dir / f"{key}_{num_frames}.gif")

        else:
            # Other data: save as CSV
            print(f"  Saving as CSV...")
            save_as_csv(key, value, output_dir / f"{key}.csv")

    print(f"\n✓ Extraction complete: {output_dir}/")
    return True


def create_gif(images: np.ndarray, output_path: pathlib.Path, duration: int = 100):
    """
    Create animated GIF from image sequence

    Args:
        images: Image array, shape (T, H, W, C) or (T, H, W)
        output_path: Output GIF path
        duration: Frame duration in milliseconds
    """
    # Normalize images to [0, 255] uint8
    if images.dtype == np.float32 or images.dtype == np.float64:
        # Assume normalized [0, 1]
        images = (images * 255).astype(np.uint8)
    else:
        images = images.astype(np.uint8)

    # Convert to PIL images
    frames = []
    for i in range(len(images)):
        img = images[i]

        # Handle grayscale
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)

        # Convert to PIL
        pil_img = Image.fromarray(img)
        frames.append(pil_img)

    # Save as GIF
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"    Saved {len(frames)} frames to {output_path.name}")


def save_as_csv(name: str, data: np.ndarray, output_path: pathlib.Path):
    """
    Save array data as CSV

    Args:
        name: Data name
        data: Numpy array
        output_path: Output CSV path
    """
    # Prepare data and column names
    if len(data.shape) == 1:
        # 1D array: single column
        columns = ['timestep', name]
        rows = [[i, data[i]] for i in range(len(data))]

    elif len(data.shape) == 2:
        # 2D array: multiple columns
        if data.shape[1] == 1:
            columns = ['timestep', name]
            rows = [[i, data[i, 0]] for i in range(len(data))]
        else:
            # Create columns like "action_0", "action_1", etc.
            columns = ['timestep'] + [f"{name}_{j}" for j in range(data.shape[1])]
            rows = [[i] + data[i].tolist() for i in range(len(data))]

    else:
        # Higher dimensional: flatten each timestep
        print(f"    Warning: {len(data.shape)}D array, flattening...")
        reshaped = data.reshape(len(data), -1)
        columns = ['timestep'] + [f"{name}_{j}" for j in range(reshaped.shape[1])]
        rows = [[i] + reshaped[i].tolist() for i in range(len(reshaped))]

    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    print(f"    Saved {len(rows)} rows to {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract episode NPZ file to folder with GIF and CSV files"
    )
    parser.add_argument(
        "npz_file",
        type=str,
        help="Path to NPZ episode file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: folder with same name as NPZ file)"
    )

    args = parser.parse_args()

    success = extract_episode(args.npz_file, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
