"""Convert VisDrone MOT annotations to YOLO detection format for fine-tuning.

VisDrone MOT format (per line):
    frame_id, track_id, x, y, w, h, confidence, category, truncation, occlusion

YOLO format (one .txt per image):
    class_id center_x center_y width height  (all normalized 0-1)

This script extracts individual frames + labels from a MOT sequence so
YOLOv8 can train on them. It creates the standard YOLO dataset structure:
    output/
        images/
            train/
            val/
        labels/
            train/
            val/
        dataset.yaml

Usage:
    python3 convert_visdrone_to_yolo.py \
        --sequences-dir /tracking_ws/data/videos \
        --annotations-dir /tracking_ws/data/annotations \
        --output /tracking_ws/data/yolo_dataset \
        --val-split 0.2
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2


# VisDrone categories → our class IDs for YOLO training
# We remap to a smaller set of classes that matter for tracking
VISDRONE_TO_CLASS = {
    1: 0,   # pedestrian → 0
    2: 0,   # people → 0 (same as pedestrian)
    3: 1,   # bicycle → 1
    4: 2,   # car → 2
    5: 3,   # van → 3
    6: 4,   # truck → 4
    7: 5,   # tricycle → 5
    8: 5,   # awning-tricycle → 5 (same as tricycle)
    9: 6,   # bus → 6
    10: 7,  # motor → 7
}

CLASS_NAMES = ['pedestrian', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motor']


def parse_annotations(ann_path):
    """Parse a VisDrone MOT annotation file.

    Returns: dict[frame_id] → list of (category, x, y, w, h)
    """
    frames = {}
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            category = int(parts[7])

            if category not in VISDRONE_TO_CLASS:
                continue
            if w <= 0 or h <= 0:
                continue

            if frame_id not in frames:
                frames[frame_id] = []
            frames[frame_id].append((category, x, y, w, h))
    return frames


def convert_sequence(seq_dir, ann_path, output_images, output_labels, prefix):
    """Convert one sequence to YOLO format.

    Returns: list of (image_dest_path, label_dest_path) pairs created.
    """
    seq_dir = Path(seq_dir)
    annotations = parse_annotations(ann_path)

    # Get image dimensions from first frame
    first_img = sorted(seq_dir.glob('*.jpg'))[0]
    img = cv2.imread(str(first_img))
    img_h, img_w = img.shape[:2]

    pairs = []

    for img_path in sorted(seq_dir.glob('*.jpg')):
        frame_id = int(img_path.stem)
        dest_name = f'{prefix}_{img_path.stem}'

        # Copy image
        img_dest = output_images / f'{dest_name}.jpg'
        shutil.copy2(img_path, img_dest)

        # Write YOLO label
        label_dest = output_labels / f'{dest_name}.txt'
        boxes = annotations.get(frame_id, [])

        with open(label_dest, 'w') as f:
            for category, x, y, w, h in boxes:
                cls_id = VISDRONE_TO_CLASS[category]
                # Convert to YOLO normalized format
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                # Clamp to [0, 1]
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))
                f.write(f'{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n')

        pairs.append((img_dest, label_dest))

    return pairs


def main():
    parser = argparse.ArgumentParser(description='Convert VisDrone to YOLO format')
    parser.add_argument('--sequences-dir', required=True,
                        help='Directory containing sequence folders')
    parser.add_argument('--annotations-dir', required=True,
                        help='Directory containing annotation .txt files')
    parser.add_argument('--output', required=True,
                        help='Output directory for YOLO dataset')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction of frames for validation (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output = Path(args.output)
    sequences_dir = Path(args.sequences_dir)
    annotations_dir = Path(args.annotations_dir)

    # Create output structure
    for split in ('train', 'val'):
        (output / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Find all sequences
    seq_dirs = sorted([d for d in sequences_dir.iterdir() if d.is_dir()])
    print(f'Found {len(seq_dirs)} sequences')

    all_pairs = []
    for seq_dir in seq_dirs:
        ann_file = annotations_dir / f'{seq_dir.name}.txt'
        if not ann_file.exists():
            print(f'  Skipping {seq_dir.name} — no annotation file')
            continue

        print(f'  Converting {seq_dir.name}...', end=' ')
        # Write to a temp location first, then split
        tmp_imgs = output / 'images' / '_tmp'
        tmp_lbls = output / 'labels' / '_tmp'
        tmp_imgs.mkdir(parents=True, exist_ok=True)
        tmp_lbls.mkdir(parents=True, exist_ok=True)

        pairs = convert_sequence(seq_dir, ann_file, tmp_imgs, tmp_lbls, seq_dir.name)
        all_pairs.extend(pairs)
        print(f'{len(pairs)} frames')

    # Shuffle and split
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * (1 - args.val_split))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    # Move files to train/val
    for pairs, split in [(train_pairs, 'train'), (val_pairs, 'val')]:
        for img_src, lbl_src in pairs:
            shutil.move(str(img_src), str(output / 'images' / split / img_src.name))
            shutil.move(str(lbl_src), str(output / 'labels' / split / lbl_src.name))

    # Clean up temp dirs
    shutil.rmtree(output / 'images' / '_tmp', ignore_errors=True)
    shutil.rmtree(output / 'labels' / '_tmp', ignore_errors=True)

    # Write dataset.yaml
    yaml_path = output / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f'path: {output.resolve()}\n')
        f.write(f'train: images/train\n')
        f.write(f'val: images/val\n')
        f.write(f'\n')
        f.write(f'nc: {len(CLASS_NAMES)}\n')
        f.write(f'names: {CLASS_NAMES}\n')

    print(f'\nDataset created:')
    print(f'  Train: {len(train_pairs)} images')
    print(f'  Val:   {len(val_pairs)} images')
    print(f'  YAML:  {yaml_path}')
    print(f'\nTo train: python3 train_visdrone.py --data {yaml_path}')


if __name__ == '__main__':
    main()
