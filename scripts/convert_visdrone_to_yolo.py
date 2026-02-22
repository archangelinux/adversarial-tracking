"""Convert VisDrone MOT annotations to YOLO detection format for fine-tuning.

VisDrone MOT format (per line):
    frame_id, track_id, x, y, w, h, confidence, category, truncation, occlusion

YOLO format (one .txt per image):
    class_id center_x center_y width height  (all normalized 0-1)

Extracts individual frames and labels from MOT sequences for YOLOv8
training, using the official VisDrone train/val splits.

Expected VisDrone folder structure:
    data/
        VisDrone2019-MOT-train/
            sequences/       (56 folders of .jpg frames)
            annotations/     (56 .txt annotation files)
        VisDrone2019-MOT-val/
            sequences/       (7 folders of .jpg frames)
            annotations/     (7 .txt annotation files)

Output:
    output/
        images/train/
        images/val/
        labels/train/
        labels/val/
        dataset.yaml

Usage:
    python3 convert_visdrone_to_yolo.py \
        --train-dir /tracking_ws/data/VisDrone2019-MOT-train \
        --val-dir /tracking_ws/data/VisDrone2019-MOT-val \
        --output /tracking_ws/data/yolo_dataset
"""

import argparse
import shutil
from pathlib import Path

import cv2


# VisDrone categories → our class IDs for YOLO training
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

    Returns: number of frames converted.
    """
    seq_dir = Path(seq_dir)
    annotations = parse_annotations(ann_path)

    # Get image dimensions from first frame
    first_img = sorted(seq_dir.glob('*.jpg'))[0]
    img = cv2.imread(str(first_img))
    img_h, img_w = img.shape[:2]

    count = 0

    for img_path in sorted(seq_dir.glob('*.jpg')):
        # Skip Google Drive duplicates like "0000179 (1).jpg"
        if not img_path.stem.isdigit():
            continue
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

        count += 1

    return count


def convert_split(visdrone_dir, output, split_name):
    """Convert one VisDrone split (train or val) to YOLO format."""
    visdrone_dir = Path(visdrone_dir)
    seq_dir = visdrone_dir / 'sequences'
    ann_dir = visdrone_dir / 'annotations'

    out_images = output / 'images' / split_name
    out_labels = output / 'labels' / split_name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])
    print(f'\n{split_name}: Found {len(seq_dirs)} sequences')

    total = 0
    for sd in seq_dirs:
        ann_file = ann_dir / f'{sd.name}.txt'
        if not ann_file.exists():
            print(f'  Skipping {sd.name} — no annotation file')
            continue

        print(f'  Converting {sd.name}...', end=' ', flush=True)
        count = convert_sequence(sd, ann_file, out_images, out_labels, sd.name)
        print(f'{count} frames')
        total += count

    return total


def main():
    parser = argparse.ArgumentParser(description='Convert VisDrone to YOLO format')
    parser.add_argument('--train-dir', required=True,
                        help='VisDrone2019-MOT-train directory')
    parser.add_argument('--val-dir', required=True,
                        help='VisDrone2019-MOT-val directory')
    parser.add_argument('--output', required=True,
                        help='Output directory for YOLO dataset')
    args = parser.parse_args()

    output = Path(args.output)

    # Clean previous output if it exists
    if output.exists():
        print(f'Removing previous dataset at {output}')
        shutil.rmtree(output)

    train_count = convert_split(args.train_dir, output, 'train')
    val_count = convert_split(args.val_dir, output, 'val')

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
    print(f'  Train: {train_count} images')
    print(f'  Val:   {val_count} images')
    print(f'  YAML:  {yaml_path}')
    print(f'\nTo train: python3 train_visdrone.py --data {yaml_path}')


if __name__ == '__main__':
    main()
