"""Fine-tune YOLOv8 on VisDrone data.

This takes a pretrained YOLO model (trained on COCO — ground-level photos)
and fine-tunes it on VisDrone drone footage. The result is a model that
actually understands what objects look like from above.

WHY THIS MATTERS:
    yolov8n.pt was trained on COCO — photos taken at eye level with large,
    clear objects. VisDrone is shot from drones at altitude — objects are tiny,
    viewed from above, and densely packed. The pretrained model has never seen
    this perspective, so it struggles. Fine-tuning teaches it the drone view.

WHAT FINE-TUNING DOES:
    - Starts from pretrained weights (doesn't learn from scratch)
    - The early layers (edge/texture detectors) stay mostly the same
    - The later layers (object classifiers) adapt to drone-perspective objects
    - Training typically takes 30-100 epochs, ~1-4 hours on GPU

Usage (inside Docker):
    # First, convert VisDrone annotations:
    python3 /tracking_ws/scripts/convert_visdrone_to_yolo.py \
        --sequences-dir /tracking_ws/data/videos \
        --annotations-dir /tracking_ws/data/annotations \
        --output /tracking_ws/data/yolo_dataset

    # Then fine-tune:
    python3 /tracking_ws/scripts/train_visdrone.py \
        --data /tracking_ws/data/yolo_dataset/dataset.yaml \
        --model yolov8n.pt \
        --epochs 50

    # The best model will be saved to runs/detect/visdrone/weights/best.pt
    # Use it in the framework:
    ros2 launch tracking_adversarial baseline.launch.py \
        model:=runs/detect/visdrone/weights/best.pt
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Fine-tune YOLOv8 on VisDrone')
    parser.add_argument('--data', required=True,
                        help='Path to dataset.yaml (from convert_visdrone_to_yolo.py)')
    parser.add_argument('--model', default='yolov8n.pt',
                        help='Base model to fine-tune (default: yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Training image size (default: 640)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16, reduce if OOM)')
    parser.add_argument('--device', default='cpu',
                        help='Device: cpu, 0 (GPU), or mps (Apple Silicon)')
    parser.add_argument('--name', default='visdrone',
                        help='Experiment name for saving results')
    args = parser.parse_args()

    print(f'Fine-tuning {args.model} on VisDrone data')
    print(f'  Dataset: {args.data}')
    print(f'  Epochs:  {args.epochs}')
    print(f'  ImgSize: {args.imgsz}')
    print(f'  Batch:   {args.batch}')
    print(f'  Device:  {args.device}')
    print()

    # Load pretrained model
    model = YOLO(args.model)

    # Fine-tune
    # Key parameters explained:
    #   - epochs: how many times to see the full dataset
    #   - imgsz: input resolution (higher = better for small objects, but slower)
    #   - batch: images per training step (lower if you run out of memory)
    #   - lr0: initial learning rate (0.01 is default, lower for fine-tuning)
    #   - lrf: final learning rate factor (lr decays to lr0 * lrf)
    #   - warmup_epochs: gradually increase lr at start to avoid destroying pretrained weights
    #   - freeze: freeze early layers so they keep COCO features (optional)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        # Fine-tuning specific settings:
        lr0=0.001,        # Lower than default (0.01) — we're fine-tuning, not training from scratch
        lrf=0.01,         # LR decays to 0.001 * 0.01 = 0.00001 by end
        warmup_epochs=3,  # Gentle warmup to not destroy pretrained features
        close_mosaic=10,  # Disable mosaic augmentation for last 10 epochs (stabilizes)
        # Augmentation (helps generalize):
        hsv_h=0.015,      # Hue variation
        hsv_s=0.7,        # Saturation variation
        hsv_v=0.4,        # Brightness variation
        degrees=10.0,     # Rotation (drones tilt)
        scale=0.5,        # Scale variation (altitude changes)
        flipud=0.5,       # Vertical flip (valid for aerial views)
        fliplr=0.5,       # Horizontal flip
        mosaic=1.0,       # Mosaic augmentation (combines 4 images)
        mixup=0.1,        # MixUp augmentation (blends images)
        verbose=True,
    )

    # Print results
    print('\n' + '='*60)
    print('Training complete!')
    print(f'Best model saved to: runs/detect/{args.name}/weights/best.pt')
    print(f'Last model saved to: runs/detect/{args.name}/weights/last.pt')
    print()
    print('To use your fine-tuned model in the framework:')
    print(f'  ros2 launch tracking_adversarial baseline.launch.py \\')
    print(f'      model:=runs/detect/{args.name}/weights/best.pt')
    print()
    print('To benchmark it:')
    print(f'  python3 /tracking_ws/scripts/benchmark.py \\')
    print(f'      --model runs/detect/{args.name}/weights/best.pt \\')
    print(f'      --sequence <your_sequence> --annotations <your_annotations>')
    print('='*60)


if __name__ == '__main__':
    main()
