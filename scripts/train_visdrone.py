"""Fine-tune YOLOv8 on VisDrone drone footage.

Adapts a COCO-pretrained model to the aerial domain using transfer learning
with a low learning rate (0.001) to preserve early-layer features while
retraining the detection head for top-down perspectives.

Supports checkpoint resumption for interrupted training runs.

Usage:
    python3 train_visdrone.py --data dataset.yaml --epochs 50 --project /output/dir
    python3 train_visdrone.py --resume /output/dir/visdrone/weights/last.pt --epochs 50
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Fine-tune YOLOv8 on VisDrone')
    parser.add_argument('--data', default=None,
                        help='Path to dataset.yaml (from convert_visdrone_to_yolo.py)')
    parser.add_argument('--model', default='yolov8n.pt',
                        help='Base model to fine-tune (default: yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Training image size (default: 640)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16, reduce if OOM)')
    parser.add_argument('--device', default='0',
                        help='Device: cpu, 0 (GPU), or mps (Apple Silicon)')
    parser.add_argument('--name', default='visdrone',
                        help='Experiment name for saving results')
    parser.add_argument('--project', default='./runs/detect',
                        help='Where to save results (use persistent storage!)')
    parser.add_argument('--save-period', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--resume', default=None,
                        help='Path to last.pt to resume interrupted training')
    args = parser.parse_args()

    # Resume from checkpoint
    if args.resume:
        print(f'Resuming training from: {args.resume}')
        model = YOLO(args.resume)
        results = model.train(
            resume=True,
        )
    else:
        if not args.data:
            parser.error('--data is required when not resuming')

        print(f'Fine-tuning {args.model} on VisDrone data')
        print(f'  Dataset:     {args.data}')
        print(f'  Epochs:      {args.epochs}')
        print(f'  ImgSize:     {args.imgsz}')
        print(f'  Batch:       {args.batch}')
        print(f'  Device:      {args.device}')
        print(f'  Save every:  {args.save_period} epochs')
        print(f'  Saving to:   {args.project}/{args.name}/')
        print()

        model = YOLO(args.model)

        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            name=args.name,
            project=args.project,
            save_period=args.save_period,
            exist_ok=True,
            # Fine-tuning specific settings:
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3,
            close_mosaic=10,
            # Augmentation:
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            scale=0.5,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            verbose=True,
        )

    print('\n' + '=' * 60)
    print('Training complete!')
    print(f'Best model: {args.project}/{args.name}/weights/best.pt')
    print(f'Last model: {args.project}/{args.name}/weights/last.pt')
    print('=' * 60)


if __name__ == '__main__':
    main()
