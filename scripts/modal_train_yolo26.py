"""Fine-tune YOLO26n on VisDrone using Modal (cloud A100 GPU).

Uses the same converted dataset from the YOLOv8 training run (already on the volume).
Same hyperparameters for a fair comparison between model generations.

Setup (one time, already done if you ran modal_train.py before):
    pip install modal
    modal setup

Data upload (skip if already uploaded for YOLOv8 training):
    modal volume create visdrone-data
    modal volume put visdrone-data ~/Documents/CODE/adversarial-tracking/tracking_ws/data/VisDrone2019-MOT-train /data/VisDrone2019-MOT-train
    modal volume put visdrone-data ~/Documents/CODE/adversarial-tracking/tracking_ws/data/VisDrone2019-MOT-val /data/VisDrone2019-MOT-val

Train:
    modal run --detach scripts/modal_train_yolo26.py

Download results:
    modal volume get visdrone-data /results_yolo26/weights/best.pt ./tracking_ws/data/yolo26n_visdrone_best.pt
"""

import modal

app = modal.App("visdrone-yolo26")

volume = modal.Volume.from_name("visdrone-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics", "opencv-python-headless", "lapx")
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": volume},
    timeout=6 * 3600,
)
def train():
    import shutil
    from pathlib import Path
    import cv2

    DATASET = Path("/vol/yolo_dataset")
    RESULTS = Path("/vol/results_yolo26")

    # Dataset should already exist from YOLOv8 training run
    if not (DATASET / "dataset.yaml").exists():
        # Need to convert — same logic as modal_train.py
        LOCAL = Path("/tmp/visdrone")
        LOCAL_OUT = Path("/tmp/yolo_dataset")

        print("Dataset not found on volume. Converting VisDrone → YOLO format...")
        import subprocess
        if LOCAL.exists():
            shutil.rmtree(LOCAL)
        subprocess.run(["cp", "-r", "/vol/data", str(LOCAL)], check=True)

        VISDRONE_TO_CLASS = {
            1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 5, 9: 6, 10: 7,
        }
        CLASS_NAMES = ['pedestrian', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motor']

        def parse_annotations(ann_path):
            frames = {}
            with open(ann_path) as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue
                    fid = int(parts[0])
                    x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                    cat = int(parts[7])
                    if cat not in VISDRONE_TO_CLASS or w <= 0 or h <= 0:
                        continue
                    if fid not in frames:
                        frames[fid] = []
                    frames[fid].append((cat, x, y, w, h))
            return frames

        def convert_split(visdrone_dir, split_name):
            vd = Path(visdrone_dir)
            out_imgs = LOCAL_OUT / "images" / split_name
            out_lbls = LOCAL_OUT / "labels" / split_name
            out_imgs.mkdir(parents=True, exist_ok=True)
            out_lbls.mkdir(parents=True, exist_ok=True)

            seq_dirs = sorted([d for d in (vd / "sequences").iterdir() if d.is_dir()])
            print(f"  {split_name}: {len(seq_dirs)} sequences")
            total = 0
            for sd in seq_dirs:
                ann = vd / "annotations" / f"{sd.name}.txt"
                if not ann.exists():
                    continue
                anns = parse_annotations(ann)
                first = sorted(sd.glob("*.jpg"))[0]
                img = cv2.imread(str(first))
                ih, iw = img.shape[:2]
                count = 0
                for ip in sorted(sd.glob("*.jpg")):
                    if not ip.stem.isdigit():
                        continue
                    fid = int(ip.stem)
                    dn = f"{sd.name}_{ip.stem}"
                    shutil.copy2(ip, out_imgs / f"{dn}.jpg")
                    boxes = anns.get(fid, [])
                    with open(out_lbls / f"{dn}.txt", "w") as f:
                        for cat, x, y, w, h in boxes:
                            cid = VISDRONE_TO_CLASS[cat]
                            cx = max(0, min(1, (x + w/2) / iw))
                            cy = max(0, min(1, (y + h/2) / ih))
                            nw = max(0, min(1, w / iw))
                            nh = max(0, min(1, h / ih))
                            f.write(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    count += 1
                total += count
                print(f"    {sd.name}: {count} frames")
            return total

        train_count = convert_split(str(LOCAL / "VisDrone2019-MOT-train"), "train")
        val_count = convert_split(str(LOCAL / "VisDrone2019-MOT-val"), "val")

        if DATASET.exists():
            shutil.rmtree(DATASET)
        shutil.copytree(LOCAL_OUT, DATASET)

        with open(DATASET / "dataset.yaml", "w") as f:
            f.write(f"path: {DATASET}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f"\nnc: 8\n")
            f.write(f"names: {CLASS_NAMES}\n")

        volume.commit()
        print(f"  Train: {train_count} | Val: {val_count}\n")
        shutil.rmtree(LOCAL, ignore_errors=True)
        shutil.rmtree(LOCAL_OUT, ignore_errors=True)
    else:
        print("Dataset already converted on volume, skipping conversion.")

    from ultralytics import YOLO

    last_pt = RESULTS / "weights" / "last.pt"

    if last_pt.exists():
        print(f"Resuming training from {last_pt}")
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        print("Starting fresh training from yolo26n.pt")
        model = YOLO("yolo26n.pt")
        model.train(
            data=str(DATASET / "dataset.yaml"),
            epochs=50,
            imgsz=640,
            batch=16,
            device=0,
            name="",
            project=str(RESULTS),
            save_period=5,
            exist_ok=True,
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3,
            close_mosaic=10,
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

    volume.commit()
    print("\nDone! Results saved to volume.")
    print("Download: modal volume get visdrone-data /results_yolo26/weights/best.pt ./tracking_ws/data/yolo26n_visdrone_best.pt")


@app.local_entrypoint()
def main():
    train.remote()
