"""Fine-tune YOLOv8 on VisDrone using Modal (cloud A100 GPU).

Setup (one time):
    pip install modal
    modal setup

Upload data (one time, ~20-30 min):
    modal volume create visdrone-data
    modal volume put visdrone-data ~/Documents/CODE/adversarial-tracking/tracking_ws/data/VisDrone2019-MOT-train /data/VisDrone2019-MOT-train
    modal volume put visdrone-data ~/Documents/CODE/adversarial-tracking/tracking_ws/data/VisDrone2019-MOT-val /data/VisDrone2019-MOT-val

Train:
    modal run --detach scripts/modal_train.py

Download results:
    modal volume get visdrone-data /results/weights/best.pt ./tracking_ws/data/yolov8n_visdrone_best.pt
"""

import modal

# --- Modal setup ---
app = modal.App("visdrone-training")

# Persistent volume for data + results
volume = modal.Volume.from_name("visdrone-data", create_if_missing=True)

# Container image with everything we need
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics", "opencv-python-headless", "lapx")
)


@app.function(
    image=image,
    gpu="A100",
    volumes={"/vol": volume},
    timeout=6 * 3600,  # 6 hour max
)
def train():
    import shutil
    import subprocess
    from pathlib import Path
    import cv2

    # Copy data from network volume to local SSD for faster I/O
    LOCAL = Path("/tmp/visdrone")
    OUTPUT = Path("/vol/yolo_dataset")

    if (OUTPUT / "dataset.yaml").exists():
        print("Dataset already converted, skipping.")
    else:
        print("Copying data from volume to local SSD (this is much faster)...")
        if LOCAL.exists():
            shutil.rmtree(LOCAL)
        subprocess.run(["cp", "-r", "/vol/data", str(LOCAL)], check=True)
        print("Copy complete. Starting conversion...\n")

        # Convert VisDrone → YOLO format
        VISDRONE_TO_CLASS = {
            1: 0, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 5, 9: 6, 10: 7,
        }
        CLASS_NAMES = ['pedestrian', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'bus', 'motor']

        # Convert to local /tmp first, then move to volume
        LOCAL_OUT = Path("/tmp/yolo_dataset")

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

        # Write dataset.yaml pointing to the volume path (where training reads from)
        with open(LOCAL_OUT / "dataset.yaml", "w") as f:
            f.write(f"path: {OUTPUT}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f"\nnc: {len(VISDRONE_TO_CLASS.values().__class__(VISDRONE_TO_CLASS.values()))}\n")
            f.write(f"names: {CLASS_NAMES}\n")

        print(f"\n  Train: {train_count} | Val: {val_count}")
        print("  Moving converted dataset to volume...")

        # Move converted dataset to persistent volume
        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        shutil.copytree(LOCAL_OUT, OUTPUT)

        # Fix dataset.yaml path
        with open(OUTPUT / "dataset.yaml", "w") as f:
            f.write(f"path: {OUTPUT}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write(f"\nnc: 8\n")
            f.write(f"names: {CLASS_NAMES}\n")

        volume.commit()
        print("  Dataset saved to volume.\n")

        # Clean up local temp
        shutil.rmtree(LOCAL, ignore_errors=True)
        shutil.rmtree(LOCAL_OUT, ignore_errors=True)

    # Train
    from ultralytics import YOLO

    RESULTS = "/vol/results"
    last_pt = Path(f"{RESULTS}/weights/last.pt")

    if last_pt.exists():
        print(f"Resuming training from {last_pt}")
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        print("Starting training from yolov8n.pt")
        model = YOLO("yolov8n.pt")
        model.train(
            data=str(OUTPUT / "dataset.yaml"),
            epochs=50,
            imgsz=640,
            batch=16,
            device=0,
            name="",
            project=RESULTS,
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
    print("Download with: modal volume get visdrone-data /results/weights/best.pt ./best.pt")


@app.local_entrypoint()
def main():
    train.remote()
