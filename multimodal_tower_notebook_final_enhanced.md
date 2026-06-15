# Multimodal Structural and Thermal Anomaly Detection for Cell Tower Inspection

## Project layout

Place the notebook at the project root.

```text
project_root/
├── multimodal_tower_demo.ipynb
├── tower_inspection_dataset/
│   ├── dataset.yaml or data.yaml
│   ├── train/images/
│   ├── train/labels/
│   ├── val/images/
│   ├── val/labels/
│   ├── test/images/
│   └── test/labels/
├── cell_tower_thermal/
│   ├── dataset.yaml
│   ├── train/images/
│   ├── train/labels/
│   ├── val/images/
│   ├── val/labels/
│   ├── test/images/
│   └── test/labels/
├── telemetry_dataset/
│   ├── dataset.yaml
│   ├── schema.json
│   ├── mission_index.csv
│   ├── train/telemetry/
│   ├── train/annotations/
│   ├── train/mission_metadata/
│   ├── val/telemetry/
│   ├── val/annotations/
│   ├── val/mission_metadata/
│   ├── test/telemetry/
│   ├── test/annotations/
│   └── test/mission_metadata/
└── demo_inputs/
    ├── sample_rgb_image.png
    ├── sample_thermal_image.jpg
    └── sample_inspection_video.mp4
```

## Cell 1 — Markdown

```markdown
# Multimodal Structural and Thermal Anomaly Detection for Cell Tower Inspection

## Abstract
This notebook presents a multimodal inspection framework for telecom tower analysis using RGB imagery, thermal imagery, and telemetry. The study addresses three coupled tasks: structural defect detection from visible imagery, hotspot and component analysis from thermal imagery, and context-aware anomaly scoring from mission telemetry. The final stage performs event-level late fusion to generate geo-tagged inspection alerts with severity labels.

## Problem context
Inspection of telecom towers requires identification of both visible structural degradation and temperature-related faults. RGB imagery captures corrosion, cracks, missing bolts, paint degradation, and member deformation. Thermal imagery captures localized hotspots and component context. Telemetry provides mission-state information such as tower proximity, platform stability, navigation quality, and environmental conditions. The objective is to integrate these modalities into a unified inspection decision.

## Study goals
1. Train an RGB detector for structural defect localization.
2. Train a thermal detector for hotspot and component localization.
3. Derive a telemetry-based contextual risk model.
4. Define a decision-level multimodal fusion rule.
5. Evaluate trained models on image or video input.
6. Construct a mission replay section for fused event interpretation.
```

## Cell 2 — Code

```python
# Run once in a fresh environment
# !pip install -q ultralytics opencv-python pillow seaborn scikit-learn plotly pyyaml tqdm
```

## Cell 3 — Code

```python
import os
import json
import math
import random
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm
import cv2
import yaml

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ultralytics import YOLO

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
```

## Cell 4 — Markdown

```markdown
## 1. Experimental design

The notebook is organized into four stages. First, the RGB dataset is used to train an object detector for visible structural anomalies. Second, the thermal dataset is used to train a detector for hotspot-related anomalies and tower components. Third, telemetry records are transformed into contextual descriptors and a risk model. Fourth, the modality-specific outputs are merged by `mission_id` and `frame_id`, and a fused risk score is computed.

A late-fusion formulation is used because the modalities differ in representation and supervision. RGB and thermal branches are spatial detectors, while telemetry is sequential tabular data. Decision-level fusion preserves interpretability and allows each branch to be evaluated independently before aggregation.
```

## Cell 5 — Code

```python
ROOT = Path(".").resolve()
RGB_DIR = ROOT / "tower_inspection_dataset"
THERMAL_DIR = ROOT / "cell_tower_thermal"
TEL_DIR = ROOT / "telemetry_dataset"
DEMO_DIR = ROOT / "demo_inputs"
OUT_DIR = ROOT / "notebook_outputs"
OUT_DIR.mkdir(exist_ok=True)

print("Project root:", ROOT)
print("RGB dataset found:", RGB_DIR.exists())
print("Thermal dataset found:", THERMAL_DIR.exists())
print("Telemetry dataset found:", TEL_DIR.exists())
print("Demo input directory found:", DEMO_DIR.exists())
```

## Cell 6 — Code

```python
def find_yaml(dataset_dir):
    for name in ["dataset.yaml", "data.yaml"]:
        p = dataset_dir / name
        if p.exists():
            return p
    return None

rgb_yaml = find_yaml(RGB_DIR)
thermal_yaml = find_yaml(THERMAL_DIR)
telemetry_yaml = find_yaml(TEL_DIR)

print("RGB yaml:", rgb_yaml)
print("Thermal yaml:", thermal_yaml)
print("Telemetry yaml:", telemetry_yaml)
```

## Cell 7 — Code

```python
def load_yaml(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

rgb_cfg = load_yaml(rgb_yaml)
thermal_cfg = load_yaml(thermal_yaml)
telemetry_cfg = load_yaml(telemetry_yaml)
```

## Cell 8 — Markdown

```markdown
## 2. Dataset specification

The RGB dataset contains six structural defect classes: crack, rust, corrosion, missing bolt, paint damage, and structural bend. The thermal dataset contains both object-context classes and heat-anomaly classes. The telemetry dataset provides frame-linked navigation, attitude, IMU, power, environmental, and anomaly-related metadata.

Both image datasets are stored in YOLO normalized bounding-box format.
```

## Cell 9 — Code

```python
RGB_CLASSES = rgb_cfg.get("names", ['crack', 'rust', 'corrosion', 'missing_bolt', 'paint_damage', 'structural_bend'])
THERMAL_CLASSES = thermal_cfg.get("names", ['tower_structure', 'antenna_panel', 'cable_harness', 'hotspot_critical', 'hotspot_moderate', 'hotspot_minor', 'connector_joint', 'equipment_cabinet'])
TEL_CLASSES = {
    0: 'hotspot_critical',
    1: 'hotspot_moderate',
    2: 'hotspot_minor',
    3: 'antenna_misalignment',
    4: 'cable_fault',
    5: 'connector_failure',
    6: 'corrosion',
    7: 'structural_crack',
    8: 'no_anomaly'
}
```

## Cell 10 — Code

```python
def split_image_label_counts(base_dir):
    rows = []
    for split in ["train", "val", "test"]:
        img_dir = base_dir / split / "images"
        lbl_dir = base_dir / split / "labels"
        rows.append({
            "split": split,
            "images": len(list(img_dir.glob("*"))) if img_dir.exists() else 0,
            "labels": len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0,
        })
    return pd.DataFrame(rows)

rgb_split_df = split_image_label_counts(RGB_DIR)
thermal_split_df = split_image_label_counts(THERMAL_DIR)

display(rgb_split_df)
display(thermal_split_df)
```

## Cell 11 — Code

```python
def telemetry_counts(tel_dir):
    rows = []
    for split in ["train", "val", "test"]:
        rows.append({
            "split": split,
            "telemetry_csv": len(list((tel_dir / split / "telemetry").glob("*.csv"))) if (tel_dir / split / "telemetry").exists() else 0,
            "annotation_json": len(list((tel_dir / split / "annotations").glob("*.json"))) if (tel_dir / split / "annotations").exists() else 0,
            "metadata_json": len(list((tel_dir / split / "mission_metadata").glob("*.json"))) if (tel_dir / split / "mission_metadata").exists() else 0,
        })
    return pd.DataFrame(rows)

display(telemetry_counts(TEL_DIR))
```

## Cell 12 — Markdown

```markdown
## 3. Annotation parsing

For each object annotation, the YOLO representation is

\[
(class\_id, c_x, c_y, w, h)
\]

where the center coordinates and side lengths are normalized to the image dimensions. These labels are expanded into tabular form for distribution analysis, visualization, and quality checks.
```

## Cell 13 — Code

```python
def parse_yolo_dataset(dataset_dir, class_names):
    rows = []
    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not lbl_dir.exists():
            continue
        for txt_path in lbl_dir.glob("*.txt"):
            stem = txt_path.stem
            img_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            for idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, w, h = parts
                cls_id = int(float(cls_id))
                rows.append({
                    "split": split,
                    "image_stem": stem,
                    "image_path": str(img_path) if img_path else None,
                    "label_path": str(txt_path),
                    "object_id": idx,
                    "class_id": cls_id,
                    "class_name": class_names[cls_id],
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                })
    return pd.DataFrame(rows)

rgb_labels_df = parse_yolo_dataset(RGB_DIR, RGB_CLASSES)
thermal_labels_df = parse_yolo_dataset(THERMAL_DIR, THERMAL_CLASSES)

print("RGB labels:", rgb_labels_df.shape)
print("Thermal labels:", thermal_labels_df.shape)
```

## Cell 14 — Code

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.countplot(data=rgb_labels_df, x="class_name", order=rgb_labels_df["class_name"].value_counts().index, ax=axes[0], palette="viridis")
axes[0].set_title("RGB defect distribution")
axes[0].tick_params(axis="x", rotation=45)

sns.countplot(data=thermal_labels_df, x="class_name", order=thermal_labels_df["class_name"].value_counts().index, ax=axes[1], palette="magma")
axes[1].set_title("Thermal class distribution")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
```

## Cell 15 — Code

```python
def draw_boxes(image_path, df_rows):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    canvas = img.copy()
    for _, row in df_rows.iterrows():
        x1 = int((row.cx - row.w / 2) * W)
        y1 = int((row.cy - row.h / 2) * H)
        x2 = int((row.cx + row.w / 2) * W)
        y2 = int((row.cy + row.h / 2) * H)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, row.class_name, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return canvas

rgb_sample = rgb_labels_df["image_path"].dropna().iloc[0]
rgb_sample_rows = rgb_labels_df[rgb_labels_df["image_path"] == rgb_sample]
plt.figure(figsize=(8, 8))
plt.imshow(draw_boxes(rgb_sample, rgb_sample_rows))
plt.title("RGB annotation example")
plt.axis("off")
plt.show()
```

## Cell 16 — Code

```python
thermal_sample = thermal_labels_df["image_path"].dropna().iloc[0]
thermal_sample_rows = thermal_labels_df[thermal_labels_df["image_path"] == thermal_sample]
plt.figure(figsize=(8, 8))
plt.imshow(draw_boxes(thermal_sample, thermal_sample_rows))
plt.title("Thermal annotation example")
plt.axis("off")
plt.show()
```

## Cell 17 — Markdown

```markdown
## 4. RGB detector training

The RGB branch is trained as a structural defect detector initialized from a pretrained YOLO backbone. The training split is used to optimize detector weights, the validation split is used to monitor generalization and guide model selection, and the test split is reserved for post-training evaluation. This separation reduces overfitting risk and preserves an unbiased estimate of model quality.

### Training, validation, and test protocol

Let the full dataset be denoted by \(\mathcal{D}\). The split is written as

\[
\mathcal{D} = \mathcal{D}_{train} \cup \mathcal{D}_{val} \cup \mathcal{D}_{test}\n\]

where \(\mathcal{D}_{train}\) is used for parameter learning, \(\mathcal{D}_{val}\) is used for validation during training, and \(\mathcal{D}_{test}\) is used only for final evaluation.
```

## Cell 18 — Code

```python
rgb_model = YOLO("yolov8n.pt")
```

## Cell 19 — Code

```python
rgb_train_results = rgb_model.train(
    data=str(rgb_yaml),
    epochs=50,
    imgsz=640,
    batch=16,
    patience=15,
    optimizer="auto",
    lr0=0.001,
    project=str(OUT_DIR / "rgb_runs"),
    name="rgb_detector",
    pretrained=True,
    verbose=True
)
```

## Cell 20 — Code

```python
rgb_val_metrics = rgb_model.val()
print("RGB mAP50-95:", rgb_val_metrics.box.map)
print("RGB mAP50:", rgb_val_metrics.box.map50)
print("RGB mAP75:", rgb_val_metrics.box.map75)
```

## Cell 20A — Code

```python
# Training diagnostics for the RGB detector
rgb_results_csv = list((OUT_DIR / "rgb_runs").glob("**/results.csv"))
if rgb_results_csv:
    rgb_hist = pd.read_csv(rgb_results_csv[0])
    display(rgb_hist.head())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(rgb_hist['epoch'], rgb_hist['train/box_loss'], label='train box loss')
    axes[0].plot(rgb_hist['epoch'], rgb_hist['val/box_loss'], label='val box loss')
    axes[0].set_title('RGB box loss')
    axes[0].legend()

    axes[1].plot(rgb_hist['epoch'], rgb_hist['metrics/precision(B)'], label='precision')
    axes[1].plot(rgb_hist['epoch'], rgb_hist['metrics/recall(B)'], label='recall')
    axes[1].set_title('RGB precision and recall')
    axes[1].legend()

    axes[2].plot(rgb_hist['epoch'], rgb_hist['metrics/mAP50(B)'], label='mAP50')
    axes[2].plot(rgb_hist['epoch'], rgb_hist['metrics/mAP50-95(B)'], label='mAP50-95')
    axes[2].set_title('RGB validation mAP')
    axes[2].legend()
    plt.tight_layout()
    plt.show()
```

## Cell 21 — Markdown

```markdown
### Detection metrics

Object detection quality is measured with precision, recall, intersection over union, average precision, and mean average precision. Precision measures how many predicted detections are correct. Recall measures how many true objects are recovered by the detector.

Precision and recall are defined by

\[
Precision = \frac{TP}{TP + FP}
\]

\[
Recall = \frac{TP}{TP + FN}
\]

The intersection over union for one predicted box and one ground-truth box is

\[
IoU = \frac{|B_p \cap B_g|}{|B_p \cup B_g|}
\]

Average precision is the area under the precision-recall curve for one class, and mean average precision is the mean of class-wise average precision values. In detector reports, \(mAP50\) uses an IoU threshold of 0.50, while \(mAP50	ext{-}95\) averages performance over thresholds from 0.50 to 0.95.
```

## Cell 22 — Code

```python
rgb_best_weights = list((OUT_DIR / "rgb_runs").glob("**/weights/best.pt"))[0]
rgb_best_model = YOLO(str(rgb_best_weights))

rgb_test_images = rgb_labels_df.query("split == 'test'")["image_path"].dropna().drop_duplicates().sample(min(4, rgb_labels_df.query("split == 'test'")["image_path"].nunique()), random_state=42).tolist()
rgb_test_preds = rgb_best_model.predict(rgb_test_images, conf=0.25, verbose=False)
```

## Cell 23 — Code

```python
fig, axes = plt.subplots(1, len(rgb_test_preds), figsize=(5 * len(rgb_test_preds), 5))
if len(rgb_test_preds) == 1:
    axes = [axes]
for ax, pred in zip(axes, rgb_test_preds):
    ax.imshow(pred.plot()[..., ::-1])
    ax.axis("off")
plt.suptitle("RGB detector predictions on held-out images")
plt.tight_layout()
plt.show()
```

## Cell 24 — Markdown

```markdown
## 5. Thermal detector training

The thermal branch is trained to identify both hotspot classes and component classes. This branch complements the RGB detector by identifying heat-related anomalies that may not be visible in standard imagery. Validation metrics are tracked separately because thermal object appearance, contrast, and class structure differ from visible-spectrum inputs.
```

## Cell 25 — Code

```python
thermal_model = YOLO("yolov8n.pt")
```

## Cell 26 — Code

```python
thermal_train_results = thermal_model.train(
    data=str(thermal_yaml),
    epochs=50,
    imgsz=640,
    batch=16,
    patience=15,
    optimizer="auto",
    lr0=0.001,
    project=str(OUT_DIR / "thermal_runs"),
    name="thermal_detector",
    pretrained=True,
    verbose=True
)
```

## Cell 27 — Code

```python
thermal_val_metrics = thermal_model.val()
print("Thermal mAP50-95:", thermal_val_metrics.box.map)
print("Thermal mAP50:", thermal_val_metrics.box.map50)
print("Thermal mAP75:", thermal_val_metrics.box.map75)
```

## Cell 27A — Code

```python
# Training diagnostics for the thermal detector
thermal_results_csv = list((OUT_DIR / "thermal_runs").glob("**/results.csv"))
if thermal_results_csv:
    thermal_hist = pd.read_csv(thermal_results_csv[0])
    display(thermal_hist.head())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(thermal_hist['epoch'], thermal_hist['train/box_loss'], label='train box loss')
    axes[0].plot(thermal_hist['epoch'], thermal_hist['val/box_loss'], label='val box loss')
    axes[0].set_title('Thermal box loss')
    axes[0].legend()

    axes[1].plot(thermal_hist['epoch'], thermal_hist['metrics/precision(B)'], label='precision')
    axes[1].plot(thermal_hist['epoch'], thermal_hist['metrics/recall(B)'], label='recall')
    axes[1].set_title('Thermal precision and recall')
    axes[1].legend()

    axes[2].plot(thermal_hist['epoch'], thermal_hist['metrics/mAP50(B)'], label='mAP50')
    axes[2].plot(thermal_hist['epoch'], thermal_hist['metrics/mAP50-95(B)'], label='mAP50-95')
    axes[2].set_title('Thermal validation mAP')
    axes[2].legend()
    plt.tight_layout()
    plt.show()
```

## Cell 28 — Code

```python
thermal_best_weights = list((OUT_DIR / "thermal_runs").glob("**/weights/best.pt"))[0]
thermal_best_model = YOLO(str(thermal_best_weights))

thermal_test_images = thermal_labels_df.query("split == 'test'")["image_path"].dropna().drop_duplicates().sample(min(4, thermal_labels_df.query("split == 'test'")["image_path"].nunique()), random_state=42).tolist()
thermal_test_preds = thermal_best_model.predict(thermal_test_images, conf=0.25, verbose=False)
```

## Cell 29 — Code

```python
fig, axes = plt.subplots(1, len(thermal_test_preds), figsize=(5 * len(thermal_test_preds), 5))
if len(thermal_test_preds) == 1:
    axes = [axes]
for ax, pred in zip(axes, thermal_test_preds):
    ax.imshow(pred.plot()[..., ::-1])
    ax.axis("off")
plt.suptitle("Thermal detector predictions on held-out images")
plt.tight_layout()
plt.show()
```

## Cell 30 — Markdown

```markdown
## 6. Standalone image and video inference

The trained detectors can be evaluated on a single image or on a stored video file. This section supports qualitative testing without requiring a live source.

### Inference logic

Given an input frame \(x_t\), the detector produces a set of candidate detections

\[
\mathcal{Y}_t = \{(b_i, c_i, s_i)\}_{i=1}^{N_t}
\]

where \(b_i\) is the bounding box, \(c_i\) is the predicted class, and \(s_i\) is the confidence score. For event-level fusion, the top-confidence detection is retained as the compact frame descriptor.
```

## Cell 31 — Code

```python
RGB_TEST_IMAGE = DEMO_DIR / "sample_rgb_image.png"
THERMAL_TEST_IMAGE = DEMO_DIR / "sample_thermal_image.jpg"
TEST_VIDEO = DEMO_DIR / "sample_inspection_video.mp4"
```

## Cell 32 — Code

```python
def run_image_inference(model, image_path, title="Inference", conf=0.25, save=True):
    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=save,
        save_txt=save,
        verbose=False
    )
    annotated = results[0].plot()[..., ::-1]

    plt.figure(figsize=(8, 6))
    plt.imshow(annotated)
    plt.title(title)
    plt.axis("off")
    plt.show()

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        df = pd.DataFrame({
            "class_id": boxes.cls.cpu().numpy().astype(int),
            "confidence": boxes.conf.cpu().numpy(),
        })
        df["class_name"] = df["class_id"].map(lambda x: model.names[int(x)])
        display(df.sort_values("confidence", ascending=False).reset_index(drop=True))
    else:
        print("No detections found.")

    return results
```

## Cell 33 — Code

```python
if RGB_TEST_IMAGE.exists():
    rgb_image_results = run_image_inference(
        rgb_best_model,
        RGB_TEST_IMAGE,
        title="RGB image inference",
        conf=0.25,
        save=True
    )
```

## Cell 34 — Code

```python
if THERMAL_TEST_IMAGE.exists():
    thermal_image_results = run_image_inference(
        thermal_best_model,
        THERMAL_TEST_IMAGE,
        title="Thermal image inference",
        conf=0.25,
        save=True
    )
```

## Cell 35 — Code

```python
def run_video_inference(model, video_path, output_path, conf=0.25, sample_every=30):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    sampled_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)

        if frame_idx % sample_every == 0:
            sampled_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Saved annotated video to: {output_path}")
    print(f"Processed frames: {frame_idx}")
    return sampled_frames
```

## Cell 36 — Code

```python
def show_sampled_frames(frames, title, max_frames=4):
    n = min(len(frames), max_frames)
    if n == 0:
        print("No sampled frames available.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, frame in zip(axes, frames[:n]):
        ax.imshow(frame)
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
```

## Cell 37 — Code

```python
if TEST_VIDEO.exists():
    rgb_video_frames = run_video_inference(
        model=rgb_best_model,
        video_path=TEST_VIDEO,
        output_path=OUT_DIR / "rgb_video_inference.mp4",
        conf=0.25,
        sample_every=40
    )
    show_sampled_frames(rgb_video_frames, "RGB video inference samples")
```

## Cell 38 — Code

```python
if TEST_VIDEO.exists():
    thermal_video_frames = run_video_inference(
        model=thermal_best_model,
        video_path=TEST_VIDEO,
        output_path=OUT_DIR / "thermal_video_inference.mp4",
        conf=0.25,
        sample_every=40
    )
    show_sampled_frames(thermal_video_frames, "Thermal video inference samples")
```

## Cell 39 — Markdown

```markdown
## 7. Telemetry loading and contextual feature engineering

The telemetry stream is treated as a frame-level context source. Four engineered terms are used. These terms convert raw mission variables into normalized quantities that are directly useful for fusion. The objective is to model contextual risk rather than to replace image-based evidence.

### Feature interpretation

The proximity term increases as the drone approaches the tower, reflecting the higher operational relevance of close-range observations. The stability term decreases when the platform exhibits vertical oscillation, vibration, roll, or pitch deviations. The thermal deviation term measures abnormality relative to the mission-level thermal distribution. The environmental term summarizes external conditions such as wind and navigation uncertainty.

**Tower proximity term**

\[
S_{prox} = \exp\left(-\frac{d_{tower}}{\tau_d}\right)
\]

**Stability term**

\[
S_{stab} = \exp\left(-\alpha|v_z| - \beta \cdot vibration - \gamma(|roll| + |pitch|)\right)
\]

**Thermal deviation term**

\[
S_{therm} = \sigma\left(\frac{\Delta T - \mu_T}{\sigma_T + \epsilon}\right)
\]

**Contextual telemetry risk**

\[
R_{tel} = w_1 S_{prox} + w_2(1 - S_{stab}) + w_3 S_{therm} + w_4 S_{env}
\]
```

## Cell 40 — Code

```python
mission_index = pd.read_csv(TEL_DIR / "mission_index.csv")
schema = json.load(open(TEL_DIR / "schema.json", "r", encoding="utf-8"))

display(mission_index.head())
```

## Cell 41 — Code

```python
def load_telemetry_split(split_name):
    split_dir = TEL_DIR / split_name
    telem_files = sorted((split_dir / "telemetry").glob("*.csv"))
    ann_dir = split_dir / "annotations"
    meta_dir = split_dir / "mission_metadata"
    frames = []
    for csv_path in tqdm(telem_files, desc=f"Loading {split_name}"):
        df = pd.read_csv(csv_path)
        mission_id = csv_path.stem
        ann_path = ann_dir / f"{mission_id}.json"
        meta_path = meta_dir / f"{mission_id}.json"
        ann = json.load(open(ann_path, "r", encoding="utf-8")) if ann_path.exists() else {}
        meta = json.load(open(meta_path, "r", encoding="utf-8")) if meta_path.exists() else {}
        df["split"] = split_name
        df["mission_id_from_file"] = mission_id
        df["annotation_json"] = json.dumps(ann)
        df["metadata_json"] = json.dumps(meta)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

telemetry_df = pd.concat([
    load_telemetry_split("train"),
    load_telemetry_split("val"),
    load_telemetry_split("test")
], ignore_index=True)
```

## Cell 42 — Code

```python
def num_col(df, name, default=0.0):
    return pd.to_numeric(df[name], errors="coerce").fillna(default) if name in df.columns else pd.Series(default, index=df.index)

telemetry_df["distance_to_tower"] = num_col(telemetry_df, "distance_to_tower")
telemetry_df["vertical_speed"] = num_col(telemetry_df, "vertical_speed")
telemetry_df["vibration_level"] = num_col(telemetry_df, "vibration_level")
telemetry_df["roll"] = num_col(telemetry_df, "roll")
telemetry_df["pitch"] = num_col(telemetry_df, "pitch")
telemetry_df["ir_temp_delta"] = num_col(telemetry_df, "ir_temp_delta")
telemetry_df["wind_speed"] = num_col(telemetry_df, "wind_speed")
telemetry_df["hdop"] = num_col(telemetry_df, "hdop")
telemetry_df["remaining_pct"] = num_col(telemetry_df, "remaining_pct")
```

## Cell 43 — Code

```python
TAU_D = max(telemetry_df["distance_to_tower"].median(), 1.0)
ALPHA, BETA, GAMMA = 0.30, 0.80, 0.02
EPS = 1e-6

telemetry_df["S_prox"] = np.exp(-telemetry_df["distance_to_tower"] / TAU_D)
telemetry_df["S_stab"] = np.exp(-(ALPHA * telemetry_df["vertical_speed"].abs() + BETA * telemetry_df["vibration_level"].abs() + GAMMA * (telemetry_df["roll"].abs() + telemetry_df["pitch"].abs())))
mu_T = telemetry_df["ir_temp_delta"].mean()
sig_T = telemetry_df["ir_temp_delta"].std() + EPS
telemetry_df["S_therm"] = 1 / (1 + np.exp(-((telemetry_df["ir_temp_delta"] - mu_T) / sig_T)))

wind_norm = telemetry_df["wind_speed"] / (telemetry_df["wind_speed"].max() + EPS)
hdop_norm = telemetry_df["hdop"] / (telemetry_df["hdop"].max() + EPS)
telemetry_df["S_env"] = np.clip(0.5 * wind_norm + 0.5 * hdop_norm, 0, 1)

w1, w2, w3, w4 = 0.30, 0.20, 0.35, 0.15
telemetry_df["telemetry_risk"] = np.clip(
    w1 * telemetry_df["S_prox"] +
    w2 * (1 - telemetry_df["S_stab"]) +
    w3 * telemetry_df["S_therm"] +
    w4 * telemetry_df["S_env"], 0, 1
)
```

## Cell 44 — Code

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
sns.histplot(telemetry_df["S_prox"], bins=30, kde=True, ax=axes[0, 0], color="royalblue")
axes[0, 0].set_title("Tower proximity term")

sns.histplot(telemetry_df["S_stab"], bins=30, kde=True, ax=axes[0, 1], color="seagreen")
axes[0, 1].set_title("Stability term")

sns.histplot(telemetry_df["S_therm"], bins=30, kde=True, ax=axes[1, 0], color="tomato")
axes[1, 0].set_title("Thermal deviation term")

sns.histplot(telemetry_df["telemetry_risk"], bins=30, kde=True, ax=axes[1, 1], color="purple")
axes[1, 1].set_title("Telemetry contextual risk")
plt.tight_layout()
plt.show()
```

## Cell 44A — Code

```python
# Relationship view for telemetry-derived factors
telemetry_plot_cols = [c for c in ['S_prox', 'S_stab', 'S_therm', 'S_env', 'telemetry_risk'] if c in telemetry_df.columns]
if len(telemetry_plot_cols) >= 2:
    sns.pairplot(telemetry_df[telemetry_plot_cols].sample(min(1000, len(telemetry_df))), corner=True, diag_kind='kde')
    plt.show()
```

## Cell 45 — Markdown

```markdown
## 8. Telemetry baseline classifier

If the telemetry stream contains anomaly labels, a frame-level classifier can be trained directly. This provides a contextual anomaly estimate derived from mission-state variables alone. The resulting probability score is not used as a replacement for RGB or thermal evidence, but as an auxiliary stream that refines final anomaly severity.
```

## Cell 46 — Code

```python
if "class" in telemetry_df.columns:
    telemetry_df["tel_label"] = telemetry_df["class"].map(lambda x: TEL_CLASSES.get(int(x), "no_anomaly") if pd.notna(x) else "no_anomaly")
else:
    telemetry_df["tel_label"] = np.where(telemetry_df["telemetry_risk"] >= 0.65, "hotspot_moderate", "no_anomaly")

feature_cols = ["S_prox", "S_stab", "S_therm", "S_env", "telemetry_risk", "remaining_pct"]
X = telemetry_df[feature_cols].fillna(0)
y = telemetry_df["tel_label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
tel_clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
tel_clf.fit(X_train, y_train)

y_pred = tel_clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Cell 47 — Code

```python
telemetry_df["telemetry_pred"] = tel_clf.predict(X)
telemetry_df["telemetry_pred_proba"] = tel_clf.predict_proba(X).max(axis=1)
```

## Cell 48 — Markdown

```markdown
## 9. Prediction tables and severity priors

Each detector output is reduced to a compact event table containing the top-confidence class and a modality score. The severity prior introduces domain knowledge by assigning greater weight to structurally critical or thermally critical detections. For example, a crack or critical hotspot should influence the final risk score more strongly than paint damage or generic structural context.
```

## Cell 49 — Code

```python
RGB_SEVERITY_PRIOR = {
    "crack": 0.95,
    "structural_bend": 0.90,
    "missing_bolt": 0.85,
    "corrosion": 0.65,
    "rust": 0.45,
    "paint_damage": 0.35,
}

THERMAL_SEVERITY_PRIOR = {
    "hotspot_critical": 1.00,
    "hotspot_moderate": 0.75,
    "hotspot_minor": 0.45,
    "connector_joint": 0.30,
    "equipment_cabinet": 0.20,
    "cable_harness": 0.20,
    "antenna_panel": 0.15,
    "tower_structure": 0.10,
}
```

## Cell 50 — Code

```python
def parse_mission_frame(path_str):
    stem = Path(path_str).stem.replace("-", "_")
    parts = stem.split("_")
    mission_id, frame_id = None, None
    for part in parts:
        low = part.lower()
        if low.startswith("mission"):
            mission_id = part
        if low.startswith("frame"):
            frame_id = part
    return mission_id, frame_id


def results_to_event_df(results, class_names, modality, severity_prior):
    rows = []
    for res in results:
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        best_idx = np.argmax(conf)
        top_class = class_names[int(cls[best_idx])]
        top_conf = float(conf[best_idx])
        mission_id, frame_id = parse_mission_frame(str(res.path))
        rows.append({
            "mission_id": mission_id,
            "frame_id": frame_id,
            "image_path": str(res.path),
            "modality": modality,
            "top_class": top_class,
            "top_conf": top_conf,
            "modality_score": np.clip(top_conf * severity_prior.get(top_class, 0.20), 0, 1)
        })
    return pd.DataFrame(rows)
```

## Cell 51 — Code

```python
rgb_all_results = rgb_best_model.predict(rgb_labels_df["image_path"].dropna().drop_duplicates().tolist(), conf=0.25, verbose=False)
thermal_all_results = thermal_best_model.predict(thermal_labels_df["image_path"].dropna().drop_duplicates().tolist(), conf=0.25, verbose=False)

rgb_event_df = results_to_event_df(rgb_all_results, RGB_CLASSES, "rgb", RGB_SEVERITY_PRIOR)
thermal_event_df = results_to_event_df(thermal_all_results, THERMAL_CLASSES, "thermal", THERMAL_SEVERITY_PRIOR)
```

## Cell 52 — Markdown

```markdown
## 10. Unified class mapping and fusion rule

The three modality streams are fused at decision level. This choice is appropriate because the modalities differ substantially in structure: RGB and thermal branches yield spatial detections, while telemetry yields tabular contextual signals. Decision-level fusion also preserves interpretability because each score can be inspected separately before aggregation.

The final event score is defined as

\[
R_{fused} = \lambda_1 R_{rgb} + \lambda_2 R_{th} + \lambda_3 R_{tel}
\]

with

\[
(\lambda_1, \lambda_2, \lambda_3) = (0.40, 0.40, 0.20)
\]

Severity is assigned by thresholding the fused score.
```

## Cell 53 — Code

```python
UNIFIED_CLASS_MAP = {
    "crack": "structural_crack",
    "structural_bend": "structural_deformation",
    "missing_bolt": "fastener_issue",
    "rust": "corrosion_issue",
    "corrosion": "corrosion_issue",
    "paint_damage": "surface_degradation",
    "hotspot_critical": "thermal_hotspot",
    "hotspot_moderate": "thermal_hotspot",
    "hotspot_minor": "thermal_hotspot",
    "connector_joint": "connector_related",
    "equipment_cabinet": "equipment_related",
    "cable_harness": "cable_related",
    "antenna_panel": "antenna_related",
    "tower_structure": "tower_context",
    "antenna_misalignment": "antenna_misalignment",
    "cable_fault": "cable_related",
    "connector_failure": "connector_related",
    "structural_crack": "structural_crack",
    "no_anomaly": "normal"
}
```

## Cell 54 — Code

```python
telemetry_events = telemetry_df.copy()
if "mission_id" not in telemetry_events.columns and "mission_id_from_file" in telemetry_events.columns:
    telemetry_events["mission_id"] = telemetry_events["mission_id_from_file"]
if "frame_id" not in telemetry_events.columns:
    telemetry_events["frame_id"] = telemetry_events.index.astype(str)

telemetry_events["telemetry_class"] = telemetry_events["telemetry_pred"]
telemetry_events["telemetry_unified_class"] = telemetry_events["telemetry_class"].map(lambda x: UNIFIED_CLASS_MAP.get(x, x))
telemetry_events = telemetry_events[[
    "mission_id", "frame_id", "timestamp_utc", "lat", "lon", "altitude_m",
    "telemetry_risk", "telemetry_pred_proba", "telemetry_class", "telemetry_unified_class"
]].copy()

rgb_event_df["rgb_unified_class"] = rgb_event_df["top_class"].map(lambda x: UNIFIED_CLASS_MAP.get(x, x))
thermal_event_df["thermal_unified_class"] = thermal_event_df["top_class"].map(lambda x: UNIFIED_CLASS_MAP.get(x, x))
```

## Cell 55 — Code

```python
fusion_df = telemetry_events.merge(
    rgb_event_df.rename(columns={
        "top_class": "rgb_top_class",
        "top_conf": "rgb_conf",
        "modality_score": "rgb_score",
        "image_path": "rgb_image_path"
    })[["mission_id", "frame_id", "rgb_top_class", "rgb_conf", "rgb_score", "rgb_image_path", "rgb_unified_class"]],
    on=["mission_id", "frame_id"], how="left"
).merge(
    thermal_event_df.rename(columns={
        "top_class": "thermal_top_class",
        "top_conf": "thermal_conf",
        "modality_score": "thermal_score",
        "image_path": "thermal_image_path"
    })[["mission_id", "frame_id", "thermal_top_class", "thermal_conf", "thermal_score", "thermal_image_path", "thermal_unified_class"]],
    on=["mission_id", "frame_id"], how="left"
)

fusion_df["rgb_score"] = fusion_df["rgb_score"].fillna(0)
fusion_df["thermal_score"] = fusion_df["thermal_score"].fillna(0)
fusion_df["telemetry_risk"] = fusion_df["telemetry_risk"].fillna(0)

L1, L2, L3 = 0.40, 0.40, 0.20
fusion_df["fused_risk"] = np.clip(L1 * fusion_df["rgb_score"] + L2 * fusion_df["thermal_score"] + L3 * fusion_df["telemetry_risk"], 0, 1)
```

## Cell 56 — Code

```python
def severity_from_score(x):
    if x >= 0.80:
        return "Critical"
    if x >= 0.60:
        return "High"
    if x >= 0.40:
        return "Medium"
    return "Low"

fusion_df["severity"] = fusion_df["fused_risk"].apply(severity_from_score)

final_classes = []
for _, row in fusion_df.iterrows():
    candidates = [
        (row.get("rgb_score", 0), row.get("rgb_unified_class", None)),
        (row.get("thermal_score", 0), row.get("thermal_unified_class", None)),
        (row.get("telemetry_risk", 0), row.get("telemetry_unified_class", None)),
    ]
    best = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
    final_classes.append(best[1] if pd.notna(best[1]) else "normal")

fusion_df["final_class"] = final_classes
```

## Cell 57 — Markdown

```markdown
### Temporal smoothing

Frame-wise decisions may fluctuate due to detector confidence variation, motion blur, or temporary occlusion. A short moving-average window is therefore applied to stabilize the event score over time.

The smoothed fused score is

\[
\widetilde{R}_t = \frac{1}{K} \sum_{i=0}^{K-1} R_{t-i}
\]

with \(K=3\).
```

## Cell 58 — Code

```python
sort_cols = ["mission_id", "timestamp_utc"] if "timestamp_utc" in fusion_df.columns else ["mission_id", "frame_id"]
fusion_df = fusion_df.sort_values(sort_cols).reset_index(drop=True)
fusion_df["fused_risk_smooth"] = fusion_df.groupby("mission_id")["fused_risk"].transform(lambda s: s.rolling(3, min_periods=1).mean())
fusion_df["severity_smooth"] = fusion_df["fused_risk_smooth"].apply(severity_from_score)
```

## Cell 59 — Code

```python
plt.figure(figsize=(10, 5))
sns.countplot(data=fusion_df, x="severity_smooth", order=["Low", "Medium", "High", "Critical"], palette="coolwarm")
plt.title("Distribution of fused severity labels")
plt.show()
```

## Cell 59A — Code

```python
plt.figure(figsize=(10, 5))
sns.histplot(fusion_df['fused_risk_smooth'], bins=30, kde=True, color='firebrick')
plt.title('Distribution of smoothed fused risk')
plt.xlabel('Smoothed fused risk')
plt.show()
```

## Cell 60 — Code

```python
display(
    fusion_df.sort_values("fused_risk_smooth", ascending=False).head(15)[[
        "mission_id", "frame_id", "final_class", "severity_smooth", "fused_risk_smooth",
        "rgb_top_class", "thermal_top_class", "telemetry_class", "lat", "lon", "altitude_m"
    ]].reset_index(drop=True)
)
```

## Cell 61 — Markdown

```markdown
## 11. Mission replay

The final stage presents a frame-ordered replay of a selected mission. For each step, the notebook shows RGB evidence, thermal evidence, telemetry context, and the fused alert. This replay section serves as the operational interpretation layer of the pipeline, transforming frame-level predictions into an inspection narrative.

### Replay objective

The mission replay should answer four questions for each alert: what was observed in the visible frame, what was observed in the thermal frame, what mission context was active at that instant, and what fused decision followed from these signals.
```

## Cell 62 — Code

```python
mission_counts = fusion_df["mission_id"].value_counts()
selected_mission = mission_counts.index[0]
mission_df = fusion_df[fusion_df["mission_id"] == selected_mission].copy()
mission_df = mission_df.sort_values("timestamp_utc") if "timestamp_utc" in mission_df.columns else mission_df.sort_values("frame_id")
mission_df = mission_df.reset_index(drop=True)

print("Selected mission:", selected_mission)
print("Mission rows:", mission_df.shape[0])
```

## Cell 63 — Code

```python
plt.figure(figsize=(14, 5))
plt.plot(mission_df.index, mission_df["fused_risk_smooth"], marker="o", linewidth=2)
plt.axhline(0.8, color="red", linestyle="--", label="Critical")
plt.axhline(0.6, color="orange", linestyle="--", label="High")
plt.axhline(0.4, color="gold", linestyle="--", label="Medium")
plt.xlabel("Frame order")
plt.ylabel("Smoothed fused risk")
plt.title(f"Mission replay risk profile: {selected_mission}")
plt.legend()
plt.show()
```

## Cell 64 — Code

```python
def print_event_summary(row):
    print("=" * 90)
    print(f"MISSION: {row['mission_id']} | FRAME: {row['frame_id']}")
    print(f"FINAL CLASS: {row['final_class']} | SEVERITY: {row['severity_smooth']} | SCORE: {row['fused_risk_smooth']:.3f}")
    print(f"RGB: {row.get('rgb_top_class', None)} | RGB score: {row.get('rgb_score', 0):.3f}")
    print(f"THERMAL: {row.get('thermal_top_class', None)} | Thermal score: {row.get('thermal_score', 0):.3f}")
    print(f"TELEMETRY: {row.get('telemetry_class', None)} | Telemetry risk: {row.get('telemetry_risk', 0):.3f}")
    print(f"LAT/LON: ({row.get('lat', None)}, {row.get('lon', None)}) | ALTITUDE: {row.get('altitude_m', None)}")
    print("=" * 90)
```

## Cell 65 — Code

```python
replay_rows = mission_df.iloc[::max(len(mission_df)//6, 1)].head(6)
display(replay_rows[["mission_id", "frame_id", "final_class", "severity_smooth", "fused_risk_smooth"]])
```

## Cell 66 — Code

```python
for _, row in replay_rows.iterrows():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    rgb_path = row.get("rgb_image_path", None)
    if isinstance(rgb_path, str) and Path(rgb_path).exists():
        axes[0].imshow(np.array(Image.open(rgb_path).convert("RGB")))
        axes[0].set_title(f"RGB\n{row.get('rgb_top_class', 'N/A')}")
    else:
        axes[0].text(0.5, 0.5, "RGB frame unavailable", ha="center", va="center")
    axes[0].axis("off")

    thermal_path = row.get("thermal_image_path", None)
    if isinstance(thermal_path, str) and Path(thermal_path).exists():
        axes[1].imshow(np.array(Image.open(thermal_path).convert("RGB")))
        axes[1].set_title(f"Thermal\n{row.get('thermal_top_class', 'N/A')}")
    else:
        axes[1].text(0.5, 0.5, "Thermal frame unavailable", ha="center", va="center")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    print_event_summary(row)
```

## Cell 67 — Code

```python
if {"lat", "lon"}.issubset(fusion_df.columns):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(fusion_df["lon"], fusion_df["lat"], c=fusion_df["fused_risk_smooth"], cmap="inferno", s=30)
    plt.colorbar(sc, label="Smoothed fused risk")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Geo-tagged event distribution")
    plt.show()
```

## Cell 68 — Code

```python
rgb_labels_df.to_csv(OUT_DIR / "rgb_labels_expanded.csv", index=False)
thermal_labels_df.to_csv(OUT_DIR / "thermal_labels_expanded.csv", index=False)
telemetry_df.to_csv(OUT_DIR / "telemetry_features.csv", index=False)
fusion_df.to_csv(OUT_DIR / "fused_events.csv", index=False)
print("Exports written to:", OUT_DIR)
```

## Cell 69 — Markdown

```markdown
## 12. Conclusion

The notebook establishes a trainable multimodal inspection workflow for cell tower monitoring using visible imagery, thermal imagery, and telemetry. The RGB and thermal branches provide detector-based evidence, the telemetry branch contributes contextual mission risk, and the late-fusion stage converts these signals into a unified event-level decision. The full pipeline therefore supports both model-centric evaluation and mission-centric interpretation.
```
