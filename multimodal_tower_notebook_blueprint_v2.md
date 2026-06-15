# Multimodal Structural and Thermal Anomaly Detection for Cell Tower Inspection

## Notebook objective

This notebook implements a complete offline demonstration of a multimodal cell tower inspection workflow using three synchronized data sources: RGB structural imagery, thermal imagery, and drone telemetry. The pipeline trains separate detectors for RGB and thermal imagery, derives a contextual risk model from telemetry, and combines all three outputs into a fused event-level decision engine. The training and inference workflow follows the Python API pattern supported by Ultralytics YOLO, including model training, validation, and prediction from notebook cells [cite:66][cite:72]. The fusion stage follows a decision-level late-fusion strategy, which is a practical choice for heterogeneous modalities because each branch can be optimized independently before aggregation [cite:68][cite:79].

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
└── telemetry_dataset/
    ├── dataset.yaml
    ├── schema.json
    ├── mission_index.csv
    ├── train/telemetry/
    ├── train/annotations/
    ├── train/mission_metadata/
    ├── val/telemetry/
    ├── val/annotations/
    ├── val/mission_metadata/
    ├── test/telemetry/
    ├── test/annotations/
    └── test/mission_metadata/
```

## Cell 1 — Markdown

```markdown
# Multimodal Structural and Thermal Anomaly Detection for Cell Tower Inspection

## Abstract
This notebook presents a multimodal inspection framework for telecom tower analysis using RGB imagery, thermal imagery, and telemetry. The study addresses three complementary tasks: structural defect detection from visible imagery, hotspot and component analysis from thermal imagery, and context-aware anomaly scoring from mission telemetry. The final stage performs event-level late fusion to generate geo-tagged inspection alerts with severity labels.

## Problem context
Inspection of telecom towers requires identification of both visible structural degradation and temperature-related faults. RGB imagery captures corrosion, cracks, missing bolts, and member deformation. Thermal imagery captures localized hotspots and component context. Telemetry provides mission-state information such as tower proximity, platform stability, navigation quality, and environmental conditions. The central objective is to integrate these modalities into a unified inspection decision rather than treating them as independent streams.

## Contributions
1. Training of an RGB detector for structural defect localization.
2. Training of a thermal detector for hotspot and component localization.
3. Engineering of a telemetry-derived contextual risk score.
4. Definition of a decision-level multimodal fusion rule.
5. Construction of a mission replay section that presents fused alerts frame by frame.
```

## Cell 2 — Code

```python
# Run once if the environment is fresh
# !pip install -q ultralytics opencv-python pillow seaborn scikit-learn plotly pyyaml tqdm
```

## Cell 3 — Code

```python
import os
import json
import glob
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

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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

The notebook is organized into four analytic stages. First, the RGB dataset is used to train an object detector for visible structural anomalies. Second, the thermal dataset is used to train a detector for hotspot-related anomalies and tower components. Third, telemetry records are transformed into frame-level contextual descriptors and a risk model. Fourth, the modality-specific outputs are merged by `mission_id` and `frame_id`, and a fused risk score is computed.

This structure follows a late-fusion design. The motivation for late fusion is that the three modalities differ in representation and supervision: RGB and thermal branches are spatial detectors, while telemetry is sequential tabular data. Decision-level fusion preserves interpretability and allows each branch to be evaluated independently before aggregation [cite:68][cite:79].
```

## Cell 5 — Code

```python
ROOT = Path(".").resolve()
RGB_DIR = ROOT / "tower_inspection_dataset"
THERMAL_DIR = ROOT / "cell_tower_thermal"
TEL_DIR = ROOT / "telemetry_dataset"
OUT_DIR = ROOT / "notebook_outputs"
OUT_DIR.mkdir(exist_ok=True)

print("Project root:", ROOT)
print("RGB dataset found:", RGB_DIR.exists())
print("Thermal dataset found:", THERMAL_DIR.exists())
print("Telemetry dataset found:", TEL_DIR.exists())
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

print(rgb_yaml)
print(thermal_yaml)
print(telemetry_yaml)
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

rgb_cfg, thermal_cfg, telemetry_cfg
```

## Cell 8 — Markdown

```markdown
## 2. Dataset specification

The RGB dataset contains six structural defect classes: crack, rust, corrosion, missing bolt, paint damage, and structural bend. The thermal dataset contains both object-context classes and heat-anomaly classes, which enables interpretation of hotspot location in relation to tower components. The telemetry dataset provides frame-linked navigation, attitude, IMU, power, environmental, and anomaly-related metadata.

The visible and thermal datasets are stored in YOLO normalized bounding-box format. Ultralytics provides direct support for training and evaluating detectors on this format using notebook-friendly Python workflows [cite:66][cite:72].
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

print(RGB_CLASSES)
print(THERMAL_CLASSES)
print(TEL_CLASSES)
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

rgb_split_df, thermal_split_df
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

telemetry_split_df = telemetry_counts(TEL_DIR)
telemetry_split_df
```

## Cell 12 — Markdown

```markdown
## 3. Label parsing and descriptive statistics

The YOLO label representation for each object is

\[
(class\_id, c_x, c_y, w, h)
\]

where all coordinates are normalized to the image dimensions. Converting these labels into tabular form enables class distribution analysis, sample visualization, and downstream sanity checks before training.
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

print(rgb_labels_df.shape)
print(thermal_labels_df.shape)
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
plt.title("RGB label visualization")
plt.axis("off")
plt.show()
```

## Cell 16 — Code

```python
thermal_sample = thermal_labels_df["image_path"].dropna().iloc[0]
thermal_sample_rows = thermal_labels_df[thermal_labels_df["image_path"] == thermal_sample]
plt.figure(figsize=(8, 8))
plt.imshow(draw_boxes(thermal_sample, thermal_sample_rows))
plt.title("Thermal label visualization")
plt.axis("off")
plt.show()
```

## Cell 17 — Markdown

```markdown
## 4. RGB detector training

The RGB branch is trained as a structural defect detector. A lightweight pretrained detector is used as initialization, and the model is fine-tuned on the tower inspection dataset. Ultralytics recommends this workflow for custom detection tasks, with training and validation directly accessible through Python methods in notebook environments [cite:66][cite:72].
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

## Cell 21 — Markdown

```markdown
### 4.1 RGB evaluation metrics

For each class, precision and recall are defined as

\[
Precision = \frac{TP}{TP + FP}
\]

\[
Recall = \frac{TP}{TP + FN}
\]

Mean average precision summarizes the detector quality across intersection-over-union thresholds and classes. In inspection tasks, these values should be interpreted together with class-wise results and qualitative overlays because rare defects can remain difficult even when aggregate metrics appear strong [cite:72].
```

## Cell 22 — Code

```python
rgb_best_weights = list((OUT_DIR / "rgb_runs").glob("**/weights/best.pt"))[0]
rgb_best_model = YOLO(str(rgb_best_weights))

rgb_test_images = rgb_labels_df.query("split == 'test'")["image_path"].dropna().drop_duplicates().sample(4, random_state=42).tolist()
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

The thermal branch is trained to identify both hotspot classes and inspection-relevant components. This joint representation is useful because a hotspot without component context may be harder to interpret than a hotspot associated with a connector, cable run, or cabinet. Thermal anomaly detection workflows commonly use dedicated image analysis or object detection methods to localize elevated-temperature regions and reduce false alarms [cite:76][cite:36].
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

## Cell 28 — Code

```python
thermal_best_weights = list((OUT_DIR / "thermal_runs").glob("**/weights/best.pt"))[0]
thermal_best_model = YOLO(str(thermal_best_weights))

thermal_test_images = thermal_labels_df.query("split == 'test'")["image_path"].dropna().drop_duplicates().sample(4, random_state=42).tolist()
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
## 6. Telemetry loading and contextual feature engineering

The telemetry stream is treated as a frame-level context source. Unlike the image branches, the telemetry branch does not localize objects. Instead, it produces a contextual risk estimate derived from tower distance, platform stability, thermal deviation, and environmental conditions.

Four engineered terms are used:

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

These terms convert raw mission measurements into bounded, interpretable factors. The formulation keeps the telemetry branch transparent and easy to inspect during error analysis.
```

## Cell 31 — Code

```python
mission_index = pd.read_csv(TEL_DIR / "mission_index.csv")
schema = json.load(open(TEL_DIR / "schema.json", "r", encoding="utf-8"))

mission_index.head()
```

## Cell 32 — Code

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

telemetry_df.head()
```

## Cell 33 — Code

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

## Cell 34 — Code

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

## Cell 35 — Code

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

## Cell 36 — Markdown

```markdown
## 7. Telemetry classification target

If the telemetry CSV already includes anomaly labels, a supervised classifier can be trained directly. If the anomaly field is missing or sparse, the engineered telemetry risk can still be used as a contextual input to the fusion stage. When labels are available, a random-forest baseline is a reasonable starting point for tabular anomaly discrimination because it handles mixed nonlinear relationships with limited preprocessing.
```

## Cell 37 — Code

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

## Cell 38 — Code

```python
telemetry_df["telemetry_pred"] = tel_clf.predict(X)
telemetry_df["telemetry_pred_proba"] = tel_clf.predict_proba(X).max(axis=1)
telemetry_df[["mission_id", "frame_id", "telemetry_pred", "telemetry_pred_proba", "telemetry_risk"]].head() if {"mission_id", "frame_id"}.issubset(telemetry_df.columns) else telemetry_df[["telemetry_pred", "telemetry_pred_proba", "telemetry_risk"]].head()
```

## Cell 39 — Markdown

```markdown
## 8. Detector prediction tables

The fusion stage operates on compact prediction tables rather than raw detector objects. Each RGB and thermal image is reduced to its top-confidence detection and a severity-weighted modality score. This creates a consistent event representation that can be merged with telemetry records.
```

## Cell 40 — Code

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

## Cell 41 — Code

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

rgb_event_df = results_to_event_df(rgb_best_model.predict(rgb_labels_df["image_path"].dropna().drop_duplicates().tolist(), conf=0.25, verbose=False), RGB_CLASSES, "rgb", RGB_SEVERITY_PRIOR)
thermal_event_df = results_to_event_df(thermal_best_model.predict(thermal_labels_df["image_path"].dropna().drop_duplicates().tolist(), conf=0.25, verbose=False), THERMAL_CLASSES, "thermal", THERMAL_SEVERITY_PRIOR)
```

## Cell 42 — Markdown

```markdown
## 9. Unified class mapping

The three datasets do not use identical label vocabularies. A unified label space is therefore introduced to support fused event reporting. This layer is semantic rather than architectural: it translates dataset-specific classes into inspection-oriented categories.
```

## Cell 43 — Code

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

## Cell 44 — Markdown

```markdown
## 10. Late-fusion rule

Let
- \(R_{rgb}\) denote the RGB modality score,
- \(R_{th}\) denote the thermal modality score,
- \(R_{tel}\) denote the telemetry contextual risk.

The fused event score is defined as

\[
R_{fused} = \lambda_1 R_{rgb} + \lambda_2 R_{th} + \lambda_3 R_{tel}
\]

subject to

\[
\lambda_1 + \lambda_2 + \lambda_3 = 1, \qquad \lambda_i \ge 0
\]

A practical initialization is

\[
(\lambda_1, \lambda_2, \lambda_3) = (0.40, 0.40, 0.20)
\]

The final severity is assigned by thresholding:

\[
Severity(R_{fused}) =
\begin{cases}
\text{Critical}, & R_{fused} \ge 0.80 \\
\text{High}, & 0.60 \le R_{fused} < 0.80 \\
\text{Medium}, & 0.40 \le R_{fused} < 0.60 \\
\text{Low}, & R_{fused} < 0.40
\end{cases}
\]

This rule preserves interpretability while allowing modality-specific detectors to contribute proportionally. Decision-level multimodal fusion is particularly suitable when modalities are heterogeneous and independently modeled [cite:68][cite:79].
```

## Cell 45 — Code

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
```

## Cell 46 — Code

```python
rgb_event_df["rgb_unified_class"] = rgb_event_df["top_class"].map(lambda x: UNIFIED_CLASS_MAP.get(x, x))
thermal_event_df["thermal_unified_class"] = thermal_event_df["top_class"].map(lambda x: UNIFIED_CLASS_MAP.get(x, x))

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

## Cell 47 — Code

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

## Cell 48 — Markdown

```markdown
### 10.1 Temporal smoothing

Frame-level inspection scores may fluctuate due to viewpoint variation, thermal noise, or brief instability in the platform. A moving-average smoother is therefore applied:

\[
\widetilde{R}_t = \frac{1}{K} \sum_{i=0}^{K-1} R_{t-i}
\]

with window size \(K=3\). The smoothed score supports cleaner mission-level interpretation and more stable replay visualization.
```

## Cell 49 — Code

```python
sort_cols = ["mission_id", "timestamp_utc"] if "timestamp_utc" in fusion_df.columns else ["mission_id", "frame_id"]
fusion_df = fusion_df.sort_values(sort_cols).reset_index(drop=True)
fusion_df["fused_risk_smooth"] = fusion_df.groupby("mission_id")["fused_risk"].transform(lambda s: s.rolling(3, min_periods=1).mean())
fusion_df["severity_smooth"] = fusion_df["fused_risk_smooth"].apply(severity_from_score)
```

## Cell 50 — Code

```python
plt.figure(figsize=(10, 5))
sns.countplot(data=fusion_df, x="severity_smooth", order=["Low", "Medium", "High", "Critical"], palette="coolwarm")
plt.title("Distribution of fused severity labels")
plt.show()
```

## Cell 51 — Code

```python
top_events = fusion_df.sort_values("fused_risk_smooth", ascending=False).head(15)
top_events[[
    "mission_id", "frame_id", "final_class", "severity_smooth", "fused_risk_smooth",
    "rgb_top_class", "thermal_top_class", "telemetry_class", "lat", "lon", "altitude_m"
]].reset_index(drop=True)
```

## Cell 52 — Markdown

```markdown
## 11. Mission replay analysis

The final stage presents a frame-ordered replay of a selected mission. For each replay step, the notebook shows RGB evidence, thermal evidence, telemetry context, and the fused alert. This converts the multimodal outputs into a mission narrative rather than a disconnected collection of model predictions.
```

## Cell 53 — Code

```python
mission_counts = fusion_df["mission_id"].value_counts()
selected_mission = mission_counts.index[0]
mission_df = fusion_df[fusion_df["mission_id"] == selected_mission].copy()
mission_df = mission_df.sort_values("timestamp_utc") if "timestamp_utc" in mission_df.columns else mission_df.sort_values("frame_id")
mission_df = mission_df.reset_index(drop=True)

selected_mission, mission_df.shape
```

## Cell 54 — Code

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

## Cell 55 — Code

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

## Cell 56 — Code

```python
replay_rows = mission_df.iloc[::max(len(mission_df)//6, 1)].head(6)
replay_rows[["mission_id", "frame_id", "final_class", "severity_smooth", "fused_risk_smooth"]]
```

## Cell 57 — Code

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

## Cell 58 — Markdown

```markdown
## 12. Result synthesis

Three levels of evidence are now available:
- spatial evidence from the RGB detector,
- thermal evidence from the infrared detector,
- contextual evidence from the telemetry branch.

Their integration at the event level produces a practical inspection output that can be prioritized by severity and linked to frame-level evidence and geographic location. This structure is consistent with multimodal anomaly workflows in which final decision quality improves when heterogeneous evidence is fused after modality-specific processing [cite:68][cite:71].
```

## Cell 59 — Code

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

## Cell 60 — Code

```python
rgb_labels_df.to_csv(OUT_DIR / "rgb_labels_expanded.csv", index=False)
thermal_labels_df.to_csv(OUT_DIR / "thermal_labels_expanded.csv", index=False)
telemetry_df.to_csv(OUT_DIR / "telemetry_features.csv", index=False)
fusion_df.to_csv(OUT_DIR / "fused_events.csv", index=False)
print("Exports written to:", OUT_DIR)
```

## Cell 61 — Markdown

```markdown
## 13. Conclusion

This notebook establishes a trainable multimodal inspection workflow for cell tower monitoring using visible imagery, thermal imagery, and telemetry. The RGB and thermal branches provide detector-based evidence, the telemetry branch contributes contextual mission risk, and the late-fusion stage converts these signals into a unified event-level decision. The resulting framework supports both quantitative evaluation and mission-style qualitative replay within a single notebook.
```
