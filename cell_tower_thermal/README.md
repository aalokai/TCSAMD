# 📡 Drone-Based Cell Tower Thermal Inspection Dataset

## Project Context
Part of a **multi-modal drone inspection pipeline** combining:
- 🎥 High-resolution 4K RGB video feed
- 🌡️ Thermal infrared imaging (this dataset)
- 📍 Geospatial telemetry (GPS/IMU)

Real-time anomaly detection and structural fault classification for 5G/LTE cell towers.

---

## Dataset Summary
| Property       | Value                           |
|----------------|---------------------------------|
| Total Images   | 100                             |
| Train / Val / Test | 70 / 20 / 10               |
| Resolution     | 640 × 512 px                    |
| Format         | YOLO (normalized xywh)          |
| Palette        | Iron / Rainbow thermal false-color |
| Classes        | 8                               |

---

## Classes
| ID | Name               | Description                                   |
|----|--------------------|-----------------------------------------------|
| 0  | tower_structure    | Lattice / monopole tower body                 |
| 1  | antenna_panel      | Sector antennas, panel arrays                 |
| 2  | cable_harness      | Feeder cables, coax bundles                   |
| 3  | hotspot_critical   | Severe anomaly >15 °C above ambient — 🔴 URGENT |
| 4  | hotspot_moderate   | Moderate anomaly 8–15 °C — 🟠 Schedule fix    |
| 5  | hotspot_minor      | Minor anomaly <8 °C — 🟡 Monitor              |
| 6  | connector_joint    | Weatherproof connectors / jumpers             |
| 7  | equipment_cabinet  | BTS / RRU / power cabinets                    |

---

## Directory Structure
```
cell_tower_thermal/
├── dataset.yaml
├── README.md
├── train/
│   ├── images/   (70 × .jpg thermal images)
│   └── labels/   (70 × .txt YOLO annotations)
├── val/
│   ├── images/   (20 × .jpg thermal images)
│   └── labels/   (20 × .txt YOLO annotations)
└── test/
    ├── images/   (10 × .jpg thermal images)
    └── labels/   (10 × .txt YOLO annotations)
```

---

## Inspection Scenarios
| Scenario                  | Description                                         |
|---------------------------|-----------------------------------------------------|
| lattice_tower_full        | Full tower in frame — multiple antenna banks        |
| monopole_closeup          | Side approach — RRU units + cable run               |
| antenna_sector_closeup    | Close-up drone hover — single sector face           |
| equipment_cabinet_bay     | Ground-level / low-altitude — BTS cabinet row      |
| cable_run_inspection      | Lateral pass — feeder cable joints                  |

---

## Annotation Format (YOLO)
```
<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
```
All values normalised to [0, 1] relative to image width × height (640 × 512).

---

## Usage with YOLOv8
```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")          # medium backbone for multi-class
results = model.train(
    data   = "cell_tower_thermal/dataset.yaml",
    epochs = 100,
    imgsz  = 640,
    batch  = 16,
    name   = "tower_thermal_v1",
)
```

## Multi-Modal Fusion Tips
```python
# Align thermal bounding boxes to 4K RGB frame using homography
import cv2
H_matrix, _ = cv2.findHomography(thermal_pts, rgb_pts)
thermal_bbox_rgb = cv2.perspectiveTransform(thermal_bbox, H_matrix)

# Merge with geospatial telemetry
anomaly_gps = {
    "lat":  drone_lat + delta_lat(bbox_cx, altitude),
    "lon":  drone_lon + delta_lon(bbox_cy, altitude),
    "severity": predicted_class,
    "temp_delta": estimated_temp_rise,
}
```

---

## Drone Sensor Spec (simulated)
- **Type:** Uncooled LWIR microbolometer
- **Spectral range:** 8–14 µm
- **NETD:** <50 mK
- **Palette:** Iron / Rainbow false-color
- **Integration:** DJI Zenmuse H20T / FLIR Vue Pro R compatible

---
*Generated for hackathon — Drone-Based Cell Tower Inspection Pipeline*
