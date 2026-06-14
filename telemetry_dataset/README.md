# 📡 Drone Geospatial Telemetry Dataset — Cell Tower Inspection

## Project Context
Telemetry module of the **multi-modal drone inspection pipeline**:

```
┌─────────────────────────────────────────────────────────────────┐
│           DRONE INSPECTION PIPELINE — DATA STREAMS              │
├───────────────┬──────────────────────┬──────────────────────────┤
│  4K RGB Video │  Thermal Imaging     │  Geospatial Telemetry    │
│  (frames)     │  cell_tower_thermal/ │  telemetry_dataset/  ◄── │
│               │  YOLO annotations    │  THIS DATASET            │
└───────────────┴──────────────────────┴──────────────────────────┘
         Fuse via:  mission_id + frame_id
```

---

## Dataset Summary
| Property            | Value                        |
|---------------------|------------------------------|
| Total Missions      | 100                          |
| Train / Val / Test  | 70 / 20 / 10                 |
| Telemetry Rate      | 10 Hz (frames/sec)           |
| Fields per Frame    | 63                           |
| Tower Sites         | 10 cities across India       |
| Coordinate System   | WGS84                        |
| Altitude Reference  | AGL (Above Ground Level)     |

---

## Directory Structure
```
telemetry_dataset/
├── dataset.yaml
├── schema.json                        ← Full field schema + units
├── mission_index.csv                  ← Master index of all 100 missions
├── README.md
├── train/
│   ├── telemetry/     (70 × mission CSV files)
│   ├── annotations/   (70 × anomaly JSON files)
│   └── mission_metadata/ (70 × metadata JSON files)
├── val/   (same structure, 20 missions)
└── test/  (same structure, 10 missions)
```

---

## Telemetry Fields (63 per frame)
| Group           | Fields                                              |
|-----------------|-----------------------------------------------------|
| Timestamps/IDs  | timestamp_utc, mission_id, frame_id, waypoint_id, inspection_phase, site_id |
| Navigation      | lat, lon, altitude_m, heading, ground_speed, vertical_speed, distance_to_tower |
| Attitude        | roll, pitch, yaw + quaternion (w, x, y, z)         |
| IMU             | accel_xyz, gyro_xyz, vibration_level                |
| GPS             | fix_type, satellite_count, hdop, vdop, position_error |
| Battery         | voltage, current, remaining_pct, temp, capacity_mah |
| Motors (×4)     | rpm, temp_c, thrust_n per motor (FL/FR/RL/RR)       |
| Environment     | wind_speed, wind_dir, air_temp, humidity, pressure  |
| Camera/Gimbal   | gimbal_pitch, gimbal_roll, zoom_factor, ir_min/max/avg |
| Anomaly Output  | detected, class, confidence, bbox_xywh, gps_tag, ir_temp_delta |

---

## Anomaly Classes
| ID | Class                | Severity    |
|----|----------------------|-------------|
| 0  | hotspot_critical     | 🔴 URGENT   |
| 1  | hotspot_moderate     | 🟠 High     |
| 2  | hotspot_minor        | 🟡 Medium   |
| 3  | antenna_misalignment | 🟠 High     |
| 4  | cable_fault          | 🔴 URGENT   |
| 5  | connector_failure    | 🔴 URGENT   |
| 6  | corrosion            | 🟡 Medium   |
| 7  | structural_crack     | 🔴 URGENT   |
| 8  | no_anomaly           | ✅ Normal   |

---

## Inspection Phases
`takeoff → transit → orbit_base → climb_tower →`
`antenna_inspection → cable_inspection → cabinet_inspection →`
`descent → landing`

---

## Usage

### Load a mission
```python
import pandas as pd, json

df = pd.read_csv("train/telemetry/MISSION_0001.csv")
with open("train/annotations/MISSION_0001_annotations.json") as f:
    annotations = json.load(f)

# Filter anomaly frames only
anomalies = df[df["anomaly_detected"] == 1]
print(anomalies[["timestamp_utc","anomaly_class","anomaly_confidence","ir_temp_delta_c"]])
```

### Geospatial visualization (GeoPandas)
```python
import geopandas as gpd
from shapely.geometry import Point

gdf = gpd.GeoDataFrame(df, geometry=[Point(r.longitude_deg, r.latitude_deg) for _,r in df.iterrows()])
gdf.to_file("mission_0001_path.geojson", driver="GeoJSON")
```

### Fuse with thermal dataset
```python
# Match telemetry frame to thermal image
frame_id  = 47
tel_row   = df[df["frame_id"] == frame_id].iloc[0]
img_path  = f"../cell_tower_thermal/train/images/tower_thermal_{frame_id:04d}.jpg"

# Get GPS-tagged anomaly location
if tel_row["anomaly_detected"]:
    print(f"Anomaly: {tel_row['anomaly_class']}")
    print(f"GPS: {tel_row['anomaly_gps_lat']}, {tel_row['anomaly_gps_lon']}")
    print(f"Temp delta: {tel_row['ir_temp_delta_c']} °C")
```

### Export to KML / Google Earth
```python
import simplekml
kml = simplekml.Kml()
for _, row in anomalies.iterrows():
    pt = kml.newpoint(name=row["anomaly_class"],
                      coords=[(row["anomaly_gps_lon"], row["anomaly_gps_lat"], row["anomaly_gps_alt_m"])])
    pt.description = f"Confidence: {row['anomaly_confidence']} | ΔT: {row['ir_temp_delta_c']}°C"
kml.save("anomalies.kml")
```

---

## Tower Sites
| Site ID       | City       | Tower Type | Height |
|---------------|------------|------------|--------|
| SITE_BOM_001  | Mumbai     | Lattice    | 42 m   |
| SITE_DEL_002  | Delhi      | Monopole   | 35 m   |
| SITE_BLR_003  | Bengaluru  | Lattice    | 55 m   |
| SITE_CHE_004  | Chennai    | Monopole   | 38 m   |
| SITE_HYD_005  | Hyderabad  | Guyed      | 60 m   |
| SITE_KOL_006  | Kolkata    | Lattice    | 48 m   |
| SITE_PUN_007  | Pune       | Monopole   | 30 m   |
| SITE_AMD_008  | Ahmedabad  | Lattice    | 45 m   |
| SITE_JAI_009  | Jaipur     | Monopole   | 33 m   |
| SITE_LKN_010  | Lucknow    | Guyed      | 52 m   |

---
*Part of the Drone-Based Cell Tower Inspection Pipeline — Hackathon Dataset*
