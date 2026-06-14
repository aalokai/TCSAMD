# Tower Inspection Dataset
Synthetic drone-view images of lattice cell towers with annotated structural defects.

## Classes
| ID | Name             | Description                            |
|----|------------------|----------------------------------------|
| 0  | crack            | Fractures on members / welds           |
| 1  | rust             | Surface iron-oxide discoloration       |
| 2  | corrosion        | Advanced electrochemical degradation   |
| 3  | missing_bolt     | Empty fastener holes                   |
| 4  | paint_damage     | Chipped / peeling protective coating   |
| 5  | structural_bend  | Buckled or kinked structural members   |

## Quick start (YOLOv8)
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="tower_inspection_dataset/dataset.yaml", epochs=50, imgsz=640)
```
