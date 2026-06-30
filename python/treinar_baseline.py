from ultralytics import YOLO

# Modelo YOLO pré-treinado pequeno
model = YOLO("yolov8n.pt")

# Treino baseline
model.train(
    data="data.yaml",
    epochs=1,
    imgsz=320,
    batch=2,
    name="baseline_smoke_yolov8n"
)