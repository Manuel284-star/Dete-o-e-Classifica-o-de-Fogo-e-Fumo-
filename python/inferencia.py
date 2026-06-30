import sys
from ultralytics import YOLO

modelo = sys.argv[2] if len(sys.argv) > 2 else "yolov8n.pt"
imagem = sys.argv[1] if len(sys.argv) > 1 else None

if not imagem:
    print("Uso: python3 inferencia.py <caminho_imagem> [modelo.pt]")
    sys.exit(1)

print(f"Modelo: {modelo}")
print(f"Imagem: {imagem}")

model = YOLO(modelo)
results = model(imagem)

for r in results:
    boxes = r.boxes
    print(f"Deteções encontradas: {len(boxes)}")
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        names = {0: "negative", 1: "smoke", 2: "fire"}
        print(f"  Classe: {names.get(cls, cls)} | Confiança: {conf:.2f}")

print("Inferência concluída!")
