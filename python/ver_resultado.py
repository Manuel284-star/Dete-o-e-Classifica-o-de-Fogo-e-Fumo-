import sys
from ultralytics import YOLO

imagem = sys.argv[1] if len(sys.argv) > 1 else None
modelo = sys.argv[2] if len(sys.argv) > 2 else 'best.pt'

if not imagem:
    print('Uso: python3 ver_resultado.py <imagem> [modelo.pt]')
    sys.exit(1)

model = YOLO(modelo)
results = model(imagem)

names = {0: 'negative', 1: 'smoke', 2: 'fire'}
for r in results:
    print(f'Deteções: {len(r.boxes)}')
    for box in r.boxes:
        cls = int(box.cls[0]); conf = float(box.conf[0])
        print(f'  {names.get(cls, cls)} | {conf:.2f}')
    # guarda a imagem com as caixas desenhadas
    r.save(filename='resultado.jpg')

print('Imagem guardada como resultado.jpg')
