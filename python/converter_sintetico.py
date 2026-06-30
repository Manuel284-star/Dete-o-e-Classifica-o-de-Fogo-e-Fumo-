import json
import os

# Caminhos
json_pt1 = "annotationsPT1.json"
json_pt2 = "annotations parte 2 lpd.json"
output_labels = "dataset_sintetico/labels"
os.makedirs(output_labels, exist_ok=True)

# Classes
class_map = {"smoke": 1, "fire": 2}

# Carregar os dois JSONs e combinar
with open(json_pt1) as f:
    pt1 = json.load(f)
with open(json_pt2) as f:
    pt2 = json.load(f)

# Criar dicionário para combinar (PT2 tem prioridade se houver duplicados)
combined = {}
for item in pt1:
    combined[item['filename']] = item
for item in pt2:
    fname = item['filename']
    # Se PT2 tem anotações e PT1 não, usa PT2
    if item['annotations'] or fname not in combined:
        combined[fname] = item
    # Se ambos têm anotações, combina
    elif combined[fname]['annotations'] and item['annotations']:
        combined[fname]['annotations'].extend(item['annotations'])

print(f"Total de imagens combinadas: {len(combined)}")

# Converter para formato YOLO
converted = 0
empty = 0

for fname, item in combined.items():
    # Nome do ficheiro .txt
    txt_name = os.path.splitext(fname)[0] + ".txt"
    txt_path = os.path.join(output_labels, txt_name)
    
    if not item['annotations']:
        # Imagem sem anotações — ficheiro vazio
        open(txt_path, 'w').close()
        empty += 1
        continue
    
    with open(txt_path, 'w') as f:
        for ann in item['annotations']:
            label = ann['label']
            if label not in class_map:
                continue
            class_id = class_map[label]
            
            # bbox_percent: [x, y, w, h] em percentagem
            x_pct, y_pct, w_pct, h_pct = ann['bbox_percent']
            
            # Converter para YOLO (cx, cy, w, h normalizados 0-1)
            cx = (x_pct + w_pct / 2) / 100
            cy = (y_pct + h_pct / 2) / 100
            w = w_pct / 100
            h = h_pct / 100
            
            # Garantir que estão dentro dos limites
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    converted += 1

print(f"Imagens com anotações convertidas: {converted}")
print(f"Imagens sem anotações: {empty}")
print("Conversão concluída!")