from datasets import load_dataset
import os

dataset = load_dataset("pyronear/pyro-sdis")

# criar pastas
os.makedirs("dataset/images/train", exist_ok=True)
os.makedirs("dataset/images/val", exist_ok=True)
os.makedirs("dataset/labels/train", exist_ok=True)
os.makedirs("dataset/labels/val", exist_ok=True)

def process(split):
    for i, item in enumerate(dataset[split]):
        img = item["image"]
        label = item["annotations"]
        name = item["image_name"].replace(".jpg", "")

        # guardar imagem
        img.save(f"dataset/images/{split}/{name}.jpg")

        # guardar label
        with open(f"dataset/labels/{split}/{name}.txt", "w") as f:
            f.write(label)

        if i % 1000 == 0:
            print(f"{split}: {i}")

# processar
process("train")
process("val")

print("Conversão completa!")