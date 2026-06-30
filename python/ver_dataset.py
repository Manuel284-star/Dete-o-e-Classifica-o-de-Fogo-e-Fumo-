from datasets import load_dataset

dataset = load_dataset("pyronear/pyro-sdis")

print(dataset)
print()
print("Splits disponíveis:", dataset.keys())
print()
print("Primeiro exemplo do train:")
print(dataset["train"][0].keys())
print()
print(dataset["train"][0])