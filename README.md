# Deteção e Classificação de Fogo e Fumo por Aprendizagem Profunda

Projeto desenvolvido no âmbito da unidade curricular de **Laboratório de Projeto (2025-2026)** da Licenciatura em Informática de Gestão da **Universidade Autónoma de Lisboa Luís de Camões**, em colaboração com a empresa **Leitek Innovative Solutions, Lda.**

**Orientador:** Professor Doutor Mário Marques da Silva
**Autores:** Francisco Ferreira (30012759), Manuel Soares (30012371), Martim Gomes (30012873)

---

## Descrição

Sistema de deteção e classificação de fogo e fumo em imagens de ambientes florestais, recorrendo a técnicas de aprendizagem profunda (*deep learning*) e aprendizagem por transferência (*transfer learning*). O projeto compara três abordagens:

- **YOLOv8n** (Python / Ultralytics) — modelo principal de deteção, treinado com GPU no Google Colab;
- **YOLOv4** (MATLAB) — modelo de deteção alternativo, treinado em CPU;
- **CNN própria** (MATLAB) — rede convolucional para classificação binária (com/sem fumo).

Inclui ainda estudos de ablação com dados sintéticos e um pipeline de inferência em Linux (WSL) sem dependência do MATLAB.

## Resultados Principais

| Modelo | Ambiente | Métrica | Valor |
|---|---|---|---|
| YOLOv8n (50 épocas) | Python / GPU T4 | mAP@0.5 | **0.706** |
| YOLOv4 | MATLAB / CPU | mAP@0.5 | 0.0153 |
| CNN balanceada | MATLAB / CPU | Exatidão | 80.63% |

## Estrutura do Repositório

### Scripts Python
- `converter.py` — conversão do dataset pyronear/pyro-sdis (HuggingFace) para formato YOLO
- `converter_sintetico.py` — conversão das anotações sintéticas (JSON) para formato YOLO
- `treinar_baseline.py` — treino do modelo YOLOv8n
- `inferencia.py` — inferência via linha de comandos (CLI)
- `ver_resultado.py` — inferência com visualização das deteções
- `ver_dataset.py` — exploração do dataset
- `data.yaml` — configuração das classes do dataset

### Scripts MATLAB
- `treinar_yolov4.m` — treino do detetor YOLOv4
- `avaliar_yolov4.m` — avaliação do YOLOv4
- `treinar_cnn.m` / `treinar_cnn_balanced.m` — treino da CNN (versões desequilibrada e balanceada)
- `avaliar_cnn_balanced1.m` — avaliação da CNN balanceada
- `preparar_dataset_matlab.m` — preparação dos dados para o YOLOv4
- `preparar_dataset_cnn.m` / `preparar_dataset_cnn_balanced.m` — preparação dos dados para a CNN

### Anotações
- `annotationsPT1.json` / `annotations parte 2 lpd.json` — anotações manuais das imagens sintéticas

## Modelos Treinados e Datasets

Devido à sua dimensão, os modelos treinados e os datasets não estão neste repositório. Estão disponíveis no seguinte link:

> **[COLOCAR AQUI O LINK DO GOOGLE DRIVE]**

Ficheiros disponíveis no Drive:
- `best.pt` — modelo YOLOv8n treinado (PyTorch)
- `best.onnx` — modelo YOLOv8n exportado para ONNX
- `yolov4_smoke.mat` — modelo YOLOv4 treinado (MATLAB)
- `cnn_smoke_classifier.mat` / `cnn_smoke_classifier_balanced.mat` — modelos CNN (MATLAB)

## Utilização do Pipeline de Inferência (Linux)

```bash
# Instalar dependências
pip3 install ultralytics --break-system-packages

# Inferência numa imagem
python3 inferencia.py imagem.jpg best.pt

# Inferência com visualização das deteções
python3 ver_resultado.py imagem.jpg best.pt
```

Consultar o **Manual de Instalação e Operação em Linux** (PDF) para instruções detalhadas.

## Dataset

- **Principal:** [pyronear/pyro-sdis](https://huggingface.co/datasets/pyronear/pyro-sdis) (HuggingFace) — 29.537 imagens de treino e 4.099 de validação
- **Sintético:** 598 imagens geradas por IA, anotadas manualmente (classes: fumo e fogo)

## Tecnologias

Python · Ultralytics (YOLOv8) · PyTorch · MATLAB · Deep Learning Toolbox · Computer Vision Toolbox · Google Colab · WSL/Ubuntu · ONNX
