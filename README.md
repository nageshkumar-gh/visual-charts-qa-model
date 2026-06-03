# Visual Question Answering From Charts

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![T5](https://img.shields.io/badge/Model-T5--small-blueviolet)
![FasterRCNN](https://img.shields.io/badge/Model-Faster%20R--CNN-orange)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-00FFFF?logo=ultralytics&logoColor=black)
![Dataset](https://img.shields.io/badge/Dataset-ChartQA-4CAF50)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-66.25%25-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

A modular VQA system that answers natural language questions about chart images by combining YOLO-based visual element detection with a fine-tuned T5 language model.

---

## Overview

Traditional VQA systems struggle with charts because charts are hybrid artifacts encoding structured numerical/categorical data visually. This system takes a structure-aware approach:

1. **YOLO** detects and localizes chart elements (bars, axes, legends, titles, pie slices)
2. **Annotation metadata** maps detected elements to text labels (bypassing OCR)
3. **T5** receives a flattened text representation of the chart and answers the question

This avoids OCR errors and enables interpretable, text-based reasoning over visual chart data.

---

## Pipeline

```
Chart Image + Question
        │
        ▼
  Feature Extraction
  (YOLOv8 / Faster R-CNN)
        │
        ▼
  Text Extraction from Annotations
  (axis labels, title, legend, data values)
        │
        ▼
  Linearized Input → T5 Model
  "Question: ... Table: ... Chart Features: ..."
        │
        ▼
     Answer
```

---

## Dataset

Built on **ChartQA** (Masry et al., 2022) — 18,317 samples across four chart types:

| Chart Type     | Train       | Val        | Test       |
|----------------|-------------|------------|------------|
| Vertical Bar   | 7110 (55%)  | 1524 (55%) | 1524 (55%) |
| Horizontal Bar | 3796 (30%)  | 814 (30%)  | 814 (30%)  |
| Line           | 1536 (12%)  | 329 (12%)  | 329 (12%)  |
| Pie            | 379 (3%)    | 81 (3%)    | 81 (3%)    |
| **Total**      | **12,821**  | **2,748**  | **2,748**  |

QA pairs include augmented (rule-generated) and human-annotated samples covering numeric, alphabetic, and yes/no question types.

---

## Project Structure

```
visual-qa-model/
├── src/
│   ├── Data/
│   │   ├── Dataset/
│   │   │   ├── augmented.json        # 20,901 synthetic QA pairs
│   │   │   └── human.json            # 7,398 human-annotated QA pairs
│   │   ├── train/
│   │   │   ├── input/                # CSV inputs for T5 training
│   │   │   ├── coco_files/           # COCO-format annotations
│   │   │   ├── train_human.json
│   │   │   └── train_augmented.json
│   │   ├── val/                      # Same structure as train/
│   │   └── test/                     # Same structure as train/
│   │
│   ├── Model/
│   │   ├── Yolo/
│   │   │   ├── yolo_v4_e50.ipynb     # YOLOv8 training notebook
│   │   │   ├── chart_yolo2_best.pt   # Trained YOLO weights
│   │   │   └── test_single_image.py  # Single image inference
│   │   ├── FastRCNN/
│   │   │   └── Faster_RCNN_v4.ipynb  # Faster R-CNN training notebook
│   │   ├── T5/
│   │   │   ├── t5.py                 # T5 training script
│   │   │   ├── test_t5.py            # T5 inference script
│   │   │   ├── T5_V2.ipynb           # T5 training (with annotations)
│   │   │   ├── T5_V3.ipynb           # T5 training (variant)
│   │   │   ├── T5_wo_annot.ipynb     # T5 training (without annotations)
│   │   │   └── Final/                # Saved fine-tuned T5 model
│   │   ├── feature_extraction_coco/
│   │   │   ├── coco_gen.py           # Convert chart annotations → COCO format
│   │   │   ├── coco_gen_v4.py        # Updated COCO generator
│   │   │   └── conver_anot_yolo.py   # Convert annotations → YOLO format
│   │   ├── input/
│   │   │   └── yolo_dataset.yaml     # YOLO dataset config (10 classes)
│   │   └── output/
│   │       └── ocr_chart_features.csv
│   │
│   ├── Utils/
│   │   ├── split_dataset.py          # Train/val/test splitting
│   │   ├── annotation_processing.py  # Process raw chart annotations
│   │   ├── detect.py                 # Run detection on images
│   │   ├── QA_pair_type.py           # Categorize QA pair types
│   │   └── T5_category_prediction_values.py
│   │
│   └── DataAnalysis/
│       └── DataAnalysis.ipynb        # Dataset exploration and analysis
│
└── docs/
    └── VisualQA_From_Charts-Final_Paper.pdf
```

---

## Models

### YOLO (YOLOv8n) — Chart Element Detection

Detects 10 chart component classes:

| Class        | Precision | Recall |
|--------------|-----------|--------|
| ChartTitle   | 0.976     | 0.985  |
| PlotArea     | 0.969     | 0.998  |
| LegendLabel  | 0.942     | 0.952  |
| xAxisLabel   | 0.953     | 0.975  |
| yAxisLabel   | 0.936     | 0.959  |
| PieLabel     | 0.912     | 0.942  |
| Vertical Bar | 0.981     | 0.937  |
| Horizontal Bar | 0.894   | 0.906  |
| Line         | 0.932     | 0.776  |
| yAxisTitle   | 0.919     | 0.978  |
| **Overall**  | **0.941** | **0.941** |

Trained for 50 epochs, batch size 16, on ~12K annotated chart images (~1.5 hrs on Google Colab Pro).

### Faster R-CNN — Preliminary Experiments

Same 10 classes, COCO-format input. Trained for 3000 iterations (batch size 2, lr 0.00025). Higher precision on small objects but computationally intensive — not used in the main pipeline.

### T5-small — Question Answering

Fine-tuned on chart QA pairs with linearized input format:

```
Question: <question> Table: <flattened chart data> Chart Features: <type, title, axes, legends>
```

- Max input length: 512 tokens | Max answer length: 64 tokens
- Trained for 6 epochs, batch size 16
- **Test accuracy: 66.25%** (vs. ChartQA baseline: 59.80%)

| QA Type | Accuracy |
|---------|----------|
| Integer | 67.41%   |
| Yes     | 66.10%   |
| No      | 66.67%   |
| String  | 57.37%   |

---

## Setup

### Requirements

```bash
pip install torch transformers ultralytics pandas scikit-learn Pillow tqdm
```

### Data Preparation

1. Download ChartQA dataset from HuggingFace and place chart images under `src/Data/{train,val,test}/png/`
2. Place annotation JSON files under `src/Data/{train,val,test}/annotations/`
3. Generate COCO-format labels for Faster R-CNN:
   ```bash
   cd src
   python Model/feature_extraction_coco/coco_gen.py
   ```
4. Generate YOLO-format labels:
   ```bash
   python Model/feature_extraction_coco/conver_anot_yolo.py
   ```
5. Generate T5 CSV inputs:
   ```bash
   python Data/train/gen_train_input_t5.py
   python Data/val/gen_val_input_t5.py
   python Data/test/gen_test_input_t5.py
   ```

### Training

**YOLO:**
Open and run `src/Model/Yolo/yolo_v4_e50.ipynb` (recommended: Google Colab Pro with GPU).

**T5:**
```bash
cd src
python Model/T5/t5.py
```
Or use the notebooks in `src/Model/T5/` for interactive training.

### Inference

**YOLO — single image:**
```bash
cd src
python Model/Yolo/test_single_image.py
```

**T5 — question answering:**
```bash
cd src
python Model/T5/test_t5.py
```

---

## Input Format Variants

Three T5 input strategies were explored (each has corresponding data generation scripts and notebooks):

| Variant | Description |
|---------|-------------|
| `annot` | Full input: question + flattened table + YOLO-extracted chart features |
| `wo_annot` | Question + table only (no chart feature annotations) |
| `wo_table` | Question + chart features only (no table data) |

---

## Limitations

- End-to-end pipeline integration not implemented due to GPU/compute constraints — each module was tested independently
- No OCR: relies on ground-truth annotation metadata; may not generalize to unannotated real-world charts
- Faster R-CNN struggles with small, closely spaced axis labels
- YOLO performance degrades on charts with overlapping or cluttered elements

---

## References

- ChartQA: Masry et al., AAAI 2022
- DVQA: Kafle & Kanan, CVPR 2018
- PlotQA: Methani et al., arXiv 2020
- DePlot: Liu et al., NeurIPS 2022
- MATCHA: Liu et al., ICLR 2022
- T5: Raffel et al., JMLR 2020
- YOLOv3: Redmon & Farhadi, arXiv 2018
- Faster R-CNN: Ren et al., NeurIPS 2015
