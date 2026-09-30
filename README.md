# Student Concentration Detection Using Computer Vision

This repository contains the implementation of a multimodal deep-learning model developed to classify student concentration during online learning.

The model predicts three concentration classes:

- **Distracted** — Class 0
- **Partially Focused** — Class 1
- **Focused** — Class 2

## Research Context

This work was developed as part of an Honours research project at Iyunivesithi Walter Sisulu.

**Project title:** Using Computer Vision Techniques for Detecting Student Concentration in Real Time on Online Learning Platforms

The project investigates whether behavioural and visual information from student video clips can be used to classify concentration levels during online learning.

## Final Model: CMOSE Experiment 3

The final model is named:

```text
CMOSE_TemporalConv_BiLSTM_GatedFusion
```

It combines two sources of information:

- **OpenFace behavioural features:** A sequence of 120 frames, with 25 behavioural features per frame. These features represent facial behaviour, head movement, eye gaze, and facial action units.
- **I3D visual embedding:** A 1024-dimensional representation of the visual content of each video clip.

### Architecture

```text
OpenFace input: (120, 25)
  → Layer Normalization
  → Multi-scale Conv1D: 64 filters with kernels 3, 5, and 7
  → Concatenation
  → Layer Normalization and Dropout (0.15)
  → Dense (160, swish)
  → BiLSTM (128 units)
  → Layer Normalization
  → BiLSTM (96 units)
  → Layer Normalization
  → Temporal Attention (128 units)
  → Dense (192, swish)
  → Dropout (0.25)

I3D input: (1024,)
  → Layer Normalization
  → Dense (512, swish)
  → Dropout (0.25)
  → Dense (256, swish)
  → Layer Normalization
  → Dropout (0.20)
  → Dense (192, swish)

Fusion and classification:
  → Gated Fusion (256 units)
  → Layer Normalization
  → Dense (256, swish)
  → Dropout (0.35)
  → Dense (128, swish)
  → Dropout (0.25)
  → Softmax output (3 classes)
```

## Final Test Results

The final Experiment 3 model was evaluated on **1,173 held-out CMOSE test samples**.

| Metric | Result |
|---|---:|
| Accuracy | 75.79% |
| Balanced Accuracy | 69.04% |
| Macro Precision | 68.35% |
| Macro Recall | 69.04% |
| Macro F1-score | 68.35% |
| Weighted F1-score | 76.24% |
| Macro ROC-AUC | 86.26% |

### Class-Level F1 Scores

| Class | F1-score |
|---|---:|
| Distracted | 61.28% |
| Partially Focused | 82.86% |
| Focused | 60.91% |

## Repository Contents

```text
model/
├── custom_layers.py     # TemporalAttention and GatedFusion layers
└── architecture.py      # CMOSE Experiment 3 model architecture

requirements.txt         # Python packages used in the development environment
```

## Development Environment

- Python 3.11.9
- TensorFlow 2.21.0
- NumPy 2.4.6
- Pandas 3.0.5
- Scikit-learn 1.9.0
- Matplotlib 3.11.1
- OpenCV 5.0.0
- MediaPipe 1.0.0
- Windows 10 (64-bit)

## Installation

Clone the repository:

```bash
git clone [https://github.com/thobel-hash/student-concentration-detection.git](https://github.com/thobel-hash/student-concentration-detection.git)
cd student-concentration-detection
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Display the architecture summary:

```bash
python model/architecture.py
```

## Dataset

This project used the **Comprehensive Multi-Modality Online Student Engagement (CMOSE)** dataset.

The CMOSE dataset, raw videos, extracted feature files, processed NumPy arrays, and trained model weights are **not included** in this repository. They are excluded because of dataset-access restrictions and file-size limitations.

Researchers who wish to reproduce the experiments should obtain the dataset from its original authors and follow the preprocessing procedures described in the research report and notebook.

## Important Note

The model was evaluated experimentally using held-out video data. A live deployment on an online learning platform was outside the scope of the study.

## Author

**Gili-gili Thobela**  
BSc Honours in Computer Science  
Iyunivesithi Walter Sisulu 
Department of Mathematical Sciences and Computing  

Supervisor: Dr. William T. Vambe
