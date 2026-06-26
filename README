# EvoForest 2.0 — Evolutionary Hyperparameter Optimization for Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Credit%20Card%20Fraud%20(Kaggle)-orange)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> A custom evolutionary algorithm that automatically searches for optimal Decision Tree hyperparameters, applied to highly imbalanced fraud detection. Achieves **F1 = 0.856** on the minority class, outperforming a standard Random Forest baseline.

---

## Overview

EvoForest 2.0 frames hyperparameter optimization as a **biological evolution problem**: a population of Decision Trees evolves across generations through selection, crossover, and mutation — converging on configurations that maximize detection of rare fraud events.

This is **not** a wrapper around existing AutoML tools. The evolutionary engine is implemented from scratch, with four novel mechanisms:

| Mechanism | Description |
|---|---|
| **Hall of Fame** | Preserves the best trees found across all generations, preventing regression |
| **Crossover** | Combines structural parameters (depth, criterion) from one parent with sampling parameters from another |
| **Directed Evolution** | Biases mutation toward hyperparameter values historically associated with winning trees (70% guided / 30% random) |
| **Internal Validation** | Uses a held-out validation split during evolution to prevent overfitting; final evaluation on a separate test set |

---

## Architecture

```
EvoForest 2.0
├── Data Pipeline
│   ├── Stratified 3-way split (Train / Validation / Test)
│   └── StandardScaler on Amount and Time features
│
├── Evolution Engine
│   ├── Population Initialization     — random hyperparameter sampling
│   ├── Fitness Evaluation            — F1 on validation set
│   ├── Selection                     — rank-based elite preservation
│   ├── Crossover                     — single-point parameter crossover
│   ├── Directed Mutation             — history-biased perturbation
│   ├── Hall of Fame                  — global best preservation
│   └── Early Stopping                — patience-based termination
│
└── Final Evaluation
    ├── Champion tree from Hall of Fame → Test set
    └── Comparison against Random Forest baseline (100 estimators)
```

---

## Results

Evaluated on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.172% fraud rate):

| Metric | EvoForest 2.0 | Random Forest (100 estimators) |
|---|---|---|
| **F1 (Fraud class)** | **0.8556** | 0.8391 |
| Recall | **0.7857** | 0.7449 |
| Precision | 0.9390 | **0.9605** |
| Training Time | 844.8s | 134.4s |
| Model complexity | **1 tree** | 100 trees |

Dataset: 284,807 transactions · 492 fraud cases (0.173%) · Test set: 56,962 samples

**Confusion matrix (EvoForest 2.0 on test set):**
```
TN = 56,859  |  FP = 5
FN = 21      |  TP = 77
```

EvoForest 2.0 achieves higher F1 and higher Recall than the Random Forest baseline using a **single evolved Decision Tree** — catching more fraud cases while maintaining near-identical precision. The trade-off is training time; once evolved, inference cost is minimal.

**Hall of Fame — top 3 champions discovered:**
| Rank | Val F1 | depth | features | criterion |
|---|---|---|---|---|
| #1 | 0.8593 | 4 | 0.7 | gini |
| #2 | 0.8571 | 10 | 0.7 | gini |
| #3 | 0.8529 | 4 | 0.9 | gini |

Evolution converged at generation 8 (Early Stopping triggered after 3 generations without improvement).

---

## Hyperparameter Search Space

```python
PARAM_SPACE = {
    "max_depth":          [3, 4, 5, 6, 7, 8, 10, 12, 15],
    "min_samples_split":  [2, 5, 10, 20, 50],
    "min_samples_leaf":   [1, 2, 5, 10, 20],
    "max_features":       [0.3, 0.5, 0.7, 0.9, "sqrt", "log2"],
    "criterion":          ["gini", "entropy"],
    "class_weight":       ["balanced", None, {0:1, 1:5}, {0:1, 1:10}],
}
```

---

## Configuration

All experiment parameters are centralized in `CONFIG`:

```python
CONFIG = {
    "population_size":    20,    # trees per generation
    "n_generations":      15,    # maximum generations
    "mutation_rate":      0.3,   # per-parameter mutation probability
    "elite_size":         4,     # trees passed unchanged to next generation
    "hall_of_fame_size":  3,     # globally preserved champions
    "patience":           3,     # early stopping threshold
    "directed_evolution": True,  # enable history-biased mutation
    "test_size":          0.2,
    "val_size":           0.15,
    "metric":             "f1",
    "random_seed":        42,
}
```

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/mhmoudmarie/evoforest-2.0.git
cd evoforest-2.0

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle and place it at:
# data/creditcard.csv

# Run the experiment
python evoforest.py
```

**On Kaggle:** Update `data_path` in `CONFIG` to `/kaggle/input/creditcardfraud/creditcard.csv` and run the notebook directly.

---

## Repository Structure

```
evoforest-2.0/
├── evoforest.py          # Main source code (self-contained)
├── notebook.ipynb        # Kaggle notebook version
├── requirements.txt      # Python dependencies
├── LICENSE               # Apache 2.0
└── README.md
```

---

## Design Decisions

**Why a single Decision Tree, not an ensemble?**
The goal is to demonstrate that evolutionary search can extract high performance from a single interpretable model — a contribution distinct from simply adding more trees.

**Why Directed Evolution over pure random mutation?**
Pure random mutation wastes compute re-exploring regions already shown to underperform. Recording winning hyperparameter values and biasing future mutations toward them concentrates search effort where it is most productive, analogous to how biological evolution exploits successful genetic patterns.

**Why a separate validation split inside the evolutionary loop?**
Using test data during evolution would cause information leakage. The validation split provides an unbiased fitness signal per generation; the test set is touched only once, at final evaluation.

---

## Dependencies

- Python ≥ 3.8
- numpy
- pandas
- scikit-learn

---

## Author

**Mahmoud Obeidat**  
Data Science & Artificial Intelligence, Jordan  
[GitHub](https://github.com/mhmoudmarie) · [Kaggle](https://www.kaggle.com/mhmoudobeidat)

---

## License

Copyright 2026 Mahmoud Obeidat
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
