# MECE-694 Predictive Maintenance Optimization

Predictive maintenance system for aircraft engines using RUL (Remaining Useful Life) prediction and multi-objective optimization with fuzzy logic.

## Overview

This project implements an intelligent maintenance scheduling system for the NASA C-MAPSS turbofan engine dataset (FD001). It combines deep learning for RUL prediction with fuzzy logic and genetic algorithms to optimize maintenance decisions across multiple objectives.

## Pipeline

### Step 1: Data Processing & Baseline Model
**File:** `step1_ml_pipeline.py`
- Loads C-MAPSS FD001, scales features, trains baseline MLP

### Step 2: Advanced RUL Prediction
**File:** `step2_model.py`
- CNN-LSTM model with sliding windows, outputs `best_cnn_lstm_rul.h5`

### Step 3: Maintenance Optimization

#### 3.1 Repair Threshold Selection
**File:** `step3_1_before_SelectRepairThreshold.py`
- Calculates optimal repair threshold from prediction bias

#### 3.2 Fuzzy Logic System
**File:** `step3_2_fuzzy_viz.py`
- Fuzzy inference: RUL + Importance → Urgency score, generates LUT

#### 3.3 Multi-Objective Optimization
**File:** `step3_3_optimization.py`
- NSGA-II optimization: minimize cost, maximize task completion

## Key Features

- **Dynamic Simulation:** Models engine states (operational, maintenance, repair) with RUL degradation
- **Fuzzy Decision Logic:** Adaptive urgency scoring based on RUL and mission importance
- **Multi-Objective Trade-offs:** Balances cost vs. operational availability
- **Importance Weighting:** Critical missions (importance=3) receive 2x impact multiplier

## Environment Setup

### Prerequisites
- Python 3.9+
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/MYronlei/MECE-694-project.git
cd MECE-694-project
```

2. **Create virtual environment (recommended):**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install numpy pandas scikit-learn tensorflow scikit-fuzzy deap matplotlib
```

## Usage

```bash
# Step 1: Preprocess data and baseline model
python step1_ml_pipeline.py

# Step 2: Train CNN-LSTM RUL predictor
python "step2_model.py"

# Step 3.2: Visualize fuzzy logic system
python step3_2_fuzzy_viz.py

# Step 3.3: Run optimization (requires output from previous steps)
python step3_3_optimization.py
```

## Outputs

- `output/test_processed.csv` - Processed test dataset
- `best_cnn_lstm_rul.h5` - Trained RUL prediction model
- `output/rul_predictions_per_engine.csv` - Per-engine RUL predictions
- `optimization_results/pareto_front_window*.csv` - Optimal maintenance schedules

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) - FD001 subset
- 100 train engines, 100 test engines
- 21 sensor readings per cycle
- Run-to-failure trajectories


## Authors

Mengyu Lei and Yongru Pan
