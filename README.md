# Smart Student Dropout Prediction & Performance Analysis System

> **Academic Minor Project — Phase 1: Problem Formulation, Dataset Acquisition, and Preprocessing Pipeline**

An AI-driven Early-Warning System designed to identify students at risk of academic dropout and forecast academic trajectories early in the degree lifecycle.

---

## 📌 Project Overview (Phase 1)
Higher educational institutions face significant student attrition, often identified too late (after semester examinations) for corrective intervention. This project develops a proactive decision-support system utilizing Supervised Machine Learning.

In **Phase 1**, the primary focus is on:
1. **Dataset Ingestion & Quality Audit:** Collecting and verifying real-world academic data from the UCI Machine Learning Repository.
2. **Exploratory Data Analysis (EDA):** Identifying socio-economic and academic risk indicators, and auditing class imbalance.
3. **Data Preprocessing & Leakage Prevention:** Engineering features, standardizing schemas, and implementing stratified data splits to prevent future data leakage.
4. **Architecture Design:** Structuring the baseline machine learning pipeline for Phase 2 training.

---

## 📊 Dataset Specifications
- **Source:** [UCI Machine Learning Repository: Predict Students' Dropout and Academic Success (ID: 697)](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Total Records:** 4,424 student profiles
- **Total Attributes:** 36 input variables spanning demographic, socio-economic, pre-enrollment qualifications, and 1st/2nd semester curricular units.
- **Target Distribution:**
  - *Graduate:* 2,209 students (49.9%)
  - *Dropout:* 1,421 students (32.1%)
  - *Enrolled:* 794 students (18.0%)
- **Class Imbalance:** Dropout instances represent ~32% of total outcomes, necessitating cost-sensitive learning (`class_weight='balanced'`) and stratified sampling.

---

## 🏗️ Phase 1 System Pipeline

```mermaid
flowchart LR
    A["Raw Data (UCI ID 697)"] --> B["Data Cleaning & Normalization"]
    B --> C["Feature Engineering"]
    C --> D["Stratified 80:20 Train/Test Split"]
    D --> E["Phase 1 Output Datasets (train.csv / test.csv)"]
    E -.-> F["[Planned Phase 2] Multi-Task ML Training"]
```

### Key Preprocessing Highlights:
- **Zero Future Data Leakage:** 2nd-semester attributes are strictly isolated from early-warning features so that the early detection model relies solely on pre-enrollment and 1st-semester observations.
- **Stratified Partitioning:** An 80:20 split (`train.csv` with 3,539 rows, `test.csv` with 885 rows) preserving the 32% minority dropout ratio across both splits.
- **Feature Engineering:** Derived academic stability indices (`sem1_approval_rate`, `sem1_eval_ratio`).

---

## 📂 Repository Structure (Phase 1)

```text
├── data/
│   ├── raw/
│   │   ├── data.csv                 # Raw UCI dataset (4,424 rows)
│   │   └── dataset_info.json        # Metadata and feature descriptions
│   └── processed/
│       ├── train.csv                # 80% Stratified training set (3,539 rows)
│       ├── test.csv                 # 20% Stratified testing set (885 rows)
│       └── data_summary.json        # Class distribution and feature inventory
├── notebooks/
│   └── 01_eda_and_data_analysis.ipynb # Interactive EDA & visualization plots
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── download_data.py         # Automated dataset download script
│       └── preprocessor.py          # Data cleaning and pipeline execution
├── .gitignore
├── requirements.txt                 # Dependencies
└── README.md
```

---

## 🚀 Execution & Verification

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Phase 1 dependencies
pip install -r requirements.txt
```

### 2. Run Preprocessing Pipeline
To execute the automated cleaning and stratified train-test split:
```bash
python -m src.data.preprocessor
```

### 3. Launch Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_eda_and_data_analysis.ipynb
```

---

## 🔮 Roadmap / Next Phases
- **Phase 2 (Upcoming):** Multi-task model training (Random Forest Classifier & Regressor), 5-Fold Stratified Cross-Validation, and hyperparameter tuning with `GridSearchCV`.
- **Phase 3 (Upcoming):** Deployment into a decision-support web application featuring risk meters, what-if simulations, and counselor alert mechanisms.
