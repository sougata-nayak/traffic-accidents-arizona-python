# 🚦 Arizona Traffic Accident Severity Prediction
> A Data Mining & Machine Learning project focused on predicting injury severity in fatal traffic accidents across Arizona using real-world crash data (2012–2016).

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Project-Completed-success)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-red)

---

## 📌 Project Overview

Traffic accidents remain a major public safety concern, especially when analyzing the severity of injuries sustained during crashes.  
This project applies **Data Mining** and **Machine Learning** techniques to analyze fatal traffic accident data from Arizona and predict the severity of injuries using demographic, environmental, and accident-related features.

The project was developed as part of the **CSE 572 - Data Mining** Master's final project at **Arizona State University**.

### 🎯 Objectives

- Analyze traffic accident patterns in Arizona
- Understand how demographic and environmental factors influence injury severity
- Build predictive machine learning models for injury severity classification
- Compare multiple ML algorithms and evaluate their performance
- Identify the most effective predictive model for real-world deployment

---

## 📂 Dataset

The dataset used for this project was sourced from Kaggle:

🔗 Dataset Link: https://www.kaggle.com/datasets/thedevastator/fatal-traffic-accidents-in-arizona-2012-2016/data

### Dataset Files Used

| File | Description |
|---|---|
| `person.csv` | Demographic and injury-related information |
| `accident.csv` | Accident conditions such as weather, lighting, road type, etc. |

### Important Features

Some of the major features used during training:

- `AGE`
- `SEX`
- `INJ_SEV`
- `WEATHER`
- `LGT_COND`
- `A_ROADFC`
- `DRINKING`
- `ALC_DET`
- `PERSONS`
- `MAN_COLL`

---

# 🧠 Problem Statement

The primary goal of this project is to predict the **injury severity (`INJ_SEV`)** sustained during a traffic accident.

The injury severity score ranges from **0 to 9**, representing increasing levels of injury severity.

This prediction problem is highly valuable for:

- Emergency response planning
- Accident prevention strategies
- Public safety analysis
- Resource allocation
- Traffic management systems

---

# 🛠️ Tech Stack

## Languages & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

# 🔄 Project Pipeline

The project follows a complete end-to-end machine learning workflow:

```text
Data Collection
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Exploratory Data Analysis
       ↓
Model Training
       ↓
Hyperparameter Tuning
       ↓
Evaluation & Comparison
       ↓
Final Model Selection
```

---

# 🧹 Data Cleaning & Preprocessing

The raw dataset contained:

- Missing values
- Invalid demographic records
- Noisy categorical values
- High-cardinality columns
- Duplicate or irrelevant features

### Cleaning Steps Performed

✅ Removed unrealistic age values (`998`, `999`)  
✅ Dropped columns with excessive null values  
✅ Removed irrelevant identifiers  
✅ Merged accident-level and person-level datasets  
✅ Encoded categorical variables  
✅ Created age bands using binning  
✅ Standardized features where required  

---

# 📊 Exploratory Data Analysis (EDA)

The project involved extensive analysis of accident trends and severity patterns.

## Age vs Injury Severity

The analysis showed that injury severity generally increases with age, with elderly individuals being more vulnerable during traffic accidents.

<img src="readme_assets/eda_analysis_page.png" alt="EDA Analysis" width="650">

### Key Insights

- Younger age groups had comparatively lower severity scores
- Severity increased consistently with age
- Environmental conditions strongly influenced outcomes
- Multi-vehicle accidents showed non-linear severity patterns

---

# ⚙️ Feature Engineering

Feature engineering played a major role in improving model performance.

### Important Engineering Techniques

## 1. Age Banding

Ages were divided into deciles to capture demographic trends more effectively.

```python
pd.cut(data['AGE'], 10)
```

## 2. Dataset Merging

The project merged:

- `person.csv`
- `accident.csv`

This enriched the feature space with environmental variables.

## 3. Feature Selection

Several methods were explored:

- Correlation Analysis
- Recursive Feature Elimination (RFE)
- Random Forest Feature Importance

---

# 🤖 Machine Learning Models Used

The project evaluated multiple classification models.

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable tree-based classifier |
| Random Forest | Ensemble learning |
| SVM | High-dimensional classification |
| Linear SVC | Efficient linear SVM |
| MLP Classifier | Neural network-based classifier |
| SGD Classifier | Stochastic optimization |
| XGBoost | Gradient boosting ensemble |

---

# 🌟 Preferred Model — XGBoost

XGBoost emerged as the best-performing model because of:

- Strong generalization capability
- Robust handling of heterogeneous data
- Reduced overfitting through regularization
- Excellent performance on structured tabular datasets
- Efficient handling of feature interactions

### Hyperparameters Used

```python
{
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'n_estimators': 100,
    'subsample': 0.8
}
```

---

# 📈 Model Performance

## Final Results

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| XGBoost | 86.95% | 82.26% | 0.76 | 0.65 | 0.66 |
| Random Forest | 82.42% | 77.65% | 0.68 | 0.54 | 0.56 |
| SVM | 70.67% | 69.93% | 0.53 | 0.51 | 0.51 |
| Decision Tree | 79.14% | 71.77% | 0.45 | 0.43 | 0.44 |
| Logistic Regression | 66.66% | 66.01% | 0.39 | 0.35 | 0.34 |

![Model Comparison](readme_assets/results_table_page.png)

---

# 📚 Concepts Used

## 🔹 XGBoost

XGBoost is an optimized gradient boosting framework that builds decision trees sequentially to minimize prediction errors.

### Why XGBoost?

- Faster training
- Better regularization
- Handles missing values efficiently
- Works exceptionally well with tabular data

---

## 🔹 Random Forest

Random Forest combines multiple decision trees to reduce overfitting and improve prediction stability.

---

## 🔹 Feature Engineering

Feature engineering transforms raw data into meaningful representations that improve machine learning performance.

---

## 🔹 Cross Validation

Cross-validation ensures the model generalizes well to unseen data by validating performance across multiple splits.

---

# 📁 Repository Structure

```text
.
├── data/
│   ├── accident.csv
│   └── person.csv
│
├── notebooks/
│   └── main.ipynb
│
├── readme_assets/
│   ├── eda_analysis_page.png
│   └── results_table_page.png
│
├── README.md
└── requirements.txt
```

---

# ▶️ How To Run The Project

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/sougata-nayak/traffic-accidents-arizona-python.git
cd traffic-accidents-arizona-python
```

---

## 2️⃣ Create a Virtual Environment

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Add Dataset Files

Download the dataset from Kaggle and place:

- `person.csv`
- `accident.csv`

inside the `data/` folder.

---

## 5️⃣ Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
main.ipynb
```

Run all cells sequentially.

---

# 📦 Example Requirements.txt

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

---

# 🔍 Key Findings

- Injury severity increases with age
- Environmental conditions affect accident outcomes
- Ensemble learning models outperform traditional linear models
- XGBoost provided the best balance between accuracy and generalization
- Feature engineering significantly improved predictive performance

---

# 🚀 Future Improvements

Possible future enhancements:

- Real-time accident prediction systems
- GPS-based accident analysis
- Deep learning approaches
- Geospatial visualization dashboards
- Live weather API integration
- Deployment as a web application

---

# 📖 References

1. Kaggle Arizona Traffic Accident Dataset  
2. Scikit-learn Documentation  
3. XGBoost Documentation  
4. Introduction to Statistical Learning  
5. Python for Data Analysis — Wes McKinney

---

# ⭐ Acknowledgements

This project was completed as part of the **CSE 572 — Data Mining** course at **Arizona State University**.

The project demonstrates the practical application of:

- Data Mining
- Machine Learning
- Predictive Analytics
- Feature Engineering
- Model Evaluation

towards solving real-world public safety challenges.

---

# 📬 Contact

For questions, collaboration, or feedback:

📧 Sougata Nayak  
GitHub: [https://github.com/](https://github.com/sougata-nayak)

---

## ⭐ If you found this project useful, consider giving the repository a star!
