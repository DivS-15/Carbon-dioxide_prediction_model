# 🌍 Climate Change CO₂ Emissions Modeling

This project presents an end-to-end machine learning workflow for analyzing and predicting CO₂ emissions using a climate change dataset. It includes exploratory data analysis, feature engineering, model building with XGBoost, target encoding, log transformations, and SHAP explainability.

---

## 📁 Dataset Overview

- Source: Climate change indicators dataset (`climate_change_dataset.csv`)
- Key target variable: `CO2 Emissions (Tons/Capita)`
- Country-year level data including:
  - Population
  - Renewable Energy (%)
  - Forest Area (%)
  - Avg Temperature
  - Rainfall
  - Sea Level Rise
  - Extreme Weather Events

---

## 🧪 Exploratory Data Analysis (EDA)

- Checked data types, nulls, duplicates
- Log-transformed skewed variables
- Heatmap of correlation between numeric features
- Visualized:
  - CO₂ distribution (raw and log)
  - CO₂ by country (boxplot)
  - CO₂ trends over time (lineplot)
  - Population vs Total CO₂ Emissions (raw and log scale)
  - Colored scatter by Renewable Energy %

### 🔍 Key Insight
> Total CO₂ emissions increase with population, but countries with higher renewable energy tend to emit less at similar population levels.

---

## ⚙️ Feature Engineering

- `Total CO2 Emissions` = Population × Per Capita CO₂
- `Energy Intensity` = CO₂ / (Renewable Energy % + 1)
- `CO2 per Capita Proxy` = CO₂ / (Population + 1)
- Log transform on: Population, Rainfall, Extreme Weather Events

---

## 🤖 Model Building (XGBoost)

### 📌 Pipeline
- `MinMaxScaler` for numeric features
- `TargetEncoder` for `Country`
- Log-transform on target using `np.log1p`
- `ColumnTransformer` to combine preprocessing
- Model: `XGBRegressor` with tuned hyperparameters

### 🔁 Cross-Validation
- 7-fold `KFold` with shuffling
- Metrics: RMSE, R² Score

### 📈 Performance
- CV RMSE: ~1.39  
- Final R² Score: ~0.96

---

## 🔎 Explainability with SHAP

- SHAP used to break down feature contributions per prediction
- Summary plot shows:
  - `log1p(Population)` and `Renewable Energy (%)` were most impactful
  - Higher renewables → lower CO₂ predictions
  - High population + low renewables → high emissions

---

## 📊 Visualizations

### Correlation Heatmap
![Correlation Heatmap](/correlation_heatmap.png)

### Log-Scale: Population vs Total CO₂
![Log Population vs CO₂](/image.png)

### Colored by Renewable Energy (%)
![Color by Renewable Energy](/renewable_energy_co2.png)

### Model's predicted values vs actual values
![predicted vs actual](/model_prediction_results.png)
---

## 🧾 Key Conclusions

- Emissions scale predictably with population
- Renewable energy reduces emissions even at high population
- SHAP provides actionable interpretability
- This pipeline is reusable for other country-level emission studies

---

## 🧰 Tech Stack

- Python 3.11
- Pandas, NumPy, Seaborn, Matplotlib
- XGBoost
- SHAP
- Scikit-learn
- Category Encoders


