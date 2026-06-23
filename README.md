# Crop Recommendation ML — NPK & Climatic Data

Implementation of machine learning models for crop recommendation based on soil nutrient levels (NPK), soil pH, and climatic variables — inspired by:

> Dey, Ferdous & Ahmed (2024). *"Machine learning based recommendation of agricultural and horticultural crop farming in India under the regime of NPK, soil pH and three climatic variables"* — [Heliyon, doi:10.1016/j.heliyon.2024.e25112](https://doi.org/10.1016/j.heliyon.2024.e25112)

---

## What This Project Does

Given soil and climate inputs — Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH, and rainfall — the system recommends a suitable crop to grow.

Three regression models are trained and compared, along with a Random Forest Classifier for direct crop prediction:

- **Linear Regression**
- **Decision Tree Regression**
- **Random Forest Regression**
- **Random Forest Classifier** (recommendation engine)

---

## Dataset

- Source: [Kaggle — Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- 2200 samples, 22 crop classes
- Features: N, P, K, temperature, humidity, pH, rainfall
- Target: crop label

---

## Results

| Model | MAE | RMSE | R² | Accuracy |
|---|---|---|---|---|
| Linear Regression | — | — | — | — |
| Decision Tree Reg. | — | — | — | — |
| Random Forest Reg. | — | — | — | — |
| RF Classifier | — | — | — | — |

*(Fill in your actual output values after running the script)*

---

## Visualizations Generated

The script produces 6 plots saved as PNG files:

1. `viz1_correlation.png` — Feature correlation heatmap
2. `viz2_distributions.png` — Distribution of all 7 input features
3. `viz3_crop_distribution.png` — Crop class frequency bar chart
4. `viz4_model_comparison.png` — R²/Accuracy comparison across models
5. `viz5_feature_importance.png` — Feature importance from Random Forest
6. `viz6_predicted_vs_actual.png` — Predicted vs actual crop codes

---

## How to Run

**1. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

**2. Download the dataset**

Get `Crop_recommendation.csv` from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) and update the path in the script:
```python
df = pd.read_csv("path/to/Crop_recommendation.csv")
```

**3. Run the script**
```bash
python crop_re.py
```

**4. Interactive prediction**

The script will prompt you to enter N, P, K, temperature, humidity, pH, and rainfall values and output the recommended crop.

---

## Project Structure

```
Crop-Recommendation-ML/
├── crop_re.py               # Main script — EDA, preprocessing, models, visualizations
├── Crop_recommendation.csv  # Dataset (download from Kaggle)
├── viz1_correlation.png
├── viz2_distributions.png
├── viz3_crop_distribution.png
├── viz4_model_comparison.png
├── viz5_feature_importance.png
└── viz6_predicted_vs_actual.png
```

---

## Key Findings from the Reference Paper

The paper (Dey et al., 2024) evaluated five ML models — RF, Decision Tree, SVM, XGBoost, and KNN — on the same dataset. Key results:

- XGBoost achieved the highest accuracy: **99.09%** for agricultural crops
- Individual crop category models outperform combined (mixed) crop models
- RF and XGBoost consistently outperformed SVM, KNN, and Decision Tree

This implementation focuses on the regression-based subset (Linear Regression, Decision Tree Regression, Random Forest Regression) as specified in the internship scope, along with a Random Forest Classifier for crop recommendation.

---

## Acknowledgements

This project was completed as part of a research internship under **Dr. Jagat Jyoti Rath** at NIT Allahabad.
