# Disease Prediction ML

This project compares four supervised machine learning models for disease prediction using the Kaggle Disease Prediction Medical Dataset.

## Folder Structure

- `models.py` — Standalone script to train and evaluate Logistic Regression, Decision Tree, Random Forest, and SVM models.
- `report.md` — Results, analysis, and code appendix.
- `notebook.ipynb` — (Optional) Jupyter notebook for interactive exploration.
- `requirements.txt` — Python dependencies.
- `disease_prediction.csv` — **Place the Kaggle dataset here.**

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/tanishchavaan/disease-prediction-medical-dataset) and place `disease_prediction.csv` in this folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python models.py
   ```

The script will print evaluation metrics and display confusion matrices for all four models.

## Objectives
- Preprocess the dataset for machine learning
- Train and evaluate four supervised models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- Compare their performance and discuss similarities and differences

## Instructions
1. Download the dataset from Kaggle and place `disease_prediction.csv` in this folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run `notebook.ipynb` in Jupyter Notebook or VS Code.
4. Review the results and analysis in `report.md`. 