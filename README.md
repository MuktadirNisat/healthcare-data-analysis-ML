# 🏥 End-to-End Healthcare Data Cleaning, Statistical Analysis & ML Pipeline

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Cleaning-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-F7931E?logo=scikit-learn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Hypothesis%20Testing-8CAAE6?logo=scipy&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview
This repository presents a complete, production-style data science workflow applied to a raw, uncleaned healthcare dataset containing **55,505 patient records across 15 features**. The project covers every stage of the data lifecycle—from resolving real-world data anomalies and standardizing mixed formats to performing statistical hypothesis testing (`SciPy`) and building a leak-free machine learning classification pipeline (`Scikit-Learn`).

---

## 📂 Repository Structure
```text
healthcare-data-cleaning/
├── data/
│   └── hospital_data_unclean.csv        # Raw dataset (55,505 rows, 15 columns)
├── scripts/
│   └── clean_data.py                    # Modular Python data cleaning script
├── healthcare_data_analysis.ipynb           # Full EDA, Hypothesis Testing & ML Notebook
├── eda_dashboard.png                    # Multi-panel Exploratory Data Analysis visual
├── ml_evaluation.png                    # Confusion Matrix & Feature Importance plot
├── requirements.txt                     # Project dependencies
├── .gitignore                           # Ignored environment & large export files
└── README.md                            # Project documentation
🧹 Stage 1: Data Cleaning & Feature EngineeringRaw clinical records contained ~4,440 missing values per column, inconsistent delimiters, currency suffixes, and malformed categorical labels. The following transformations were applied:Column(s)Data Issue IdentifiedCleaning & Standardization StrategyNameInconsistent capitalization & NaNStandardized to Title Case (.str.title()); missing values imputed as 'Unknown'GenderMixed labels (M, FEMALE) & NaNMapped to 'Male' / 'Female'; imputed missing entries using ModeBlood TypeRepeated signs (A++, AB+-, O--)Collapsed duplicate symbols via Regex (r'[+-]+') and imputed missing values with ModeDoctor & HospitalStray symbols (#, ,) & NaNStripped punctuation artifacts and filled missing records with 'Unknown'Insurance ProviderMissing whitespace (UnitedHealthcare)Standardized spacing (United Healthcare) and filled nulls as 'Unknown'Billing AmountString format with ' USD' suffixStripped ' USD', cast to float, imputed nulls with Mean ($25,553.79), rounded to 2 decimalsTest ResultsLowercase/abbreviated (abn, N, unknown)Normalized to 'Abnormal', 'Normal', and 'Unknown'Date of Admission & Discharge DateMixed formats (YYYY.MM.DD, DD/MM/YYYY)Unified separators, applied two-pass pd.to_datetime parsing (dayfirst=True fallback), imputed nulls with MedianFeature EngineeringRaw dates onlyEngineered Stay_Days, Admission_Year, and Admission_Month📊 Stage 2: Exploratory Data Analysis (EDA)Key insights uncovered from the cleaned dataset:Patient Demographics: Patient ages range from 13 to 89 years with a mean of 51.58 years ($\sigma = 18.79$).Billing Distribution: Average treatment cost is $25,553.79 (max $52,764.28). Identified 99 negative billing entries (down to -$2,008.49), representing hospital refunds or billing adjustments.Condition-Wise Cost: Mean treatment costs remain consistent across conditions, led slightly by Obesity ($25,743.04) and Asthma ($25,711.24).🧪 Stage 3: Statistical Hypothesis Testing ($\alpha = 0.05$)To validate whether observed patterns were statistically significant or due to random variation, three formal tests were conducted:Hypothesis TestVariables TestedTest Statisticp-valueStatistical ConclusionTwo-Sample T-TestBilling Amount (Male vs. Female)$t = 0.9581$0.3380Fail to Reject $H_0$: No significant difference in billing by gender.One-Way ANOVABilling Amount across 6 Medical Conditions$F = 2.4197$0.0335Reject $H_0$: Statistically significant variation exists across conditions.Chi-Square TestAdmission Type vs. Test Results$\chi^2 = 0.2591$0.8785Fail to Reject $H_0$: Admission urgency and test outcomes are independent.🤖 Stage 4: Machine Learning Pipeline (Scikit-Learn)Built a leak-free ColumnTransformer + Pipeline workflow (StandardScaler for numerical features, OneHotEncoder for categorical features) using a Random Forest Classifier (n_estimators=150, max_depth=10) to predict binary lab outcomes (Normal vs. Abnormal).Test Accuracy: 51.16%Analytical Takeaway: The near-baseline classification accuracy aligns directly with the statistical hypothesis tests ($r \approx 0.00$ correlation matrix and $\chi^2$ $p = 0.8785$), confirming that the underlying synthetic dataset features are statistically independent rather than suffering from model underfitting.