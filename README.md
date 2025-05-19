# Loan Default Prediction Analysis

## Overview
This Jupyter notebook implements a machine learning pipeline to predict loan default risk. The analysis follows the steps of data collection, visualization, preprocessing, feature engineering, model development, evaluation, and provides recommendations for lenders.

## Table of Contents
1. [Data Collection](#data-collection)  
2. [Data Visualization](#data-visualization)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Data Analysis](#data-analysis)  
5. [Feature Engineering](#feature-engineering)  
6. [Model Development](#model-development)  
7. [Model Evaluation](#model-evaluation)  
8. [Results & Recommendations](#results--recommendations)  
9. [Usage Instructions](#usage-instructions)  
10. [Contact](#contact)  

## Data Collection
- **Training Data:** Contains borrower features (demographics, financials) and the target label `Risk_Flag` (0 = no default, 1 = default).  
- **Test Data:** Contains the same input features with an identifier column `Id`, but no target label. Reserved for final predictions.

## Data Visualization
- Bar charts of default rates by categorical variables (Marital Status, House Ownership, Car Ownership, Profession, City, State).  
- KDE plots comparing distributions of numeric features (Income, Age, Experience, Job Tenure, House Tenure) for defaulters vs. non-defaulters.

## Data Preprocessing
1. **Duplicate Removal & Column Normalization:** Remove repeated records; standardize column names (lowercase, no whitespace).  
2. **ID Column Removal:** Drop `Id` column to avoid leakage.  
3. **Encoding:** Convert categorical and boolean fields to numeric.  
4. **Out-of-Fold Target Encoding:** Apply to high-cardinality features (Profession, City, State) to generate risk scores without leakage.

## Data Analysis
### Correlation & Mutual Information
- Compute Pearson correlation between features and `Risk_Flag`.  
- Assess non-linear relationships with mutual information.

### Interaction Heatmap
- Bin Income and Job Tenure into quintiles and visualize default rates to justify interaction features.

## Feature Engineering
- **Job Stability:** `current_job_yrs / (experience + 1)`  
- **Residence Stability:** `current_house_yrs * (1 + house_owned)`  
- **Age Buckets:** Discretize age into cohorts to capture non-linear effects.  
- **Interaction Terms:** `income * job_stability` and `income * profession_risk`.

## Model Development
- **Data Split:** 60% training (with SMOTE), 20% validation, 20% test (stratified).  
- **Hyperparameter Tuning:** Manual validation curves for:  
  - Logistic Regression (`C`)  
  - Decision Tree (`max_depth`, `min_samples_leaf`, `max_features`)  
  - Random Forest (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`)
- **Final Model:** Random Forest with 300 trees, `max_depth=15`, `min_samples_leaf=5`, `max_features='sqrt'`.

## Model Evaluation
- **Metrics:** ROC AUC, precision, recall, F1-score on training, validation, and test sets.  
- **Lift Curve:** Shows concentration of defaulters across risk deciles.

## Results & Recommendations
1. **Key Risk Factors:** Stability ratios, age cohorts, geographic/occupational risk scores, interaction terms.  
2. **Model Performance:** Training AUC ~0.98, Validation/Test AUC ~0.93/~0.92, recall of defaults ~0.74.  
3. **Recommended Actions:**  
   - Integrate stability ratios into underwriting.  
   - Tailor policies for youngest/oldest age buckets.  
   - Adjust pricing/documentation by profession and region.  
   - Monitor top risk deciles with targeted outreach.  
   - Retrain model quarterly to adapt to new conditions.

## Usage Instructions
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

