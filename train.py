import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

# Load dataset
if 'df' not in globals():
    df = pd.read_csv('emp.csv')

# Encode categorical variables
df_encoded = df.copy()
label_encoders = {}
categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Feature Engineering
df_encoded['Satisfaction_Score'] = (
    df_encoded['EnvironmentSatisfaction'] +
    df_encoded['JobSatisfaction'] +
    df_encoded['RelationshipSatisfaction'] +
    df_encoded['WorkLifeBalance']
) / 4

df_encoded['Experience_Ratio'] = df_encoded['YearsAtCompany'] / (df_encoded['TotalWorkingYears'] + 1)
df_encoded['Years_Since_Promotion'] = df_encoded['YearsAtCompany'] - df_encoded['YearsSinceLastPromotion']
df_encoded['Income_by_Age'] = df_encoded['MonthlyIncome'] / (df_encoded['Age'] + 1)
df_encoded['Work_Stress'] = df_encoded['OverTime'] * (1 - df_encoded['WorkLifeBalance'] / 4)
df_encoded['Job_Stability_Index'] = df_encoded['YearsInCurrentRole'] / (df_encoded['NumCompaniesWorked'] + 1)

df_encoded['Income_Percentile_by_Role'] = df_encoded.groupby('JobRole')['MonthlyIncome'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1)
)
df_encoded['Promotion_Velocity'] = df_encoded['YearsAtCompany'] / (df_encoded['YearsSinceLastPromotion'] + 1)
df_encoded['Training_per_Year'] = df_encoded['TrainingTimesLastYear'] / (df_encoded['TotalWorkingYears'] + 1)
df_encoded['Stock_Income_Ratio'] = df_encoded['StockOptionLevel'] / (df_encoded['MonthlyIncome'] / 10000 + 1)
df_encoded['Age_Experience_Gap'] = np.abs(df_encoded['Age'] - df_encoded['TotalWorkingYears'] - 18)
df_encoded['Department_Seniority'] = df_encoded.groupby('Department')['JobLevel'].transform('rank', pct=True)
df_encoded['Travel_Satisfaction_Impact'] = df_encoded['BusinessTravel'] * (1 - df_encoded['WorkLifeBalance'] / 4)
df_encoded['Overtime_Tenure_Ratio'] = df_encoded['OverTime'] / (df_encoded['YearsInCurrentRole'] + 1)
df_encoded['Career_Progression'] = (df_encoded['JobLevel'] / (df_encoded['YearsAtCompany'] + 1)) * 100
df_encoded['Comp_Satisfaction_Gap'] = (df_encoded['MonthlyIncome'] / 10000) - df_encoded['JobSatisfaction']

# Prepare data
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_base = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")

stacking_clf = StackingClassifier(
    estimators=[('xgb', xgb_base), ('rf', rf_base)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

# Train
stacking_clf.fit(X_train_scaled, y_train)

# ======================
# MODEL EVALUATION
# ======================
y_pred = stacking_clf.predict(X_test_scaled)
y_pred_proba = stacking_clf.predict_proba(X_test_scaled)[:, 1]

print("\n📊 MODEL PERFORMANCE")
print(f"Accuracy   : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision  : {precision_score(y_test, y_pred):.4f}")
print(f"Recall     : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score   : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC    : {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid()
plt.show()

# Save model
joblib.dump(stacking_clf, 'stacking_clf.pkl')

# ======================
# ATTRITION RISK
# ======================
X_full_scaled = scaler.transform(X)
df_encoded['Attrition_Risk'] = stacking_clf.predict_proba(X_full_scaled)[:, 1]

# Critical employees
def is_critical(row):
    return (
        (row['JobLevel'] >= 4) or
        (row['PerformanceRating'] >= 4) or
        (row['YearsAtCompany'] >= 10) or
        (row['Department'] in [
            label_encoders['Department'].transform(['Research & Development'])[0],
            label_encoders['Department'].transform(['Sales'])[0]
        ])
    )

df_encoded['Is_Critical'] = df_encoded.apply(is_critical, axis=1)

critical_employees = df_encoded[
    (df_encoded['Attrition_Risk'] > 0.6) &
    (df_encoded['Is_Critical'])
].copy()

# Retention suggestions
def retention_suggestion(row):
    suggestions = []
    if row['JobSatisfaction'] <= 2:
        suggestions.append("Mentorship")
    if row['MonthlyIncome'] < df_encoded['MonthlyIncome'].mean():
        suggestions.append("Increase Salary")
    if row['YearsSinceLastPromotion'] > 3:
        suggestions.append("Promotion Plan")
    if row['OverTime'] == 1:
        suggestions.append("Reduce Workload")
    return "; ".join(suggestions) if suggestions else "Monitor"

critical_employees['Retention_Suggestion'] = critical_employees.apply(retention_suggestion, axis=1)

# Output
output_df = df_encoded.loc[
    critical_employees.index,
    ['EmployeeNumber', 'JobRole', 'Department', 'Attrition_Risk']
]

output_df['Retention_Suggestion'] = critical_employees['Retention_Suggestion']

output_df['JobRole'] = label_encoders['JobRole'].inverse_transform(output_df['JobRole'])
output_df['Department'] = label_encoders['Department'].inverse_transform(output_df['Department'])

print(output_df)

# ======================
# FEATURE IMPORTANCE
# ======================
rf_base.fit(X_train_scaled, y_train)
xgb_base.fit(X_train_scaled, y_train)

feature_names = X.columns

rf_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_base.feature_importances_
}).sort_values(by="Importance", ascending=False).head(15)

plt.figure()
plt.barh(rf_imp["Feature"], rf_imp["Importance"])
plt.gca().invert_yaxis()
plt.title("RF Feature Importance")
plt.grid()
plt.show()

xgb_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": xgb_base.feature_importances_
}).sort_values(by="Importance", ascending=False).head(15)

plt.figure()
plt.barh(xgb_imp["Feature"], xgb_imp["Importance"])
plt.gca().invert_yaxis()
plt.title("XGB Feature Importance")
plt.grid()
plt.show()

# ======================
# SHAP EXPLAINABILITY
# ======================
print("\n🔍 SHAP Analysis")

X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
explainer = shap.Explainer(xgb_base, X_test_df)
shap_values = explainer(X_test_df)

shap.summary_plot(shap_values, X_test_df)
shap.summary_plot(shap_values, X_test_df, plot_type="bar")

# ======================
# SAVE EVERYTHING
# ======================

joblib.dump(stacking_clf, 'stacking_clf.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print("✅ Model, scaler, encoders, features saved!")