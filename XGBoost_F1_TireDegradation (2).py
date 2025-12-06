# Databricks notebook source
# MAGIC %pip install xgboost
# MAGIC %pip install mlflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# Databricks - MLflow is auto-configured, no credentials needed
# (Colab version used userdata for DATABRICKS_HOST/TOKEN)

# End runs
while mlflow.active_run():
    mlflow.end_run(status='FINISHED')
mlflow.end_run()

mlflow.set_experiment("/Users/jrcarrascofrasqu@gmail.com/F1_Tire_Deg_Demo_v1")

# COMMAND ----------

# Unity Catalog Volume path (replaces Google Drive path)
CATALOG = "workspace"
SCHEMA = "default"
VOLUME = "f1"
DATA_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/simulated_dataset.csv"

df = pd.read_csv(DATA_PATH, engine='pyarrow')
# df = df.iloc[:100000]  # Limit if needed

# New filter: Focus only on F1 and specified tracks
df = df[(df['Motorsport_Type'] == 'F1') & df['Track'].isin(['Monza', 'Monaco', 'Red Bull Ring'])]
print(f"Rows after F1 and track filter: {len(df)}")
print("Tracks:\n", df['Track'].value_counts(normalize=True))
print("Motorsport Types:\n", df['Motorsport_Type'].value_counts(normalize=True))  # Verify F1 only

# COMMAND ----------

df = df[df['Track'] != 'Nürburgring Nordschleife']
print(f"Rows after filter: {len(df)}")
print("Tracks:\n", df['Track'].value_counts(normalize=True))

# COMMAND ----------

numerical_features = ['Throttle', 'Brake', 'Speed', 'Surface_Roughness',
                      'Ambient_Temperature', 'Lateral_G_Force', 'Longitudinal_G_Force',
                      'Tire_Friction_Coefficient', 'Tire_Tread_Depth',
                      'force_on_tire', 'front_surface_temp', 'rear_surface_temp',
                      'front_inner_temp', 'rear_inner_temp']
categorical_features = ['Tire_Compound', 'Driving_Style', 'Track']
target_reg = 'cumilative_Tire_Wear'  # Fix typo
target_class = 'degradation_risk'

# COMMAND ----------

# Optional: Track mult for deg (uncomment to apply; adjusts wear, model learns)
track_mult = {'Monza': 1.2, 'Monaco': 0.9, 'Red Bull Ring': 1.1}
df[target_reg] *= df['Track'].map(track_mult).fillna(1.0)
print(df[target_reg].describe())

# COMMAND ----------

# Derive class (qcut for balance; or fixed bins [0,0.3,0.6,max] for realism)
if target_class not in df.columns:
    df[target_class] = pd.qcut(df[target_reg], q=3, labels=['safe', 'medium', 'critical'], duplicates='drop')
    print("Risk Dist:\n", df[target_class].value_counts(normalize=True))

df = df.dropna(subset=[target_class])
print(f"Rows clean: {len(df)}")

# COMMAND ----------

X_num = df[numerical_features]
X_cat = df[categorical_features]
y_class = df[target_class]

encoder = OneHotEncoder(sparse=False, drop='first')
X_cat_encoded = encoder.fit_transform(X_cat)
X = np.hstack((X_num, X_cat_encoded))

le = LabelEncoder()
y_class_encoded = le.fit_transform(y_class)

# COMMAND ----------

X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sample_weights = compute_sample_weight(class_weight='balanced', y=y_class_train)

# COMMAND ----------

def generate_confusion_matrix(y_true, y_pred, class_labels, run_id=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if run_id:
        mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)
    return cm

# COMMAND ----------

params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'seed': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 10,
    'tree_method': 'hist'  # 'gpu_hist' on GPU
}

# COMMAND ----------

with mlflow.start_run(run_name="XGBoost_Classifier_POC") as xgb_run:
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(
        X_train_scaled,
        y_class_train,
        sample_weight=sample_weights,
        eval_set=[(X_test_scaled, y_class_test)],
        verbose=False
    )
    import joblib  # Better than pickle for large models like XGBoost

    joblib.dump(xgb_model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'le.pkl')
    joblib.dump(encoder, 'ohe.pkl')
    print("PKL files generated: model.pkl, scaler.pkl, le.pkl, ohe.pkl")

    y_class_pred = xgb_model.predict(X_test_scaled)
    y_class_probs = xgb_model.predict_proba(X_test_scaled)
    class_report = classification_report(
        y_class_test,
        y_class_pred,
        output_dict=True,
        zero_division=0
    )
    auc_roc = roc_auc_score(
        y_class_test,
        y_class_probs,
        multi_class='ovr',
        average='weighted'
    )
    metrics = {
        'accuracy': class_report['accuracy'],
        'precision_safe': class_report.get('0', {'precision': 0.0})['precision'],
        'recall_safe': class_report.get('0', {'recall': 0.0})['recall'],
        'precision_medium': class_report.get('1', {'precision': 0.0})['precision'],
        'recall_medium': class_report.get('1', {'recall': 0.0})['recall'],
        'precision_critical': class_report.get('2', {'precision': 0.0})['precision'],
        'recall_critical': class_report.get('2', {'recall': 0.0})['recall'],
        'f1_critical': class_report.get('2', {'f1-score': 0.0})['f1-score'],
        'auc_roc_weighted': auc_roc
    }
    mlflow.log_metrics(metrics)
    mlflow.log_params(params)

    signature = infer_signature(X_train_scaled, y_class_pred)
    mlflow.xgboost.log_model(xgb_model, "xgb_model", signature=signature)

    unique_classes = np.unique(np.concatenate((y_class_test, y_class_pred)))
    class_labels = le.inverse_transform(unique_classes)
    generate_confusion_matrix(
        y_class_test,
        y_class_pred,
        class_labels,
        run_id=xgb_run.info.run_id
    )
    print("Report:\n", classification_report(y_class_test, y_class_pred, zero_division=0))
    print(f"AUC-ROC: {auc_roc:.4f}")

# COMMAND ----------

with mlflow.start_run(run_id=xgb_run.info.run_id):
    feature_names = numerical_features + list(encoder.get_feature_names_out(categorical_features))
    importances = xgb_model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    importance_dict = {f'importance_{feature_names[i]}': importances[i] for i in sorted_idx[:5]}
    mlflow.log_metrics(importance_dict)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in sorted_idx[:10]], importances[sorted_idx[:10]])
    plt.xlabel('Importance')
    plt.title('Top Importances')
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)
