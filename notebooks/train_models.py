# %%
#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# %%
#import data set
df = pd.read_csv('data/preprocessed/airline_data_feature_processed.csv')

# %%
# Display basic info
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nTarget variable distribution:")
print(df['satisfaction'].value_counts())

# %%
X = df.iloc[:,:22].values
y = df.iloc[:,22].values


# %%
# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# %%
# Standardize features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Number of features: {X_train.shape[1]}")

# %%
#Model 1
from sklearn.linear_model import LogisticRegression

print("\n" + "="*50)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*50)

# Train the model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred_lr))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba_lr))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))


# %%
#Model 2
from sklearn.ensemble import RandomForestClassifier

print("\n" + "="*50)
print("MODEL 2: RANDOM FOREST CLASSIFIER")
print("="*50)

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)  # Random Forest doesn't require scaling

# Make predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))



# # %%
# #Model 3
# from sklearn.svm import SVC

# print("\n" + "="*50)
# print("MODEL 3: SUPPORT VECTOR MACHINE (SVM)")
# print("="*50)

# # Train the model
# svm_model = SVC(kernel='rbf', probability=True, random_state=42)
# svm_model.fit(X_train_scaled, y_train)



# # %%
# #SVC
# # Make predictions
# y_pred_svm = svm_model.predict(X_test_scaled)
# y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

# # Evaluate the model
# print("\nAccuracy:", accuracy_score(y_test, y_pred_svm))
# print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba_svm))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_svm))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred_svm))


# %%
#Model 4
from sklearn.ensemble import GradientBoostingClassifier

print("\n" + "="*50)
print("MODEL 4: GRADIENT BOOSTING CLASSIFIER")
print("="*50)

# Train the model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Make predictions
y_pred_gb = gb_model.predict(X_test)
y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred_gb))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba_gb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb))

# %%
#Model 5
from sklearn.neighbors import KNeighborsClassifier

print("\n" + "="*50)
print("MODEL 5: K-NEAREST NEIGHBORS (KNN)")
print("="*50)

# Train the model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_knn = knn_model.predict(X_test_scaled)
y_pred_proba_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred_knn))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba_knn))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# %%
# Create comparison dataframe
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'KNN'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_rf),
        #accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_gb),
        accuracy_score(y_test, y_pred_knn)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_pred_proba_lr),
        roc_auc_score(y_test, y_pred_proba_rf),
        #roc_auc_score(y_test, y_pred_proba_svm),
        roc_auc_score(y_test, y_pred_proba_gb),
        roc_auc_score(y_test, y_pred_proba_knn)
    ]
})

results = results.sort_values('Accuracy', ascending=False)
print("\n", results.to_string(index=False))
print("\n" + "="*50)
print(f"Best Model: {results.iloc[0]['Model']} with Accuracy: {results.iloc[0]['Accuracy']:.4f}")
print("="*50)

# %%


import joblib
from pathlib import Path

# Create models directory
Path('models').mkdir(exist_ok=True)

# Get the feature names from your processed data
feature_names = df.drop(['satisfaction'], axis=1).columns.tolist()

# Save the best model (Random Forest based on your results)
joblib.dump(rf_model, 'models/best_model.pkl')
print("✓ Model saved")

# Save the scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Scaler saved")

# Save feature names (CRITICAL - must be in exact order)
joblib.dump(feature_names, 'models/feature_names.pkl')
print("✓ Feature names saved")

# Save label mappings (how you encoded categories)
label_mappings = {
    'Gender': {'Female': 0, 'Male': 1},
    'Customer Type': {'Loyal Customer': 0, 'disloyal Customer': 1},
    'Type of Travel': {'Business travel': 0, 'Personal Travel': 1},
    'Class': {'Business': 0, 'Eco': 1, 'Eco Plus': 2}
}
joblib.dump(label_mappings, 'models/label_mappings.pkl')
print("✓ Label mappings saved")

# Save model metadata
metadata = {
    'model_name': 'Random Forest',
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_rf),
    'n_features': len(feature_names),
    'feature_names': feature_names,
    'requires_scaling': False  # Random Forest doesn't need scaling
}
joblib.dump(metadata, 'models/model_metadata.pkl')
print("✓ Metadata saved")

print("\n" + "="*50)
print("ALL MODEL ARTIFACTS SAVED SUCCESSFULLY")
print("="*50)
print(f"Features saved: {len(feature_names)}")
print(f"Feature list: {feature_names}")
