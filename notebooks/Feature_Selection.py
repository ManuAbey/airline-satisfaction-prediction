# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# %%
# Load your dataset
df = pd.read_csv("data/preprocessed/Airline_preprocessed.csv")
print(f"Dataset shape: {df.shape}")
df.head()

# %% [markdown]
# Check for Missing Values

# %%
# Check missing values before feature selection
missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0])

# %% [markdown]
# Encode Categorical Variables for Analysis

# %%
# Create a copy for encoding
df_encoded = df.copy()

# Encode target variable
le_target = LabelEncoder()
df_encoded['satisfaction'] = le_target.fit_transform(df['satisfaction'])

# Encode categorical features
cat_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
for col in cat_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])

print("Encoding completed")
df_encoded.head()

# %% [markdown]
# Feature Correlation Analysis

# %%
# Calculate correlation with target variable
correlation = df_encoded.corr()['satisfaction'].abs().sort_values(ascending=False)
print("Feature Correlation with Satisfaction:")
print(correlation)

# Visualize top correlations
plt.figure(figsize=(10, 8))
top_corr = correlation[1:16]  # Exclude satisfaction itself
sns.barplot(x=top_corr.values, y=top_corr.index, palette='viridis')
plt.title('Top 15 Features Correlated with Satisfaction')
plt.xlabel('Absolute Correlation')
plt.tight_layout()
plt.show()

# %% [markdown]
#  Identify Multicollinearity (Redundant Features)

# %%
# Check correlation between features to identify redundancy
corr_matrix = df_encoded.corr().abs()

# Find highly correlated feature pairs (threshold: 0.8)
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_features = [
    column for column in upper_triangle.columns 
    if any(upper_triangle[column] > 0.8)
]

print("Highly correlated (redundant) features:")
print(high_corr_features)

# %% [markdown]
# Feature Engineering - Create New Features

# %%
# Based on member 1's work, create aggregate features
service_cols = [
    'Inflight wifi service', 'Departure/Arrival time convenient', 
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Inflight service', 'Cleanliness'
]

# Create aggregated service scores
df_encoded['TotalServiceScore'] = df_encoded[service_cols].sum(axis=1)
df_encoded['AvgServiceScore'] = df_encoded[service_cols].mean(axis=1)

# Create delay indicator
df_encoded['HasDelay'] = ((df_encoded['Departure Delay in Minutes'] > 0) | 
                          (df_encoded['Arrival Delay in Minutes'] > 0)).astype(int)

# Create age groups
df_encoded['AgeGroup'] = pd.cut(df_encoded['Age'], 
                                 bins=[0, 25, 40, 60, 100], 
                                 labels=[0, 1, 2, 3])

# Create distance categories
df_encoded['DistanceCategory'] = pd.cut(df_encoded['Flight Distance'], 
                                         bins=[0, 500, 1500, 5000], 
                                         labels=[0, 1, 2])

print("New features created successfully")
print(f"New dataset shape: {df_encoded.shape}")
df_encoded.head()

# %% [markdown]
# Feature Importance using Random Forest

# %%
# Prepare features and target
X = df_encoded.drop(['satisfaction'], axis=1)
y = df_encoded['satisfaction']

# Train Random Forest to get feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 20 Important Features:")
print(feature_importance.head(20))

# Visualize
plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(20), x='Importance', y='Feature', palette='rocket')
plt.title('Top 20 Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()

# %% [markdown]
# Select Top K Features (Chi-Square Test)

# %%
# Select top features using Chi-Square test
k_best = 20  # Select top 20 features

# Convert all categorical columns to numeric
X_numeric = X.copy()
for col in X_numeric.columns:
    if X_numeric[col].dtype.name == 'category':
        X_numeric[col] = X_numeric[col].astype(int)

# Ensure all values are non-negative for chi2
X_positive = X_numeric - X_numeric.min() + 1

# Apply Chi-Square feature selection
selector = SelectKBest(score_func=chi2, k=k_best)
X_selected = selector.fit_transform(X_positive, y)

# Get selected feature names
selected_features = X_numeric.columns[selector.get_support()].tolist()
print(f"\nTop {k_best} Selected Features (Chi-Square):")
print(selected_features)

# Get chi-square scores
chi2_scores = pd.DataFrame({
    'Feature': X_numeric.columns,
    'Chi2_Score': selector.scores_
}).sort_values('Chi2_Score', ascending=False)

print("\nChi-Square Scores:")
print(chi2_scores.head(20))

# Visualize
plt.figure(figsize=(10, 8))
sns.barplot(data=chi2_scores.head(20), x='Chi2_Score', y='Feature', palette='coolwarm')
plt.title('Top 20 Features by Chi-Square Score')
plt.tight_layout()
plt.show()

# %% [markdown]
# Mutual Information Feature Selection

# %%
# Calculate mutual information scores
X_numeric = X.copy()
for col in X_numeric.columns:
    if X_numeric[col].dtype.name == 'category':
        X_numeric[col] = X_numeric[col].astype(int)

mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
mi_df = pd.DataFrame({
    'Feature': X_numeric.columns,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print("Top 20 Features by Mutual Information:")
print(mi_df.head(20))

# Visualize
plt.figure(figsize=(10, 8))
sns.barplot(data=mi_df.head(20), x='MI_Score', y='Feature', palette='mako')
plt.title('Top 20 Features by Mutual Information')
plt.tight_layout()
plt.show()

# %% [markdown]
# Remove Low Importance Features

# %%
# Define threshold for feature importance
importance_threshold = 0.01

# Select features above threshold
important_features = feature_importance[
    feature_importance['Importance'] > importance_threshold
]['Feature'].tolist()

print(f"Features selected (importance > {importance_threshold}): {len(important_features)}")
print(f"Features removed: {len(X.columns) - len(important_features)}")

# Create final feature set
X_final = df_encoded[important_features]
print(f"\nFinal feature set shape: {X_final.shape}")

# Features dropped (importance <= threshold)
dropped_features = feature_importance[
    feature_importance['Importance'] <= importance_threshold
]['Feature'].tolist()

print("\nDropped Features:")
print(dropped_features)

# %% [markdown]
# Save the dataset after processing with selection features

# %%
# Save the final processed dataset for the next member
df_final = df_encoded[important_features + ['satisfaction']]
df_final.to_csv('airline_processed_data.csv', index=False)

print("✓ Processed dataset saved successfully!")
print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Features saved: {len(important_features)}")
print(f"File created: processed_data.csv")
print("\nThis file is ready for Member 3 (Data Preprocessing) and Member 4 (Model Development)")

# %% [markdown]
# Feature Selection Summary Report

# %%
# Create summary report
summary = {
    'Total Features (Original)': len(df.columns) - 1,  # Exclude target
    'Features After Engineering': len(X.columns),
    'Features Selected (Final)': len(important_features),
    'Features Removed': len(X.columns) - len(important_features),
    'Total Samples': len(df_final)
}

summary_df = pd.DataFrame(summary.items(), columns=['Metric', 'Value'])
print("\n=== FEATURE SELECTION SUMMARY ===")
print(summary_df.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 5))
plt.bar(summary_df['Metric'][:4], summary_df['Value'][:4], 
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Feature Selection Summary')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


