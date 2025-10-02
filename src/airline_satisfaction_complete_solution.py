"""
Airline Passenger Satisfaction Prediction System
FDM Mini Project 2025
Team: Y3.S1.DS.01.01

Team Members:
- IT23185616 G.H.R.W. Madubashini (EDA)
- IT23409446 M.D.B.Abeygunawardana (Feature Selection)
- IT23398252 W.A.A.V. Perera (Data Preprocessing)
- IT23409514 S.D.B.Abeygunawardana (Model Development)

This script contains the complete ML pipeline for predicting airline passenger satisfaction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class AirlineSatisfactionPipeline:
    """Complete pipeline for airline satisfaction prediction"""
    
    def __init__(self, data_path):
        """Initialize the pipeline with data path"""
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    # ==================== MEMBER 3: DATA PREPROCESSING ====================
    
    def load_data(self):
        """Load the dataset"""
        print("="*70)
        print("STEP 1: LOADING DATA")
        print("="*70)
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        return self
    
    def preprocess_data(self):
        """Preprocess the dataset - Member 3 work"""
        print("\n" + "="*70)
        print("STEP 2: DATA PREPROCESSING")
        print("="*70)
        
        # Drop unnecessary columns
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop(['Unnamed: 0'], axis=1)
        if 'id' in self.df.columns:
            self.df = self.df.drop(['id'], axis=1)
        print("Unnecessary columns dropped")
        
        # Check missing values
        missing = self.df.isnull().sum()
        print(f"\nMissing values:\n{missing[missing > 0]}")
        
        # Handle missing values (Arrival Delay)
        if 'Arrival Delay in Minutes' in self.df.columns:
            arrival_delay_mean = self.df['Arrival Delay in Minutes'].mean()
            self.df['Arrival Delay in Minutes'].fillna(arrival_delay_mean, inplace=True)
            print(f"Filled missing Arrival Delay with mean: {arrival_delay_mean:.2f}")
        
        # Label encoding for categorical variables
        categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
        self.df_processed = self.df.copy()
        
        for col in categorical_cols:
            if col in self.df_processed.columns:
                le = LabelEncoder()
                self.df_processed[col] = le.fit_transform(self.df_processed[col])
        
        print(f"Encoded categorical features: {categorical_cols}")
        print(f"Processed dataset shape: {self.df_processed.shape}")
        
        return self
    
    # ==================== MEMBER 1: EXPLORATORY DATA ANALYSIS ====================
    
    def exploratory_analysis(self, save_plots=False):
        """Perform EDA - Member 1 work"""
        print("\n" + "="*70)
        print("STEP 3: EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        # Target distribution
        print("\nTarget Variable Distribution:")
        print(self.df_processed['satisfaction'].value_counts())
        print("\nPercentage:")
        print((self.df_processed['satisfaction'].value_counts(normalize=True) * 100).round(2))
        
        # Service columns
        service_cols = [
            'Inflight wifi service', 'Departure/Arrival time convenient',
            'Ease of Online booking', 'Gate location', 'Food and drink',
            'Online boarding', 'Seat comfort', 'Inflight entertainment',
            'On-board service', 'Leg room service', 'Baggage handling',
            'Checkin service', 'Inflight service', 'Cleanliness'
        ]
        
        # Feature engineering - aggregate scores
        self.df_processed['TotalServiceScore'] = self.df_processed[service_cols].sum(axis=1)
        self.df_processed['AvgServiceScore'] = self.df_processed[service_cols].mean(axis=1)
        
        frontline = ['Seat comfort', 'Inflight entertainment', 'On-board service',
                     'Inflight service', 'Cleanliness']
        logistics = ['Checkin service', 'Baggage handling', 'Gate location',
                     'Ease of Online booking']
        
        self.df_processed['FrontlineScore'] = self.df_processed[frontline].sum(axis=1)
        self.df_processed['LogisticsScore'] = self.df_processed[logistics].sum(axis=1)
        
        print("\nEngineered Features Created:")
        print("- TotalServiceScore, AvgServiceScore")
        print("- FrontlineScore, LogisticsScore")
        
        # Correlation analysis
        corr_with_target = self.df_processed.corr()['satisfaction'].abs().sort_values(ascending=False)
        print("\nTop 10 Features Correlated with Satisfaction:")
        print(corr_with_target.head(11))  # 11 to exclude satisfaction itself
        
        return self
    
    # ==================== MEMBER 2: FEATURE SELECTION ====================
    
    def feature_selection(self, importance_threshold=0.01):
        """Perform feature selection - Member 2 work"""
        print("\n" + "="*70)
        print("STEP 4: FEATURE SELECTION")
        print("="*70)
        
        # Prepare features and target
        X = self.df_processed.drop(['satisfaction'], axis=1)
        y = self.df_processed['satisfaction']
        
        # Random Forest for feature importance
        print("\nTraining Random Forest for feature importance...")
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_temp.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_temp.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 15 Important Features:")
        print(feature_importance.head(15))
        
        # Select features above threshold
        important_features = feature_importance[
            feature_importance['Importance'] > importance_threshold
        ]['Feature'].tolist()
        
        print(f"\nFeatures selected (importance > {importance_threshold}): {len(important_features)}")
        print(f"Features removed: {len(X.columns) - len(important_features)}")
        
        # Update processed dataframe with selected features
        self.df_processed = self.df_processed[important_features + ['satisfaction']]
        
        print(f"Final feature set shape: {self.df_processed.shape}")
        
        return self
    
    # ==================== DATA SPLITTING ====================
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("\n" + "="*70)
        print("STEP 5: DATA SPLITTING")
        print("="*70)
        
        X = self.df_processed.drop(['satisfaction'], axis=1).values
        y = self.df_processed['satisfaction'].values
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape[0]} samples")
        print(f"Testing set size: {self.X_test.shape[0]} samples")
        print(f"Number of features: {self.X_train.shape[1]}")
        
        return self
    
    def scale_features(self):
        """Scale features for models that need it"""
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("Features scaled using StandardScaler")
        
        return self
    
    # ==================== MEMBER 4: MODEL DEVELOPMENT ====================
    
    def train_models(self):
        """Train multiple classification models - Member 4 work"""
        print("\n" + "="*70)
        print("STEP 6: MODEL TRAINING")
        print("="*70)
        
        # Model 1: Logistic Regression
        print("\n" + "-"*50)
        print("MODEL 1: LOGISTIC REGRESSION")
        print("-"*50)
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(self.X_train_scaled, self.y_train)
        self.models['Logistic Regression'] = lr_model
        self._evaluate_model('Logistic Regression', lr_model, self.X_test_scaled, self.y_test)
        
        # Model 2: Random Forest
        print("\n" + "-"*50)
        print("MODEL 2: RANDOM FOREST CLASSIFIER")
        print("-"*50)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        self._evaluate_model('Random Forest', rf_model, self.X_test, self.y_test)
        
        # Model 3: SVM
        print("\n" + "-"*50)
        print("MODEL 3: SUPPORT VECTOR MACHINE")
        print("-"*50)
        svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        svm_model.fit(self.X_train_scaled, self.y_train)
        self.models['SVM'] = svm_model
        self._evaluate_model('SVM', svm_model, self.X_test_scaled, self.y_test)
        
        # Model 4: Gradient Boosting
        print("\n" + "-"*50)
        print("MODEL 4: GRADIENT BOOSTING CLASSIFIER")
        print("-"*50)
        gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        gb_model.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb_model
        self._evaluate_model('Gradient Boosting', gb_model, self.X_test, self.y_test)
        
        # Model 5: KNN
        print("\n" + "-"*50)
        print("MODEL 5: K-NEAREST NEIGHBORS")
        print("-"*50)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(self.X_train_scaled, self.y_train)
        self.models['KNN'] = knn_model
        self._evaluate_model('KNN', knn_model, self.X_test_scaled, self.y_test)
        
        return self
    
    def _evaluate_model(self, model_name, model, X_test, y_test):
        """Evaluate a single model"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        self.results[model_name] = {
            'Accuracy': accuracy,
            'ROC-AUC': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*70)
        print("STEP 7: MODEL COMPARISON")
        print("="*70)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[m]['Accuracy'] for m in self.results.keys()],
            'ROC-AUC': [self.results[m]['ROC-AUC'] for m in self.results.keys()]
        }).sort_values('Accuracy', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        best_model = comparison_df.iloc[0]['Model']
        best_accuracy = comparison_df.iloc[0]['Accuracy']
        
        print("\n" + "="*70)
        print(f"BEST MODEL: {best_model}")
        print(f"ACCURACY: {best_accuracy:.4f}")
        print("="*70)
        
        return comparison_df
    
    def save_results(self, output_path='airline_results.csv'):
        """Save processed data and results"""
        print(f"\nSaving processed dataset to {output_path}")
        self.df_processed.to_csv(output_path, index=False)
        print("Results saved successfully!")
    
    def run_complete_pipeline(self):
        """Execute the complete ML pipeline"""
        print("\n" + "="*70)
        print("AIRLINE PASSENGER SATISFACTION PREDICTION SYSTEM")
        print("FDM Mini Project 2025 - Team Y3.S1.DS.01.01")
        print("="*70)
        
        self.load_data()
        self.preprocess_data()
        self.exploratory_analysis()
        self.feature_selection()
        self.split_data()
        self.scale_features()
        self.train_models()
        comparison_df = self.compare_models()
        self.save_results()
        
        print("\n" + "="*70)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return comparison_df


# ==================== MAIN EXECUTION ====================

def main():
    """Main function to run the complete pipeline"""
    
    # Initialize pipeline with your data path
    data_path = 'data/raw/Airline.csv'  # Update this path
    
    pipeline = AirlineSatisfactionPipeline(data_path)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    print("\nPipeline execution completed!")
    print("Check 'airline_results.csv' for processed data")
    
    return pipeline, results


if __name__ == "__main__":
    pipeline, results = main()