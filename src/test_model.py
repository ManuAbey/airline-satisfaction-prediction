"""
Test script to verify your trained model works correctly
Run this after saving your model from the training notebook
"""

import joblib
import numpy as np
from pathlib import Path

def test_model_loading():
    """Test if all model files load correctly"""
    print("="*70)
    print("TESTING MODEL ARTIFACTS")
    print("="*70)
    
    models_dir = Path('models')
    
    if not models_dir.exists():
        print("❌ ERROR: models/ directory not found!")
        print("Please run your training notebook and save the model first.")
        return False
    
    try:
        # Test loading model
        model = joblib.load(models_dir / 'best_model.pkl')
        print("✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        
        # Test loading scaler
        scaler = joblib.load(models_dir / 'scaler.pkl')
        print("✓ Scaler loaded successfully")
        
        # Test loading feature names
        feature_names = joblib.load(models_dir / 'feature_names.pkl')
        print(f"✓ Feature names loaded: {len(feature_names)} features")
        print(f"  Features: {feature_names[:5]}... (showing first 5)")
        
        # Test loading label mappings
        label_mappings = joblib.load(models_dir / 'label_mappings.pkl')
        print("✓ Label mappings loaded")
        print(f"  Mappings: {list(label_mappings.keys())}")
        
        # Test loading metadata
        metadata = joblib.load(models_dir / 'model_metadata.pkl')
        print("✓ Metadata loaded")
        print(f"  Model name: {metadata['model_name']}")
        print(f"  Accuracy: {metadata['accuracy']*100:.2f}%")
        print(f"  ROC-AUC: {metadata['roc_auc']:.4f}")
        print(f"  Features count: {metadata['n_features']}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: File not found - {e}")
        print("Make sure all model files are saved.")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_prediction():
    """Test making a prediction"""
    print("\n" + "="*70)
    print("TESTING MODEL PREDICTION")
    print("="*70)
    
    try:
        models_dir = Path('models')
        
        # Load artifacts
        model = joblib.load(models_dir / 'best_model.pkl')
        feature_names = joblib.load(models_dir / 'feature_names.pkl')
        metadata = joblib.load(models_dir / 'model_metadata.pkl')
        
        # Create a test input (all features set to middle values)
        # This should match the number of features in your trained model
        n_features = len(feature_names)
        test_input = np.array([[3] * n_features])  # All features = 3 (neutral)
        
        print(f"Creating test input with {n_features} features...")
        
        # Make prediction
        if metadata.get('requires_scaling', False):
            scaler = joblib.load(models_dir / 'scaler.pkl')
            test_input_scaled = scaler.transform(test_input)
            prediction = model.predict(test_input_scaled)[0]
            prediction_proba = model.predict_proba(test_input_scaled)[0]
        else:
            prediction = model.predict(test_input)[0]
            prediction_proba = model.predict_proba(test_input)[0]
        
        print(f"✓ Prediction successful!")
        print(f"  Predicted class: {prediction} ({'Satisfied' if prediction == 1 else 'Dissatisfied'})")
        print(f"  Confidence: {prediction_proba[prediction]*100:.1f}%")
        print(f"  Probabilities: [Dissatisfied: {prediction_proba[0]:.3f}, Satisfied: {prediction_proba[1]:.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during prediction: {e}")
        return False


def main():
    """Run all tests"""
    print("\nAIRLINE SATISFACTION MODEL TESTING")
    print("="*70)
    
    # Test 1: Loading
    loading_ok = test_model_loading()
    
    if not loading_ok:
        print("\n" + "="*70)
        print("TESTS FAILED")
        print("="*70)
        print("\nPlease fix the errors above before running the Streamlit app.")
        return
    
    # Test 2: Prediction
    prediction_ok = test_prediction()
    
    if not prediction_ok:
        print("\n" + "="*70)
        print("TESTS FAILED")
        print("="*70)
        return
    
    # All tests passed
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    print("\nYour model is ready to use.")
    print("You can now run: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()