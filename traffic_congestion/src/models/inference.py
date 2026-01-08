"""
Inference module for AGV zone congestion classification.
Provides predictions with probabilities and dominant signals.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import TRAINED_MODEL_PATH, SCALER_PATH


class ZoneCongestionPredictor:
    """Predictor class for AGV zone congestion states."""
    
    def __init__(self, model_path=TRAINED_MODEL_PATH, scaler_path=SCALER_PATH):
        """
        Initialize the predictor with trained model and scaler.
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the fitted scaler
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Get feature names (this assumes we know them or can extract from training)
        # For this implementation, we'll define the expected features
        self.feature_names = [
            'task_arrival_rate', 'zone_agv_count', 'avg_zone_speed', 
            'path_overlap_score', 'zone_density', 'avg_zone_wait_time',
            'task_to_agv_ratio', 'speed_drop_factor', 'spatial_conflict_index', 
            'agv_density_pressure', 'wait_time_pressure', 'congestion_pressure_score'
        ]
        
        # Define class names
        self.class_names = {0: 'SAFE', 1: 'WARNING', 2: 'CRITICAL'}
    
    def predict(self, zone_snapshot):
        """
        Predict zone state from a single snapshot.
        
        Args:
            zone_snapshot: dict or pd.Series with required features
            
        Returns:
            dict: {
                "state": "SAFE | WARNING | CRITICAL",
                "probabilities": {...},
                "dominant_signals": [...]
            }
        """
        # Convert to DataFrame if it's a dict
        if isinstance(zone_snapshot, dict):
            df = pd.DataFrame([zone_snapshot])
        elif isinstance(zone_snapshot, pd.Series):
            df = pd.DataFrame([zone_snapshot])
        else:
            df = zone_snapshot
            
        # Calculate derived features if not already present
        if 'task_to_agv_ratio' not in df.columns:
            df = self._calculate_derived_features(df)
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Extract features in the correct order
        X = df[self.feature_names].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probabilities
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Get dominant signals based on model coefficients
        dominant_signals = self._get_dominant_signals(X_scaled[0], prediction)
        
        # Format probabilities
        prob_dict = {self.class_names[i]: prob for i, prob in enumerate(probabilities)}
        
        result = {
            "state": self.class_names[prediction],
            "probabilities": prob_dict,
            "dominant_signals": dominant_signals
        }
        
        return result
    
    def _calculate_derived_features(self, df):
        """Calculate derived features from base features."""
        df_features = df.copy()
        
        # Derived signals as specified
        df_features['task_to_agv_ratio'] = df_features['task_arrival_rate'] / np.maximum(df_features['zone_agv_count'], 1)
        df_features['speed_drop_factor'] = 1 / np.maximum(df_features['avg_zone_speed'], 0.01)
        df_features['spatial_conflict_index'] = df_features['path_overlap_score'] * df_features['zone_agv_count']
        df_features['agv_density_pressure'] = df_features['zone_agv_count'] * df_features['zone_density']
        df_features['wait_time_pressure'] = df_features['avg_zone_wait_time'] * df_features['task_arrival_rate']
        
        # Congestion pressure score
        df_features['congestion_pressure_score'] = (
            0.30 * df_features['agv_density_pressure'] +
            0.25 * df_features['task_to_agv_ratio'] +
            0.20 * df_features['speed_drop_factor'] +
            0.15 * df_features['spatial_conflict_index'] +
            0.10 * df_features['wait_time_pressure']
        )
        
        return df_features
    
    def _get_dominant_signals(self, features, predicted_class):
        """Get dominant signals based on model coefficients."""
        # Get the coefficients for the predicted class
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_[predicted_class]  # Coefficients for predicted class
            feature_importance = np.abs(coef)  # Use absolute values for importance
            
            # Get indices of top 3 most important features
            top_indices = np.argsort(feature_importance)[-3:][::-1]
            
            # Map to feature names
            dominant_signals = [self.feature_names[i] for i in top_indices]
            
            return dominant_signals
        else:
            # Fallback if coefficients are not available
            return self.feature_names[:3]  # Return first 3 features as default


def main():
    """Example usage of the predictor."""
    # Example zone snapshot (this would typically come from your AGV system)
    example_snapshot = {
        'task_arrival_rate': 5.0,
        'zone_agv_count': 3,
        'avg_zone_speed': 1.0,
        'path_overlap_score': 0.3,
        'zone_density': 0.6,
        'avg_zone_wait_time': 15.0
    }
    
    # Initialize predictor
    predictor = ZoneCongestionPredictor()
    
    # Make prediction
    result = predictor.predict(example_snapshot)
    
    print("Prediction Result:")
    print(f"State: {result['state']}")
    print(f"Probabilities: {result['probabilities']}")
    print(f"Dominant Signals: {result['dominant_signals']}")


if __name__ == "__main__":
    main()