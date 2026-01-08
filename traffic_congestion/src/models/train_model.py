"""
Model training module for AGV zone congestion classification.
Uses StandardScaler + Multinomial Logistic Regression with specified parameters.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, recall_score
import joblib
from pathlib import Path
import sys

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATASET_PATH, TRAINED_MODEL_PATH, SCALER_PATH


def train_model():
    """Train the multinomial logistic regression model."""
    print("Loading processed dataset...")
    df = pd.read_csv(PROCESSED_DATASET_PATH)
    
    # Define feature columns (exclude non-numeric columns like timestamp, zone_id)
    non_feature_columns = ['timestamp', 'zone_id', 'label']
    feature_columns = [col for col in df.columns if col not in non_feature_columns]
    
    # Prepare features (X) and target (y)
    X = df[feature_columns].values
    y = df['label'].values
    
    print(f"Training data shape: {X.shape}")
    print(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Initialize scaler and model
    scaler = StandardScaler()
    model = LogisticRegression(
        solver='saga',
        multi_class='multinomial',
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    
    # Perform stratified 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_f1_scores = []
    cv_recall_scores = []
    
    print("Performing 5-fold cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Scale features
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)
        
        # Train model
        model_fold = LogisticRegression(
            solver='saga',
            multi_class='multinomial',
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        model_fold.fit(X_train_fold_scaled, y_train_fold)
        
        # Predictions
        y_pred = model_fold.predict(X_val_fold_scaled)
        
        # Calculate metrics
        macro_f1 = f1_score(y_val_fold, y_pred, average='macro')
        recall = recall_score(y_val_fold, y_pred, average='macro')
        
        cv_f1_scores.append(macro_f1)
        cv_recall_scores.append(recall)
        
        print(f"Fold {fold + 1}: Macro-F1 = {macro_f1:.4f}, Macro-Recall = {recall:.4f}")
    
    print(f"Average CV Macro-F1: {np.mean(cv_f1_scores):.4f} (+/- {np.std(cv_f1_scores) * 2:.4f})")
    print(f"Average CV Macro-Recall: {np.mean(cv_recall_scores):.4f} (+/- {np.std(cv_recall_scores) * 2:.4f})")
    
    # Train final model on full dataset
    print("Training final model on full dataset...")
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    
    # Final predictions for detailed metrics
    y_pred_final = model.predict(X_scaled)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y, y_pred_final, target_names=['SAFE', 'WARNING', 'CRITICAL']))
    
    # Per-class recall
    per_class_recall = recall_score(y, y_pred_final, average=None)
    print(f"\nPer-class Recall:")
    for i, recall in enumerate(per_class_recall):
        class_name = ['SAFE', 'WARNING', 'CRITICAL'][i]
        print(f"  {class_name}: {recall:.4f}")
    
    # Save the trained model and scaler
    TRAINED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, TRAINED_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"\nModel training complete.")
    print(f"Trained model saved to: {TRAINED_MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    
    return model, scaler


def main():
    """Main function to run the training pipeline."""
    model, scaler = train_model()


if __name__ == "__main__":
    main()