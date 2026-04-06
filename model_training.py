import pandas as pd
import numpy as np
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pre_processing import (
    load_and_clean_data, 
    create_time_features, 
    create_lag_features, 
    encode_categorical_features, 
    split_and_scale_data_3way
)

def evaluate_model(name, model, X_val, y_val, fit_time):
    """Generates evaluation metrics for a trained model."""
    start_time = time.time()
    predictions = model.predict(X_val)
    predict_time = time.time() - start_time
    
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    
    return {
        "Model": name,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2 Score": round(r2, 4),
        "Train Time (s)": round(fit_time, 2),
        "Predict Time (s)": round(predict_time, 4)
    }

if __name__ == "__main__":
    print("=" * 60)
    print("1. RUNNING PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # 1. Get the data
    raw_df = load_and_clean_data(r"dataset\city_day.csv") 
    df_time = create_time_features(raw_df)
    df_lagged = create_lag_features(df_time, lags=1)
    df_encoded = encode_categorical_features(df_lagged)
    
    # 2. Split and Scale
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale_data_3way(df_encoded)
    
    print(f"Train size: {X_train.shape[0]} rows | Validation size: {X_val.shape[0]} rows")
    
    print("\n" + "=" * 60)
    print("2. INITIALIZING MODELS")
    print("=" * 60)
    
    # Initialize the 3 models with sensible baseline parameters
    models = {
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    }
    
    results = []
    print("Training models... (This may take a minute)\n")
    
    for name, model in models.items():
        print(f" -> Training {name}...")
        start_train = time.time()
        
        # Fit the model
        model.fit(X_train, y_train)
        fit_time = time.time() - start_train
        
        # Evaluate on the VALIDATION set
        metrics = evaluate_model(name, model, X_val, y_val, fit_time)
        results.append(metrics)

    print("\n" + "=" * 60)
    print("3. VALIDATION SET PERFORMANCE COMPARISON")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    # Sort by R2 Score (highest is best)
    results_df = results_df.sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
    
    print(results_df.to_string(index=False))
    print("=" * 60)