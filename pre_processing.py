import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath="dataset/city_day.csv"):
    """Loads the dataset, fixes column names, and removes target leakage columns."""
    # 1. Parse dates using the actual column name in the CSV ('Datetime')
    df = pd.read_csv(filepath, parse_dates=["Datetime"])
    
    df.rename(columns={"Datetime": "Date"}, inplace=True)
    
    # 2. DROP TARGET LEAKAGE: 
    # 'AQI_Bucket' is derived directly from 'AQI'. If left in the dataset, causes leakage. We must drop it for a regression task.
    if "AQI_Bucket" in df.columns:
        df = df.drop(columns=["AQI_Bucket"])
        
    return df

def create_time_features(df):
    """Extracts seasonal and cyclical features from the Date column."""
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Cyclical encoding for Month so the model knows Dec (12) is next to Jan (1)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    return df

def create_lag_features(df, target_col="AQI", lags=1):
    df = df.copy()
    df = df.sort_values(by=['City', 'Date'])
    
    for i in range(1, lags + 1):
        # Shift the target variable by 'i' days within each city group
        df[f'{target_col}_lag_{i}'] = df.groupby('City')[target_col].shift(i)
        
        # lag important pollutants
        if 'PM2.5' in df.columns:
            df[f'PM2.5_lag_{i}'] = df.groupby('City')['PM2.5'].shift(i)
        if 'PM10' in df.columns:
            df[f'PM10_lag_{i}'] = df.groupby('City')['PM10'].shift(i)
        
    # Shifting creates NaN values for the first 'lag' days of each city. We must drop them.
    df = df.dropna()
    return df

def encode_categorical_features(df):
    """One-Hot Encodes the 'City' column."""
    # pandas get_dummies for simple one-hot encoding
    df = pd.get_dummies(df, columns=['City'], drop_first=True, dtype=int)
    return df

def split_and_scale_data_3way(df, target_col="AQI"):
    # 1. Sort by Date to ensure chronological integrity
    df = df.sort_values(by='Date')
    
    # 2. Separate features (X) and target (y)
    # We drop 'Date' because machine learning models cannot process datetime objects directly
    X = df.drop(columns=[target_col, 'Date'])
    y = df[target_col]
    
    # 3. First Split: Isolate the Training set (70%) from the rest (30%)
    # shuffle=False is critical to maintain the timeline!
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, shuffle=False)
    
    # 4. Second Split: Divide the remaining 30% evenly into Validation (15%) and Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, shuffle=False)
    
    # 5. Feature Scaling
    scaler = StandardScaler()
    
    # Fit the scaler ONLY on the Training data to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform Validation and Test data using the Train scaler
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to retain column names and index structure
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

if __name__ == "__main__":
    print("Starting Data Preprocessing...")
    
    # 1. Load Data 
    raw_df = load_and_clean_data(r"dataset\city_day.csv") 
    
    # 2. Feature Engineering (Time components)
    df_time = create_time_features(raw_df)
    
    # 3. Add Lag Features (Using 1-day lag to capture 'yesterday's' data)
    df_lagged = create_lag_features(df_time, lags=1)
    
    # 4. Categorical Encoding (Convert 'City' to numbers)
    df_encoded = encode_categorical_features(df_lagged)
    
    # 5. Split and Scale (Train / Val / Test)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale_data_3way(df_encoded)
    
    print(f"Preprocessing Complete.")
    print("=" * 40)
    print(f"       Train shape      : {X_train.shape} (70%)")
    print(f"       Validation shape : {X_val.shape} (15%)")
    print(f"       Test shape       : {X_test.shape} (15%)")
    print("=" * 40)
    print(f"       Target variable  : AQI")
    print(f"       Total features   : {len(X_train.columns)}")