import pandas as pd
import numpy as np

def preprocess_inference_data(raw_df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """
    Preprocesses raw transaction data to prepare it for inference.
    
    Parameters:
    - raw_df: pd.DataFrame containing columns 'Time', 'Amount', and 'V1' through 'V28'.
    - scalers: Dict containing scikit-learn StandardScaler instances:
               - 'amount_scaler': scaler for the 'Amount' feature.
               - 'time_scaler': scaler for the 'Time' feature.
               
    Returns:
    - pd.DataFrame with scaled columns and correct feature ordering for the model.
    """
    data = raw_df.copy()
    
    # Ensure all required PCA columns exist, fill missing ones with 0.0
    for i in range(1, 29):
        col = f'V{i}'
        if col not in data.columns:
            data[col] = 0.0
            
    # Handle scaling for 'Amount'
    if 'Amount' in data.columns:
        # Use values.reshape(-1, 1) to avoid potential user warning from scikit-learn
        data['scaled_amount'] = scalers['amount_scaler'].transform(data[['Amount']].values)
    else:
        data['scaled_amount'] = 0.0
        
    # Handle scaling for 'Time'
    if 'Time' in data.columns:
        data['scaled_time'] = scalers['time_scaler'].transform(data[['Time']].values)
    else:
        data['scaled_time'] = 0.0
        
    # Drop original columns if they exist
    cols_to_drop = [col for col in ['Amount', 'Time'] if col in data.columns]
    if cols_to_drop:
        data.drop(cols_to_drop, axis=1, inplace=True)
        
    # Reorder columns to match the training feature set: V1...V28, scaled_amount, scaled_time
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
    return data[feature_cols]
