# Use this script to calculate prediction bias and suggest a repair threshold
import pandas as pd
import os

def suggest_repair_threshold(file_90, file_60, lead_time=2):
    """
    Calculate optimal repair threshold from prediction bias analysis.
    
    Parameters:
    -----------
    file_90 : str
        Path to RUL predictions with 90-cycle window
    file_60 : str
        Path to RUL predictions with 60-cycle window
    lead_time : int
        Lead time in shifts for maintenance planning (default: 2)
    
    Returns:
    --------
    int
        Suggested repair threshold in RUL cycles
    """
    df90 = pd.read_csv(file_90)
    bias_90 = (df90['pred_RUL'] - df90['true_RUL']).dropna()
    mean_bias_90 = bias_90.mean()
    std_bias_90 = bias_90.std()
    suggested_90 = int((mean_bias_90 + lead_time + std_bias_90).round())

    df60 = pd.read_csv(file_60)
    bias_60 = (df60['pred_RUL'] - df60['true_RUL']).dropna()
    mean_bias_60 = bias_60.mean()
    std_bias_60 = bias_60.std()
    suggested_60 = int((mean_bias_60 + lead_time + std_bias_60).round())

    suggested_final = max(suggested_60, suggested_90)
    return suggested_final


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_90 = os.path.join(base_dir, 'output', 'rul_predictions_per_engine_cycle90.csv')
    file_60 = os.path.join(base_dir, 'output', 'rul_predictions_per_engine_cycle60.csv')
    
    threshold = suggest_repair_threshold(file_90, file_60, lead_time=4)
    print(threshold)