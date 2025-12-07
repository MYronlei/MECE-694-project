# Use this script to calculate prediction bias and suggest a repair threshold
import pandas as pd

def suggest_repair_threshold(file_90, file_60, lead_time=2):
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