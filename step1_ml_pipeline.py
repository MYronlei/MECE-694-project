#!/usr/bin/env python3

import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================================
# 1. Data processing
# ============================================================================

def load_cmapss_data(dataname = 'FD001') -> Tuple[pd.DataFrame, pd.DataFrame]:
    columns = ['engine_id', 'cycle', 'operation1', 'operation2', 'operation3'] + [f'sensor_{i}' for i in range(1, 22)]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'cmapss', f'train_{dataname}.txt.gz')
    test_path = os.path.join(base_dir, 'cmapss', f'test_{dataname}.txt.gz')

    train = pd.read_csv(train_path, sep='\s+', header=None, names=columns, compression='infer')
    test = pd.read_csv(test_path, sep='\s+', header=None, names=columns, compression='infer')

    # check and impute missing values
    def check_and_impute(dataset: pd.DataFrame, name: str) -> pd.DataFrame:
        missing = dataset.isna().sum()
        total_missing = missing.sum()
        if total_missing > 0:
            print(f"[{name}] Missing values detected: total={total_missing}")
            print(missing[missing > 0])
            # perform per-engine forward/backward fill (preserve time order)
            dataset = dataset.sort_values(['engine_id', 'cycle']).groupby('engine_id').apply(lambda g: g.ffill().bfill())
            dataset.index = dataset.index.droplevel(0) if isinstance(dataset.index, pd.MultiIndex) else dataset.index
            # if still missing, fill with column mean
            if dataset.isna().sum().sum() > 0:
                mean = dataset.mean()
                dataset = dataset.fillna(mean)
            print(f"[{name}] Missing values after imputation: {dataset.isna().sum().sum()}")
        return dataset

    train = check_and_impute(train, 'train')
    test = check_and_impute(test, 'test')

    # drop known constant sensors if present
    constant_features = [c for c in ['operation1', 'operation2', 'operation3'] if c in train.columns]
    if constant_features:
        train.drop(constant_features, axis=1, inplace=True)
        test.drop(constant_features, axis=1, inplace=True)


    # for train set, RUL = max_cycle_per_engine - current_cycle
    train['RUL'] = train.groupby('engine_id')['cycle'].transform('max') - train['cycle']

    # For test set, the true failure cycles are usually provided in a separate file RUL_FDXXX.txt
    rul_path_gz = os.path.join(base_dir, 'cmapss', f'RUL_{dataname}.txt.gz')
    labels = pd.read_csv(rul_path_gz, header=None).squeeze().astype(int).to_numpy()
    # labels = remaining cycles AFTER the last observed test cycle for each engine
    # map labels to engine ids (preserve test engine order to match RUL file)
    engine_ids = test['engine_id'].unique()
    label_map = {int(eid): int(lbl) for eid, lbl in zip(engine_ids, labels)}
    label_series = test['engine_id'].map(label_map)
    # last observed cycle per engine in test
    last_cycle = test.groupby('engine_id')['cycle'].transform('max')
    # per-row RUL = provided remaining_after_last + (last_cycle - current_cycle)
    test['RUL'] = label_series + (last_cycle - test['cycle'])

    return train, test

def data_processing(train, test) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # scale only feature columns
    exclude = ['engine_id', 'cycle', 'RUL']
    feature_cols = [c for c in train.columns if c not in exclude]

    # do robust scaling, based on train set
    scaler = RobustScaler().fit(train[feature_cols])
    train_scaled = train.copy()
    test_scaled = test.copy()

    train_scaled[feature_cols] = scaler.transform(train[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test[feature_cols])

    return train_scaled, test_scaled

def main():
    # Load data
    train, test = load_cmapss_data('FD001')
    train_std, test_std = data_processing(train, test)
    print("Training data shape:", train_std.shape)
    print("Testing data shape:", test_std.shape)
    print(test_std)
    # Prepare features and target (RUL)
    exclude = ['engine_id', 'cycle', 'RUL']
    feature_cols = [c for c in train_std.columns if c not in exclude]

    X_train = train_std[feature_cols].values
    y_train = train_std['RUL'].values
    X_test = test_std[feature_cols].values
    y_test = test_std['RUL'].values

    # Train a sample MLPRegressor on RUL
    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 32, 64),
        activation='relu',
        solver='adam',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )

    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(((y_test - y_test_pred) ** 2).mean())

    print(f"Baseline Model MAE: {mae:.3f}")
    print(f"Baseline Model RMSE: {rmse:.3f}")
    print(f"Baseline Model R2 score: {r2:.4f}")

if __name__ == "__main__":
    main()