import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import os
import time
from typing import List, Tuple, Dict

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"R2": r2, "MAE": mae, "MAPE": mape, "RMSE": rmse}

def plot_comparison(df_plot, title, save_path, time_col='date', actual_col='travel_time', pred_col='prediction'):
    plt.figure(figsize=(15, 6))
    plt.plot(df_plot[time_col], df_plot[actual_col], label='Actual', marker='o', markersize=4)
    plt.plot(df_plot[time_col], df_plot[pred_col], label='Predicted', marker='x', markersize=4)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Travel Time (min)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 1. Load Data
    data_path = 'data/processed/data_travel_time_xgboost.csv'
    output_dir = 'results/xgboost_v2'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    # Day mapping
    day_map = {'월': 0, '화': 1, '수': 2, '목': 3, '금': 4, '토': 5, '일': 6}
    if 'day_numeric' not in df.columns:
        # If day_numeric was not in the csv, map it. The preprocess_xgboost didn't add it.
        # But data_travel_time.csv has 'day' column.
        df['day_numeric'] = df['day'].map(day_map)
    
    # Sort for safety
    df = df.sort_values(['gate_start', 'gate_end', 'date'])
    
    # Define features
    features = [
        'hour', 'gate_start', 'gate_end', 'holiday', 'link_length', 'day_numeric',
        'travel_time(t-1)', 'travel_time(t-2)', 'travel_time(t-3)', 
        'travel_time(t-24)', 'travel_time(t-168)'
    ]
    
    # Drop rows with NaN in features
    df_clean = df.dropna(subset=features + ['travel_time']).copy()
    
    # Split Data: 2025 for Training, 2026 Jan-Feb for Testing
    df_train = df_clean[df_clean['date'].dt.year == 2025].copy()
    df_test = df_clean[(df_clean['date'].dt.year == 2026) & (df_clean['date'].dt.month <= 20)].copy() # Jan-Feb
    
    X_train = df_train[features]
    y_train = df_train['travel_time']
    X_test = df_test[features]
    y_test = df_test['travel_time']
    
    # 2. Train Model
    print("Training XGBoost model...")
    start_train = time.time()
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    end_train = time.time()
    train_time = end_train - start_train
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # 3. Next-Step Prediction
    print("Performing 1-step predictions...")
    start_pred_1 = time.time()
    df_test['pred_1step'] = model.predict(X_test)
    end_pred_1 = time.time()
    pred_time_1 = end_pred_1 - start_pred_1
    
    # 4. 24-Step Recursive Prediction
    print("Performing 24-step recursive predictions...")
    # We will predict for each day starting at 00:00 for each route
    start_pred_24 = time.time()
    
    results_24 = []
    routes = df_test.groupby(['gate_start', 'gate_end']).groups.keys()
    
    # Group test data by date and route to easily pick 24-hour blocks
    df_test['date_only'] = df_test['date'].dt.date
    
    for (gs, ge) in routes:
        df_route = df_test[(df_test['gate_start'] == gs) & (df_test['gate_end'] == ge)].copy()
        test_days = df_route['date_only'].unique()
        
        for day in test_days:
            day_data = df_route[df_route['date_only'] == day].sort_values('date')
            if len(day_data) < 24: continue # Skip partial days
            
            # Start recursion
            current_day_preds = []
            # Initial state: we need the features for the first hour (00:00)
            # All lags (t-1, t-2, t-3, t-24, t-168) are already in df_test for the first hour.
            
            # To simulate true recursive prediction, we update lags as we go.
            # However, t-24 and t-168 for a 24-hour horizon will always refer to ground truth 
            # (since we haven't predicted those yet in this window).
            
            row = day_data.iloc[0].copy()
            for h in range(24):
                feat_vec = row[features].values.reshape(1, -1)
                pred = model.predict(feat_vec)[0]
                current_day_preds.append({
                    'date': row['date'],
                    'gate_start': gs,
                    'gate_end': ge,
                    'actual': row['travel_time'],
                    'prediction': pred
                })
                
                # Prepare row for next hour
                if h < 23:
                    next_row_actual = day_data.iloc[h+1].copy()
                    row = next_row_actual.copy()
                    # Update recursive lags
                    # t-1 for next hour is current pred
                    row['travel_time(t-1)'] = pred
                    # t-2 for next hour is t-1 of current hour
                    row['travel_time(t-2)'] = current_day_preds[h]['prediction'] if h >= 0 else next_row_actual['travel_time(t-2)']
                    # t-3 for next hour is t-2 of current hour
                    row['travel_time(t-3)'] = current_day_preds[h-1]['prediction'] if h >= 1 else next_row_actual['travel_time(t-3)']
                    
                    # t-24 and t-168 stay as ground truth (as they are outside the 24h prediction window)
            
            results_24.extend(current_day_preds)
            
    df_24 = pd.DataFrame(results_24)
    end_pred_24 = time.time()
    pred_time_24 = end_pred_24 - start_pred_24
    
    # 5. Evaluation Metrics
    print("Calculating metrics...")
    performance_file = os.path.join(output_dir, 'performance.txt')
    with open(performance_file, 'w') as f:
        f.write(f"Training Time: {train_time:.4f}s\n")
        f.write(f"1-Step Prediction Time: {pred_time_1:.4f}s\n")
        f.write(f"24-Step Prediction Time: {pred_time_24:.4f}s\n\n")
        
        for horizon, data, act_col, pred_col in [('1-Step', df_test, 'travel_time', 'pred_1step'), ('24-Step', df_24, 'actual', 'prediction')]:
            f.write(f"--- {horizon} Metrics ---\n")
            route_metrics = []
            for (gs, ge), group in data.groupby(['gate_start', 'gate_end']):
                m = calculate_metrics(group[act_col], group[pred_col])
                route_metrics.append(m)
                f.write(f"Route {gs}->{ge}: R2={m['R2']:.4f}, MAE={m['MAE']:.4f}, MAPE={m['MAPE']:.4f}, RMSE={m['RMSE']:.4f}\n")
            
            avg_m = {k: np.mean([x[k] for x in route_metrics]) for k in route_metrics[0].keys()}
            f.write(f"\nMean {horizon}: R2={avg_m['R2']:.4f}, MAE={avg_m['MAE']:.4f}, MAPE={avg_m['MAPE']:.4f}, RMSE={avg_m['RMSE']:.4f}\n\n")

    # 6. Visualization
    print("Generating plots...")
    
    # 1-Step Plots
    for start_date, end_date in [('2026-02-16', '2026-02-22'), ('2026-02-02', '2026-02-08')]:
        mask = (df_test['date'] >= start_date) & (df_test['date'] <= end_date)
        df_sub = df_test[mask]
        for (gs, ge), group in df_sub.groupby(['gate_start', 'gate_end']):
            plot_comparison(group, f"1-Step: Route {gs}-{ge} ({start_date} to {end_date})", 
                            os.path.join(output_dir, f"1step_{gs}_{ge}_{start_date}.png"),
                            pred_col='pred_1step')

    # 1-Step Outliers (Top 5)
    df_test['abs_err'] = (df_test['travel_time'] - df_test['pred_1step']).abs()
    # Find top 5 outlier periods (e.g. 24h windows with max error)
    outliers_1 = df_test.groupby(['gate_start', 'gate_end', 'date_only'])['abs_err'].mean().nlargest(5).reset_index()
    for i, row in outliers_1.iterrows():
        mask = (df_test['gate_start'] == row['gate_start']) & (df_test['gate_end'] == row['gate_end']) & (df_test['date_only'] == row['date_only'])
        plot_comparison(df_test[mask], f"1-Step Outlier #{i+1}: {row['gate_start']}-{row['gate_end']} on {row['date_only']}",
                        os.path.join(output_dir, f"1step_outlier_{i+1}.png"), pred_col='pred_1step')

    # 24-Step Plots
    df_24['date_only'] = df_24['date'].dt.date
    for target_date in ['2026-02-16', '2026-02-02']:
        target_dt = pd.to_datetime(target_date).date()
        df_sub = df_24[df_24['date_only'] == target_dt]
        for (gs, ge), group in df_sub.groupby(['gate_start', 'gate_end']):
            plot_comparison(group, f"24-Step: Route {gs}-{ge} ({target_date})",
                            os.path.join(output_dir, f"24step_{gs}_{ge}_{target_date}.png"),
                            actual_col='actual', pred_col='prediction', time_col='date')

    # 24-Step Outliers (Top 5)
    df_24['abs_err'] = (df_24['actual'] - df_24['prediction']).abs()
    outliers_24 = df_24.groupby(['gate_start', 'gate_end', 'date_only'])['abs_err'].mean().nlargest(5).reset_index()
    for i, row in outliers_24.iterrows():
        mask = (df_24['gate_start'] == row['gate_start']) & (df_24['gate_end'] == row['gate_end']) & (df_24['date_only'] == row['date_only'])
        plot_comparison(df_24[mask], f"24-Step Outlier #{i+1}: {row['gate_start']}-{row['gate_end']} on {row['date_only']}",
                        os.path.join(output_dir, f"24step_outlier_{i+1}.png"),
                        actual_col='actual', pred_col='prediction', time_col='date')

    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
