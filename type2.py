import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def fetch_data(leader_filepath, laggard_filepath):
    print(f"Loading local data...")
    print(f"Leader File:  {leader_filepath}")
    print(f"Laggard File: {laggard_filepath}")
    
    def load_formatted_csv(filepath):
       
        df = pd.read_csv(filepath, parse_dates=['date'])
        
        df.set_index('date', inplace=True)
        
        df.rename(columns={'close': 'Close', 'volume': 'Volume'}, inplace=True)
        
        return df

    leader = load_formatted_csv(leader_filepath)
    laggard = load_formatted_csv(laggard_filepath)
   
    common_index = leader.index.intersection(laggard.index)
   
    df = pd.DataFrame()
    df['Leader_Price']  = leader.loc[common_index]['Close']
    df['Leader_Vol']    = leader.loc[common_index]['Volume']
    df['Laggard_Price'] = laggard.loc[common_index]['Close']
    df['Laggard_Vol']   = laggard.loc[common_index]['Volume']
    
    # Drop any remaining empty rows
    df.dropna(inplace=True)
    
    print(f"Successfully loaded {len(df)} aligned rows.")
    return df


def analyze_lead_lag(df, max_lag=10):
    l_ret = df['Leader_Price'].pct_change()
    f_ret = df['Laggard_Price'].pct_change()
    
    correlations =[]
    lags = range(0, max_lag + 1)
    
    for lag in lags:
        c = l_ret.corr(f_ret.shift(-lag))
        correlations.append(c)
        
    optimal_lag = np.argmax(correlations)
    max_corr = max(correlations)
    
    print(f"\n--- ANALYSIS RESULTS ---")
    print(f"Max Correlation: {max_corr:.4f} at Lag: {optimal_lag} minutes")
    
    return optimal_lag

def calculate_volume_threshold(df, window=20, sigma=2.0):
    vol_mean = df['Leader_Vol'].rolling(window=window).mean()
    vol_std = df['Leader_Vol'].rolling(window=window).std()
    
    Threshold = vol_mean + (sigma * vol_std)
    
    return Threshold



def backtest_strategy(df, optimal_lag, sigma, stop_loss_pct, target_pct):
    # Transaction Cost (Brokerage + STT) ~ 0.083%
    cost_per_trade = 0.00083 
    
    entry_price = 0
    position = 0 # 0: Flat, 1: Long, -1: Short
    
    equity_curve = [100000] # Start with 1 Lakh capital
    
    for i in range(len(df) - optimal_lag):
        current_equity = equity_curve[-1]
        
        #  We use .iloc[i] to get the value at the current row
        current_vol = df['Leader_Vol'].iloc[i]
        
        # We must select the 'Threshold' column specifically!
        current_threshold = df['Threshold'].iloc[i]
        
        # Check for NaN (at start of data)
        if pd.isna(current_threshold):
            equity_curve.append(current_equity)
            continue

        is_volume_spike = current_vol > current_threshold
        
        # Check Direction
        leader_return = df['Leader_Price'].pct_change().iloc[i]
        
        # EXECUTION LOGIC
        if position == 0 and is_volume_spike:
            # Entry logic (Simulated immediate entry for testing)
            entry_price = df['Laggard_Price'].iloc[i]
            
            if leader_return > 0:
                position = 1 # Long
            elif leader_return < 0:
                position = -1 # Short
            
            equity_curve.append(current_equity)
                
        elif position != 0:
            # Exit Logic
            curr_price = df['Laggard_Price'].iloc[i]
            
            if position == 1:
                pct_change = (curr_price - entry_price) / entry_price
            else:
                pct_change = (entry_price - curr_price) / entry_price
            
            # Check Exit Conditions
            if pct_change >= target_pct or pct_change <= -stop_loss_pct:
                pnl = (current_equity * pct_change) - (current_equity * cost_per_trade * 2)
                equity_curve.append(current_equity + pnl)
                position = 0
            else:
                equity_curve.append(current_equity)
        else:
            equity_curve.append(current_equity)

    return equity_curve


if __name__ == "__main__":
   
    leader_filename = "leader_sbi.csv"
    laggard_filename = "lagger_boi.csv"

    try:
        data = fetch_data(leader_filename, laggard_filename)

        if len(data) > 0:
            lag_minutes = analyze_lead_lag(data)
            
            data['Threshold'] = calculate_volume_threshold(data, window=20, sigma=2.0)
            
            equity = backtest_strategy(data, optimal_lag=lag_minutes, sigma=2.0, stop_loss_pct=0.005, target_pct=0.01)

            #  Plot Results
            print(f"Final Capital: ₹{equity[-1]:.2f}")
            plt.plot(equity)
            plt.title("Strategy Equity Curve")
            plt.show()
        else:
            print("Data loaded but empty. Check if dates in both CSVs match.")
            
    except FileNotFoundError:
        print("ERROR: Could not find the CSV files.")
        print("Make sure 'leader_sbi.csv' and 'lagger_boi.csv' are in the same folder as this script.")