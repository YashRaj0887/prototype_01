import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. DATA INGESTION
# ==========================================
def fetch_data(ticker_leader, ticker_laggard, period="5d", interval="1m"):
    """
    Fetches intraday data. In production, replace this with Kite Connect API
    to get months of 1-minute or tick data.
    """
    print(f"Fetching data for {ticker_leader} (Leader) and {ticker_laggard} (Laggard)...")

    # Download data
    leader = yf.download(ticker_leader, period=period, interval=interval, progress=False)
    laggard = yf.download(ticker_laggard, period=period, interval=interval, progress=False)

    # Clean and Align Dataframes
    df = pd.DataFrame()
    df['Leader_Price'] = leader['Close']
    df['Leader_Vol'] = leader['Volume']
    df['Laggard_Price'] = laggard['Close']
    df['Laggard_Vol'] = laggard['Volume']

    # Drop NaN values (synchronize timestamps)
    df.dropna(inplace=True)
    return df


# ==========================================
# 2. PARAMETER CALCULATION (The "Research" Phase)
# ==========================================
def analyze_lead_lag(df, max_lag=10):
    """
    Calculates Cross-Correlation to find 'del t' (Optimal Time Lag).
    """
    # Calculate returns for correlation analysis
    l_ret = df['Leader_Price'].pct_change()
    f_ret = df['Laggard_Price'].pct_change()

    correlations =lags = range(0, max_lag + 1)

    for lag in lags:
        # Shift laggard backwards to see if it matches leader's past
        # Correlation(Leader[t], Laggard[t+lag])
        c = l_ret.corr(f_ret.shift(-lag))
        correlations.append(c)

    # Find the lag with maximum correlation
    optimal_lag = np.argmax(correlations)
    max_corr = max(correlations)

    print(f"\n--- ANALYSIS RESULTS ---")
    print(f"Max Correlation: {max_corr:.4f} at Lag: {optimal_lag} minutes")

    return optimal_lag


def calculate_volume_threshold(df, window=20, sigma=2.0):
    """
    Calculates 'V' (Volume Threshold) using dynamic Z-Score.
    V_th = Mean + (Sigma * StdDev)
    """
    # Rolling Mean and Std Dev of Volume
    vol_mean = df['Leader_Vol'].rolling(window=window).mean()
    vol_std = df['Leader_Vol'].rolling(window=window).std()

    # Calculate Z-Score
    df = (df['Leader_Vol'] - vol_mean) / vol_std

    # Define Dynamic Threshold
    df = vol_mean + (sigma * vol_std)

    return df


# ==========================================
# 3. BACKTESTING ENGINE
# ==========================================
def backtest_strategy(df, optimal_lag, sigma, stop_loss_pct, target_pct):
    """
    Simulates trades based on Volume Spikes in Leader.
    """
    signals =[]

    # Transaction Cost (Brokerage + STT + Exchange Charges)
    # Approx 0.083% for intraday equity in India (post-Oct 2024 rates)
    cost_per_trade = 0.00083

    entry_price = 0
    position = 0  # 0: Flat, 1: Long, -1: Short

    equity_curve = [100000] # Start with 1 Lakh capital

    for i in range(len(df) - optimal_lag):
        current_equity = equity_curve[-1]

        # 1. Check for Volume Spike in Leader (Signal Generation)
        is_volume_spike = df['Leader_Vol'].iloc[i] > df.iloc[i]

        # 2. Check Direction (Did SBI move up or down?)
        leader_return = df['Leader_Price'].pct_change().iloc[i]

        # EXECUTION LOGIC
        if position == 0 and is_volume_spike:
            # Entry logic: Wait 'optimal_lag' periods then enter Laggard
            # Note: In real tick-trading, we enter immediately *anticipating* the lag.
            # Here we simulate entering BOI immediately to catch the move.

            entry_price = df['Laggard_Price'].iloc[i]

            if leader_return > 0:
                position = 1  # Long
                # print(f"LONG at {entry_price}")
            elif leader_return < 0:
                position = -1  # Short
                # print(f"SHORT at {entry_price}")

        elif position != 0:
            # Exit Logic: Target or Stop Loss
            curr_price = df['Laggard_Price'].iloc[i]

            if position == 1:
                pct_change = (curr_price - entry_price) / entry_price
            else:
                pct_change = (entry_price - curr_price) / entry_price

            # Check Exit Conditions
            if pct_change >= target_pct or pct_change <= -stop_loss_pct:
                # Close Trade
                pnl = (current_equity * pct_change) - (current_equity * cost_per_trade * 2)
                equity_curve.append(current_equity + pnl)
                position = 0
                entry_price = 0
            else:
                # Holding
                equity_curve.append(current_equity)
        else:
            equity_curve.append(current_equity)

    return equity_curve


# ==========================================
# MAIN EXECUTION
# ==========================================
# Symbols for NSE (yfinance uses.NS extension)
sbi_ticker = "SBIN.NS"
boi_ticker = "BANKINDIA.NS"

# 1. Get Data
data = fetch_data(sbi_ticker, boi_ticker, period="5d", interval="1m")

# 2. Analyze
# Find 'del t'
lag_minutes = analyze_lead_lag(data)

# Find 'V' (Volume Threshold parameters)
# We test a 20-period moving average with a 2-sigma threshold
data = calculate_volume_threshold(data, window=20, sigma=2.0)

# 3. Backtest
# Parameters: Lag from analysis, Sigma=2, SL=0.5%, Target=1.0%
equity = backtest_strategy(data, optimal_lag=lag_minutes, sigma=2.0, stop_loss_pct=0.005, target_pct=0.01)

# 4. Results
print(f"Final Capital: ₹{equity[-1]:.2f}")
plt.plot(equity)
plt.title("Strategy Equity Curve (Simulated)")
plt.show()