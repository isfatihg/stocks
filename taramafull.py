#!/usr/bin/env python3
"""
BIST100 Master Strategy Scanner
--------------------------------
Combines all 8 strategy scanners into one program:
1. 52-Week High + Volume Surge
2. Bollinger Band Breakout + Volume 3x
3. Classic EMA 8/21 + RSI Filter
4. Post-Consolidation Breakout
5. Monthly MACD + Daily Doji Breakout
6. Golden Cross + Momentum Bomb
7. Weekly SuperTrend
8. VWAP + Volume Signal

All outputs saved to 'tarama' folder.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Check for TA-Lib
try:
    import talib
except ImportError:
    print("❌ TA-Lib is not installed!")
    print("Please install it first:")
    print("  pip install TA-Lib")
    exit()

# Create output directory
OUTPUT_DIR = "tarama070226"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_bist100_stocks():
    """Read BIST100 stocks from bisttum.csv"""
    try:
        df = pd.read_csv('bisttum.csv', header=None)
        stocks = df[0].tolist()
        return [f"{s}.IS" for s in stocks if isinstance(s, str) and s.isalpha()]
    except FileNotFoundError:
        print("❌ bisttum.csv not found!")
        return []

# ==================== STRATEGY 1: 52-Week High + Volume ====================
def run_52week_scanner(stocks):
    """52-Week High with Volume Surge Strategy"""
    print("\n📈 Strategy 1: 52-Week High + Volume Surge")
    
    def calculate_indicators(df):
        if len(df) < 252:
            return df
        high_prices = df['High'].astype(np.float64).values
        volume_data = df['Volume'].astype(np.float64).values
        df['High52Week'] = talib.MAX(high_prices, timeperiod=252)
        df['VolAvg50'] = talib.SMA(volume_data, timeperiod=50)
        return df
    
    def check_signal(stock, weeks=104):
        try:
            ticker = yf.Ticker(stock)
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks + 10)
            df = ticker.history(start=start_date, end=end_date, interval='1wk')
            if df.empty or len(df) < 252:
                return False, None, None
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = df[col].astype(np.float64)
            df = calculate_indicators(df)
            last = df.iloc[-1]
            close_at_high = last['Close'] >= (last['High52Week'] * 0.999)
            volume_surge = last['Volume'] > (last['VolAvg50'] * 1.5)
            if close_at_high and volume_surge:
                conditions = {
                    'Date': last.name.strftime('%Y-%m-%d'),
                    'Close': f"{last['Close']:.2f}",
                    'High52Week': f"{last['High52Week']:.2f}",
                    'Volume': f"{last['Volume']:,.0f}",
                    'VolAvg50': f"{last['VolAvg50']:,.0f}",
                    'Volume_Ratio': f"{last['Volume'] / last['VolAvg50']:.1f}x"
                }
                return True, conditions, df
            return False, None, None
        except Exception as e:
            print(f"❌ Error: {e}")
            return False, None, None
    
    def visualize(df, stock, signal_date):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        ax1.plot(df.index, df['Close'], 'k-', label='Close', linewidth=1.5)
        ax1.plot(df.index, df['High52Week'], 'orange', linestyle='--', label='52-Week High', linewidth=2)
        close_near_high = df['Close'] >= (df['High52Week'] * 0.999)
        ax1.fill_between(df.index, df['Close'], df['High52Week'], where=close_near_high, alpha=0.2, color='orange')
        signal_idx = df.index.get_loc(signal_date)
        ax1.scatter(df.index[signal_idx], df['Close'].iloc[signal_idx], color='orange', s=300, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        ax1.set_title(f'{stock} - 52-Week High + Volume Surge', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (TL)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax2.bar(df.index, df['Volume'], color='gray', alpha=0.6, width=5, label='Volume')
        ax2.plot(df.index, df['VolAvg50'], 'orange', linewidth=2, label='Volume Avg 50')
        ax2.bar(df.index[signal_idx], df['Volume'].iloc[signal_idx], color='red', alpha=0.8, width=5, label='Volume Surge')
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    signals = []
    for i, stock in enumerate(stocks, 1):
        print(f"[{i:3d}/{len(stocks)}] {stock:<12}", end=' ')
        signal, conditions, df = check_signal(stock)
        if signal:
            print("🎯 SIGNAL!")
            signals.append(conditions)
            try:
                fig = visualize(df, stock, conditions['Date'])
                chart_file = os.path.join(OUTPUT_DIR, f"{stock}_{conditions['Date']}_52week.png")
                fig.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  📊 Chart saved: {chart_file}")
            except Exception as e:
                print(f"  ⚠️ Chart error: {e}")
        else:
            print("✓ No signal")
    
    if signals:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(OUTPUT_DIR, f'bist100_52week_signals_{timestamp}.csv')
        pd.DataFrame(signals).to_csv(filename, index=False)
        print(f"💾 Saved to {filename}")

# ==================== STRATEGY 2: Bollinger Band Breakout ====================
def run_bollinger_scanner(stocks):
    """Bollinger Band Breakout with Triple Volume"""
    print("\n📈 Strategy 2: Bollinger Band Breakout")
    
    def calculate_bollinger_bands(df, period=20, std_dev=2):
        if len(df) < period:
            return df
        close_prices = df['Close'].astype(np.float64).values
        df['Basis'] = talib.SMA(close_prices, timeperiod=period)
        df['StdDev'] = talib.STDDEV(close_prices, timeperiod=period)
        df['Upper'] = df['Basis'] + (std_dev * df['StdDev'])
        df['Lower'] = df['Basis'] - (std_dev * df['StdDev'])
        return df
    
    def calculate_volume_sma(df, period=20):
        if len(df) < period:
            return df
        volume_data = df['Volume'].astype(np.float64).values
        df['VolAvg'] = talib.SMA(volume_data, timeperiod=period)
        return df
    
    def check_signal(stock, weeks=52):
        try:
            ticker = yf.Ticker(stock)
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks + 10)
            df = ticker.history(start=start_date, end=end_date, interval='1wk')
            if df.empty or len(df) < 21:
                return False, None, None
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = df[col].astype(np.float64)
            df = calculate_bollinger_bands(df)
            df = calculate_volume_sma(df)
            if len(df) < 2:
                return False, None, None
            prev_week, curr_week = df.iloc[-2], df.iloc[-1]
            prev_close_below_upper = prev_week['Close'] <= prev_week['Upper']
            curr_close_above_prev_upper = curr_week['Close'] > prev_week['Upper']
            volume_surge = curr_week['Volume'] > (prev_week['VolAvg'] * 3)
            if prev_close_below_upper and curr_close_above_prev_upper and volume_surge:
                conditions = {
                    'Date': curr_week.name.strftime('%Y-%m-%d'),
                    'Close': f"{curr_week['Close']:.2f}",
                    'Upper_Band': f"{prev_week['Upper']:.2f}",
                    'Volume': f"{curr_week['Volume']:,.0f}",
                    'Vol_Avg': f"{prev_week['VolAvg']:,.0f}",
                    'Volume_Ratio': f"{curr_week['Volume'] / prev_week['VolAvg']:.1f}x"
                }
                return True, conditions, df
            return False, None, None
        except Exception as e:
            print(f"❌ Error: {e}")
            return False, None, None
    
    def visualize(df, stock, signal_date):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        ax1.plot(df.index, df['Close'], 'k-', label='Close', linewidth=1.5)
        ax1.plot(df.index, df['Basis'], 'b-', label='Basis (SMA 20)', linewidth=1)
        ax1.plot(df.index, df['Upper'], 'r-', label='Upper Band', linewidth=1)
        ax1.plot(df.index, df['Lower'], 'g-', label='Lower Band', linewidth=1)
        ax1.fill_between(df.index, df['Upper'], df['Lower'], alpha=0.1, color='gray', label='Bollinger Band')
        signal_idx = df.index.get_loc(signal_date)
        ax1.scatter(df.index[signal_idx], df['Close'].iloc[signal_idx], color='red', s=300, marker='^', edgecolors='black', linewidths=1.5, zorder=5)
        ax1.set_title(f'{stock} - Bollinger Band Breakout + Volume 3x', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (TL)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax2.bar(df.index, df['Volume'], color='gray', alpha=0.6, width=5, label='Volume')
        ax2.plot(df.index, df['VolAvg'], 'orange', linewidth=2, label='Volume Avg 20')
        ax2.bar(df.index[signal_idx], df['Volume'].iloc[signal_idx], color='red', alpha=0.8, width=5, label='Volume Surge')
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    signals = []
    for i, stock in enumerate(stocks, 1):
        print(f"[{i:3d}/{len(stocks)}] {stock:<12}", end=' ')
        signal, conditions, df = check_signal(stock)
        if signal:
            print("🎯 SIGNAL!")
            signals.append(conditions)
            try:
                fig = visualize(df, stock, conditions['Date'])
                chart_file = os.path.join(OUTPUT_DIR, f"{stock}_{conditions['Date']}_bollinger.png")
                fig.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  📊 Chart saved: {chart_file}")
            except Exception as e:
                print(f"  ⚠️ Chart error: {e}")
        else:
            print("✓ No signal")
    
    if signals:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(OUTPUT_DIR, f'bist100_bollinger_signals_{timestamp}.csv')
        pd.DataFrame(signals).to_csv(filename, index=False)
        print(f"💾 Saved to {filename}")

# ==================== STRATEGY 3: Classic EMA + RSI ====================
def run_classic_scanner(stocks):
    """Classic EMA 8/21 Crossover with RSI Filter"""
    print("\n📈 Strategy 3: Classic EMA + RSI")
    
    def calculate_indicators(df):
        if len(df) < 21:
            return df
        df['EMA8'] = talib.EMA(df['Close'].values, timeperiod=8)
        df['EMA21'] = talib.EMA(df['Close'].values, timeperiod=21)
        df['RSI'] = talib.RSI(df['Close'].values, timeperiod=14)
        return df
    
    def check_signal(stock, weeks=52):
        try:
            ticker = yf.Ticker(stock)
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks + 10)
            df = ticker.history(start=start_date, end=end_date, interval='1wk')
            if df.empty or len(df) < 21:
                return False, None, None
            df = calculate_indicators(df)
            if len(df) < 2:
                return False, None, None
            prev_week, curr_week = df.iloc[-2], df.iloc[-1]
            crossover = prev_week['EMA8'] <= prev_week['EMA21'] and curr_week['EMA8'] > curr_week['EMA21']
            close_above_ema21 = curr_week['Close'] > curr_week['EMA21']
            rsi_in_range = 55 < curr_week['RSI'] < 80
            if crossover and close_above_ema21 and rsi_in_range:
                conditions = {
                    'Date': curr_week.name.strftime('%Y-%m-%d'),
                    'Close': f"{curr_week['Close']:.2f}",
                    'EMA8': f"{curr_week['EMA8']:.2f}",
                    'EMA21': f"{curr_week['EMA21']:.2f}",
                    'RSI': f"{curr_week['RSI']:.1f}"
                }
                return True, conditions, df
            return False, None, None
        except Exception as e:
            print(f"❌ Error: {e}")
            return False, None, None
    
    def visualize(df, stock, signal_date):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        ax1.plot(df.index, df['Close'], 'k-', label='Close', linewidth=1.5)
        ax1.plot(df.index, df['EMA8'], 'b-', label='EMA 8', linewidth=1)
        ax1.plot(df.index, df['EMA21'], 'r-', label='EMA 21', linewidth=1)
        signal_idx = df.index.get_loc(signal_date)
        ax1.scatter(df.index[signal_idx], df['Close'].iloc[signal_idx], color='fuchsia', s=300, marker='^', edgecolors='black', linewidths=1.5, zorder=5)
        ax1.set_title(f'{stock} - EMA 8/21 Crossover + RSI Filter', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (TL)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax2.plot(df.index, df['RSI'], 'blue', linewidth=1.5)
        ax2.plot(df.index, df['RSI'], 'purple', linewidth=1.5)
        ax2.axhline(55, color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(80, color='orange', linestyle='--', alpha=0.7)
        ax2.fill_between(df.index, 55, 80, alpha=0.1, color='orange', label='RSI 55-80 Zone')
        ax2.scatter(df.index[signal_idx], df['RSI'].iloc[signal_idx], color='fuchsia', s=200, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
        ax2.set_ylabel('RSI')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    signals = []
    for i, stock in enumerate(stocks, 1):
        print(f"[{i:3d}/{len(stocks)}] {stock:<12}", end=' ')
        signal, conditions, df = check_signal(stock)
        if signal:
            print("🎯 SIGNAL!")
            signals.append(conditions)
            try:
                fig = visualize(df, stock, conditions['Date'])
                chart_file = os.path.join(OUTPUT_DIR, f"{stock}_{conditions['Date']}_classic.png")
                fig.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  📊 Chart saved: {chart_file}")
            except Exception as e:
                print(f"  ⚠️ Chart error: {e}")
        else:
            print("✓ No signal")
    
    if signals:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(OUTPUT_DIR, f'bist100_classic_signals_{timestamp}.csv')
        pd.DataFrame(signals).to_csv(filename, index=False)
        print(f"💾 Saved to {filename}")

# ==================== STRATEGY 4: Post-Consolidation Breakout ====================
def run_consolidation_scanner(stocks):
    """Post-Consolidation Breakout"""
    print("\n📈 Strategy 4: Post-Consolidation Breakout")
    
    def calculate_indicators(df):
        if len(df) < 14:
            return df
        high_prices = df['High'].astype(np.float64).values
        low_prices = df['Low'].astype(np.float64).values
        close_prices = df['Close'].astype(np.float64).values
        volume_data = df['Volume'].astype(np.float64).values
        df['High3'] = talib.MAX(high_prices, timeperiod=3)
        df['Low3'] = talib.MIN(low_prices, timeperiod=3)
        df['Range3'] = df['High3'] - df['Low3']
        df['ATR14'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        df['VolAvg20'] = talib.SMA(volume_data, timeperiod=20)
        return df
    
    def check_signal(stock, weeks=104):
        try:
            ticker = yf.Ticker(stock)
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks + 10)
            df = ticker.history(start=start_date, end=end_date, interval='1wk')
            if df.empty or len(df) < 14:
                return False, None, None
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = df[col].astype(np.float64)
            df = calculate_indicators(df)
            if len(df) < 2:
                return False, None, None
            prev_week, curr_week = df.iloc[-2], df.iloc[-1]
            consolidation = prev_week['Range3'] < (prev_week['ATR14'] * 1.5)
            breakout = curr_week['Close'] > prev_week['High3']
            volume_surge = curr_week['Volume'] > (prev_week['VolAvg20'] * 2)
            if consolidation and breakout and volume_surge:
                conditions = {
                    'Date': curr_week.name.strftime('%Y-%m-%d'),
                    'Close': f"{curr_week['Close']:.2f}",
                    'Prev_High3': f"{prev_week['High3']:.2f}",
                    'Prev_Range3': f"{prev_week['Range3']:.2f}",
                    'Prev_ATR14': f"{prev_week['ATR14']:.2f}",
                    'Volume': f"{curr_week['Volume']:,.0f}",
                    'Vol_Avg': f"{prev_week['VolAvg20']:,.0f}",
                    'Volume_Ratio': f"{curr_week['Volume'] / prev_week['VolAvg20']:.1f}x"
                }
                return True, conditions, df
            return False, None, None
        except Exception as e:
            print(f"❌ Error: {e}")
            return False, None, None
    
    def visualize(df, stock, signal_date):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        ax1.plot(df.index, df['Close'], 'k-', label='Close', linewidth=1.5)
        ax1.plot(df.index, df['High3'], 'b-', label='3-Day High', linewidth=1)
        ax1.plot(df.index, df['Low3'], 'g-', label='3-Day Low', linewidth=1)
        consolidation = df['Range3'] < (df['ATR14'] * 1.5)
        ax1.fill_between(df.index, df['High3'], df['Low3'], where=consolidation, alpha=0.2, color='yellow', label='Consolidation')
        signal_idx = df.index.get_loc(signal_date)
        ax1.scatter(df.index[signal_idx], df['Close'].iloc[signal_idx], color='white', s=300, marker='^', edgecolors='black', linewidths=1.5, zorder=5)
        ax1.set_title(f'{stock} - Post-Consolidation Breakout', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (TL)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax2.plot(df.index, df['Range3'], 'b-', label='3-Day Range', linewidth=1.5)
        ax2.plot(df.index, df['ATR14'] * 1.5, 'r-', label='1.5 * ATR(14)', linewidth=1.5)
        ax2.scatter(df.index[signal_idx-1], df['Range3'].iloc[signal_idx-1], color='yellow', s=200, marker='o', edgecolors='black', linewidths=1.5, zorder=5)
        ax2.set_ylabel('Range/ATR')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax3.bar(df.index, df['Volume'], color='gray', alpha=0.6, width=5, label='Volume')
        ax3.plot(df.index, df['VolAvg20'], 'orange', linewidth=2, label='Volume Avg 20')
        ax3.bar(df.index[signal_idx], df['Volume'].iloc[signal_idx], color='red', alpha=0.8, width=5, label='Volume Surge')
        ax3.set_ylabel('Volume')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    signals = []
    for i, stock in enumerate(stocks, 1):
        print(f"[{i:3d}/{len(stocks)}] {stock:<12}", end=' ')
        signal, conditions, df = check_signal(stock)
        if signal:
            print("🎯 SIGNAL!")
            signals.append(conditions)
            try:
                fig = visualize(df, stock, conditions['Date'])
                chart_file = os.path.join(OUTPUT_DIR, f"{stock}_{conditions['Date']}_consolidation.png")
                fig.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  📊 Chart saved: {chart_file}")
            except Exception as e:
                print(f"  ⚠️ Chart error: {e}")
        else:
            print("✓ No signal")
    
    if signals:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(OUTPUT_DIR, f'bist100_consolidation_signals_{timestamp}.csv')
        pd.DataFrame(signals).to_csv(filename, index=False)
        print(f"💾 Saved to {filename}")

# ==================== STRATEGY 5: Monthly MACD + Daily Doji ====================
def run_macd_scanner(stocks):
    """Monthly MACD Crossover and Daily Doji Breakout"""
    print("\n📈 Strategy 5: Monthly MACD + Daily Doji Breakout")
    
    def calculate_monthly_macd(df):
        if len(df) < 26:
            return df
        close_prices = df['Close'].astype(np.float64).values
        macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Hist'] = hist
        return df
    
    def check_monthly_macd_crossover(df):
        if len(df) < 2:
            return False, None
        prev_month, curr_month = df.iloc[-2], df.iloc[-1]
        crossover = prev_month['MACD'] <= prev_month['MACD_Signal'] and curr_month['MACD'] > curr_month['MACD_Signal']
        if crossover:
            return True, curr_month.name.strftime('%Y-%m')
        return False, None
    
    def check_daily_breakout(stock, crossover_month):
        try:
            ticker = yf.Ticker(stock)
            month_start = crossover_month.replace(day=1)
            next_month = month_start.replace(month=month_start.month + 1) if month_start.month < 12 else month_start.replace(year=month_start.year + 1, month=1)
            month_end = next_month - timedelta(days=1)
            start_date = month_start - timedelta(days=5)
            end_date = month_end + timedelta(days=5)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            if df.empty or len(df) < 2:
                return False, None
            df['Close'] = df['Close'].astype(np.float64)
            df['Open'] = df['Open'].astype(np.float64)
            df['High'] = df['High'].astype(np.float64)
            for i in range(1, len(df)):
                prev_day, curr_day = df.iloc[i-1], df.iloc[i]
                bullish_candle = curr_day['Close'] > curr_day['Open']
                breakout = curr_day['Close'] > prev_day['High']
                if bullish_candle and breakout:
                    conditions = {
                        'Date': curr_day.name.strftime('%Y-%m-%d'),
                        'Close': f"{curr_day['Close']:.2f}",
                        'Open': f"{curr_day['Open']:.2f}",
                        'Prev_High': f"{prev_day['High']:.2f}",
                        'Volume': f"{curr_day['Volume']:,.0f}"
                    }
                    return True, conditions
            return False, None
        except Exception as e:
            print(f"❌ Error: {e}")
            return False, None
    
    def visualize(monthly_df, daily_df, stock, signal_date):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
        ax1.plot(monthly_df.index, monthly_df['MACD'], 'b-', label='MACD', linewidth=2)
        ax1.plot(monthly_df.index, monthly_df['MACD_Signal'], 'r-', label='MACD Signal', linewidth=2)
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.7)
        crossover_idx = monthly_df.index.get_loc(signal_date)
        ax1.scatter(monthly_df.index[crossover_idx], monthly_df['MACD'].iloc[crossover_idx], color='purple', s=300, marker='^', edgecolors='black', linewidths=1.5, zorder=5)
        ax1.set_title(f'{stock} - Monthly MACD', fontsize=12, fontweight='bold')
        ax1.set_ylabel('MACD')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(monthly_df.index, monthly_df['Close'], 'k-', label='Close', linewidth=1.5)
        ax2.set_title('Monthly Price', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Price (TL)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        daily_start = pd.to_datetime(signal_date) - timedelta(days=30)
        daily_end = pd.to_datetime(signal_date) + timedelta(days=30)
        daily_subset = daily_df[(daily_df.index >= daily_start) & (daily_df.index <= daily_end)]
        ax3.plot(daily_subset.index, daily_subset['Close'], 'k-', label='Close', linewidth=1.5)
        ax3.plot(daily_subset.index, daily_subset['Open'], 'b-', label='Open', linewidth=1)
        breakout_idx = daily_subset.index.get_loc(signal_date)
        ax3.scatter(daily_subset.index[breakout_idx], daily_subset['Close'].iloc[breakout_idx], color='purple', s=300, marker='^', edgecolors='black', linewidths=1.5, zorder=5)
        ax3.set_title('Daily Price', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Price (TL)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        ax4.bar(daily_subset.index, daily_subset['Volume'], color='gray', alpha=0.6, width=1)
        ax4.set_title('Daily Volume', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Volume')
        ax4.set_xlabel('Date')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        return fig
    
    signals = []
    for i, stock in enumerate(stocks, 1):
        print(f"[{i:3d}/{len(stocks)}] {stock:<12}", end=' ')
        try:
            ticker = yf.Ticker(stock)
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=52*3)
            monthly_df = ticker.history(start=start_date, end=end_date, interval='1mo')
            if monthly_df.empty or len(monthly_df) < 26:
                print("⚠️ Insufficient monthly data")
                continue
            monthly_df = calculate_monthly_macd(monthly_df)
            crossover_found, crossover_date = check_monthly_macd_crossover(monthly_df)
            if not crossover_found:
                print("✓ No monthly MACD crossover")
                continue
            breakout_found, breakout_conditions = check_daily_breakout(stock, pd.to_datetime(crossover_date))
            if breakout_found:
                print("🎯 SIGNAL!")
                signals.append({
                    'Stock': stock,
                    'Crossover_Month': crossover_date,
                    'Breakout_Date': breakout_conditions['Date'],
                    'Close': breakout_conditions['Close'],
                    'Open': breakout_conditions['Open'],
                    'Prev_High': breakout_conditions['Prev_High'],
                    'Volume': breakout_conditions['Volume']
                })
                try:
                    daily_start = pd.to_datetime(crossover_date) - timedelta(days=60)
                    daily_end = pd.to_datetime(crossover_date) + timedelta(days=60)
                    daily_df = ticker.history(start=daily_start, end=daily_end, interval='1d')
                    fig = visualize(monthly_df, daily_df, stock, crossover_date)
                    chart_file = os.path.join(OUTPUT_DIR, f"{stock}_{breakout_conditions['Date']}_macd.png")
                    fig.savefig(chart_file, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  📊 Chart saved: {chart_file}")
                except Exception as e:
                    print(f"  ⚠️ Chart error: {e}")
            else:
                print("✓ No daily breakout")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if signals:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(OUTPUT_DIR, f'bist100_macd_signals_{timestamp}.csv')
        pd.DataFrame(signals).to_csv(filename, index=False)
        print(f"💾 Saved to {filename}")

# ==================== STRATEGY 6: Golden Cross + Momentum Bomb ====================
def run_golden_cross_scanner(stocks):
    """Golden Cross + Momentum Bomb"""
    print("\n📈 Strategy 6: Golden Cross + Momentum Bomb")
    
    def calculate_indicators(df):
        if len(df) < 21:
            return df
        close_prices = df['Close'].astype(np.float64).values
        volume_data = df['Volume'].astype(np.float64).values
        df['EMA9'] = talib.EMA(close_prices, timeperiod=9)
        df['EMA21'] = talib.EMA(close_prices, timeperiod=21)
        df['RSI'] = talib.RSI(close_prices, timeperiod=14)
        df['VolAvg20'] = talib.SMA(volume_data, timeperiod=20)
        df['20DayHigh'] = talib.MAX(close_prices, timeperiod=20)
        return df
    
    def check_signal(stock, days=90):
        try:
            ticker = yf.Ticker(stock)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 10)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            if df.empty or len(df) < 21:
                return False, None, None
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = df[col].astype(np.float64)
            df = calculate_indicators(df)
            if len(df) < 2:
                return False, None, None
            prev_day, curr_day = df.iloc[-2], df.iloc[-1]
            crossover = prev_day['EMA9'] <= prev_day['EMA21'] and curr_day['EMA9'] > curr_day['EMA21']
            close_above_ema21 = curr_day['Close'] > curr_day['EMA21']
            rsi_in_range = 50 < curr_day['RSI'] < 80
            volume_surge = curr_day['Volume'] > (curr_day['VolAvg20'] * 1.5)
            new_high = curr_day['Close'] >= curr_day['20DayHigh']
            if crossover and close_above_ema21 and rsi_in_range and volume_surge and new_high:
                conditions = {
                    'Date': curr_day.name.strftime('%Y-%m-%d'),
                    'Close': f"{curr_day['Close']:.2f}",
                    'RSI': f"{curr_day['RSI']:.1f}",
                    'Volume': f"{curr_day['Volume']:,.0f}",
                    'VolAvg': f"{curr_day['VolAvg20']:,.0f}"
                }
                return True, conditions, df
            return False, None, None
        except Exception as e:
            print(f"❌ Error: {e}")
            return False, None, None
    
    def visualize(df, stock, signal_date):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        ax1.plot(df.index, df['Close'], 'k-', label='Close', linewidth=1.5)
        ax1.plot(df.index, df['EMA9'], 'b-', label='EMA 9', linewidth=1)
        ax1.plot(df.index, df['EMA21'], 'r-', label='EMA 21', linewidth=1)
        ax1.plot(df.index, df['20DayHigh'], 'g--', label='20-Day High', alpha=0.7)
        signal_idx = df.index.get_loc(signal_date)
        ax1.scatter(df.index[signal_idx], df['Close'].iloc[signal_idx], color='yellow', s=300, marker='*', edgecolors='black', linewidths=1.5, zorder=5)
        ax1.set_title(f'{stock} - Golden Cross + Momentum Bomb', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (TL)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax2.plot(df.index, df['RSI'], 'purple', linewidth=1.5)
        ax2.axhline(50, color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(80, color='orange', linestyle='--', alpha=0.7)
        ax2.fill_between(df.index, 50, 80, alpha=0.1, color='orange')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        ax3.bar(df.index, df['Volume'], color='gray', alpha=0.6, width=1, label='Volume')
        ax3.plot(df.index, df['VolAvg20'], 'orange', linewidth=2, label='Volume Avg 20')
        ax3.bar(df.index[signal_idx], df['Volume'].iloc[signal_idx], color='red', alpha=0.8, width=1)
        ax3.set_ylabel('Volume')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    signals = []
    for i, stock in enumerate(stocks, 1):
        print(f"[{i:3d}/{len(stocks)}] {stock:<12}", end=' ')
        signal, conditions, df = check_signal(stock)
        if signal:
            print("🎯 SIGNAL!")
            signals.append(conditions)
            try:
                fig = visualize(df, stock, conditions['Date'])
                chart_file = os.path.join(OUTPUT_DIR, f"{stock}_{conditions['Date']}_golden.png")
                fig.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  📊 Chart saved: {chart_file}")
            except Exception as e:
                print(f"  ⚠️ Chart error: {e}")
        else:
            print("✓ No signal")
    
    if signals:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(OUTPUT_DIR, f'bist100_golden_signals_{timestamp}.csv')
        pd.DataFrame(signals).to_csv(filename, index=False)
        print(f"💾 Saved to {filename}")

# ==================== STRATEGY 7: Weekly SuperTrend ====================
def run_supertrend_scanner(stocks):
    """Weekly SuperTrend"""
    print("\n📈 Strategy 7: Weekly SuperTrend")
    
    def calculate_supertrend(df, period=3, multiplier=10):
        if len(df) < period:
            return df
        df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())))
        df['ATR'] = df['TR'].rolling(window=period).mean()
        df['HL2'] = (df['High'] + df['Low']) / 2
        df['Upper'] = df['HL2'] + (multiplier * df['ATR'])
        df['Lower'] = df['HL2'] - (multiplier * df['ATR'])
        df['SuperTrend'] = np.nan
        for i in range(period, len(df)):
            if i == period:
                df.at[df.index[i], 'SuperTrend'] = df['Lower'].iloc[i] if df['Close'].iloc[i] > df['Upper'].iloc[i] else df['Upper'].iloc[i]
            else:
                prev_st = df['SuperTrend'].iloc[i-1]
                if df['Close'].iloc[i-1] <= prev_st:
                    df.at[df.index[i], 'SuperTrend'] = df['Upper'].iloc[i]
                else:
                    df.at[df.index[i], 'SuperTrend'] = df['Lower'].iloc[i]
        df.dropna(subset=['SuperTrend'], inplace=True)
        return df
    
    def check_signal(stock, weeks=52):
        try:
            ticker = yf.Ticker(stock)
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks + 10)
            df = ticker.history(start=start_date, end=end_date, interval='1wk')
            if df.empty or len(df) < 20:
                return False, None, None
            df = calculate_supertrend(df)
            if len(df) < 5:
                return False, None, None
            prev_week, curr_week = df.iloc[-2], df.iloc[-1]
            price_crossover = curr_week['Close'] > prev_week['SuperTrend']
            prev_below_st = prev_week['Close'] <= prev_week['SuperTrend']
            price_above_st = curr_week['Close'] > curr_week['SuperTrend']
            if price_crossover and prev_below_st and price_above_st:
                conditions = {
                    'Date': curr_week.name.strftime('%Y-%m-%d'),
                    'Close': f"{curr_week['Close']:.2f}",
                    'SuperTrend': f"{curr_week['SuperTrend']:.2f}",
                    'Volume': f"{curr_week['Volume']:,.0f}"
                }
                return True, conditions, df
            return False, None, None
        except Exception as e:
            print(f"❌ Error: {e}")
            return False, None, None
    
    def visualize(df, stock, signal_date):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        ax1.plot(df.index, df['Close'], 'k-', label='Kapanış', linewidth=2)
        ax1.plot(df.index, df['SuperTrend'], 'teal', label='SuperTrend (3,10)', linewidth=2)
        signal_idx = df.index.get_loc(signal_date)
        ax1.scatter(df.index[signal_idx], df['Close'].iloc[signal_idx], color='teal', s=200, marker='^', edgecolors='black', linewidths=1.5, zorder=5)
        prev_idx = signal_idx - 1
        ax1.scatter(df.index[prev_idx], df['Close'].iloc[prev_idx], color='red', s=100, marker='o', alpha=0.7)
        ax1.set_title(f'{stock} - Haftalık SuperTrend Alım Sinyali', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Fiyat (TL)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax2.bar(df.index, df['Volume'], color='gray', alpha=0.6, width=5)
        ax2.set_ylabel('Hacim')
        ax2.set_xlabel('Tarih')
        ax2.grid(True, alpha=0.3)
        ax2.bar(df.index[signal_idx], df['Volume'].iloc[signal_idx], color='teal', alpha=0.8, width=5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    signals = []
    for i, stock in enumerate(stocks, 1):
        print(f"[{i:3d}/{len(stocks)}] {stock:<12}", end=' ')
        signal, conditions, df = check_signal(stock)
        if signal:
            print("🎯 SİNYAL!")
            signals.append(conditions)
            try:
                fig = visualize(df, stock, conditions['Date'])
                chart_file = os.path.join(OUTPUT_DIR, f"{stock}_{conditions['Date']}_supertrend.png")
                fig.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  📊 Chart saved: {chart_file}")
            except Exception as e:
                print(f"  ⚠️ Chart error: {e}")
        else:
            print("✓ No signal")
    
    if signals:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(OUTPUT_DIR, f'bist100_supertrend_signals_{timestamp}.csv')
        pd.DataFrame(signals).to_csv(filename, index=False)
        print(f"💾 Saved to {filename}")

# ==================== STRATEGY 8: VWAP + Volume ====================
def run_vwap_scanner(stocks):
    """VWAP + Volume Signal"""
    print("\n📈 Strategy 8: VWAP + Volume Signal")
    
    def calculate_vwap(df):
        if len(df) < 1:
            return df
        high_prices = df['High'].astype(np.float64).values
        low_prices = df['Low'].astype(np.float64).values
        close_prices = df['Close'].astype(np.float64).values
        volume_data = df['Volume'].astype(np.float64).values
        hlc3 = (high_prices + low_prices + close_prices) / 3
        cumulative_hlc3_volume = np.cumsum(hlc3 * volume_data)
        cumulative_volume = np.cumsum(volume_data)
        df['VWAP'] = cumulative_hlc3_volume / cumulative_volume
        return df
    
    def calculate_volume_sma(df, period=20):
        if len(df) < period:
            return df
        volume_data = df['Volume'].astype(np.float64).values
        df['VolAvg20'] = talib.SMA(volume_data, timeperiod=period)
        return df
    
    def check_signal(stock, days=90):
        try:
            ticker = yf.Ticker(stock)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 10)
            df = ticker.history(start=start_date, end=end_date, interval='1d')
            if df.empty or len(df) < 20:
                return False, None, None
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = df[col].astype(np.float64)
            df = calculate_vwap(df)
            df = calculate_volume_sma(df)
            last = df.iloc[-1]
            close_above_vwap = last['Close'] > last['VWAP']
            volume_surge = last['Volume'] > (last['VolAvg20'] * 2)
            bullish_candle = last['Close'] > last['Open']
            if close_above_vwap and volume_surge and bullish_candle:
                conditions = {
                    'Date': last.name.strftime('%Y-%m-%d'),
                    'Close': f"{last['Close']:.2f}",
                    'Open': f"{last['Open']:.2f}",
                    'VWAP': f"{last['VWAP']:.2f}",
                    'Volume': f"{last['Volume']:,.0f}",
                    'Vol_Avg': f"{last['VolAvg20']:,.0f}",
                    'Volume_Ratio': f"{last['Volume'] / last['VolAvg20']:.1f}x"
                }
                return True, conditions, df
            return False, None, None
        except Exception as e:
            print(f"❌ Error: {e}")
            return False, None, None
    
    def visualize(df, stock, signal_date):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        ax1.plot(df.index, df['Close'], 'k-', label='Close', linewidth=1.5)
        ax1.plot(df.index, df['VWAP'], 'gold', linestyle='--', label='VWAP', linewidth=2)
        close_above_vwap = df['Close'] > df['VWAP']
        ax1.fill_between(df.index, df['Close'], df['VWAP'], where=close_above_vwap, alpha=0.2, color='gold', label='Above VWAP')
        signal_idx = df.index.get_loc(signal_date)
        ax1.scatter(df.index[signal_idx], df['Close'].iloc[signal_idx], color='gold', s=300, marker='^', edgecolors='black', linewidths=1.5, zorder=5)
        ax1.set_title(f'{stock} - VWAP + Volume 2x', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (TL)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax2.bar(df.index, df['Volume'], color='gray', alpha=0.6, width=1, label='Volume')
        ax2.plot(df.index, df['VolAvg20'], 'orange', linewidth=2, label='Volume Avg 20')
        ax2.bar(df.index[signal_idx], df['Volume'].iloc[signal_idx], color='red', alpha=0.8, width=1, label='Volume Surge')
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    signals = []
    for i, stock in enumerate(stocks, 1):
        print(f"[{i:3d}/{len(stocks)}] {stock:<12}", end=' ')
        signal, conditions, df = check_signal(stock)
        if signal:
            print("🎯 SIGNAL!")
            signals.append(conditions)
            try:
                fig = visualize(df, stock, conditions['Date'])
                chart_file = os.path.join(OUTPUT_DIR, f"{stock}_{conditions['Date']}_vwap.png")
                fig.savefig(chart_file, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"  📊 Chart saved: {chart_file}")
            except Exception as e:
                print(f"  ⚠️ Chart error: {e}")
        else:
            print("✓ No signal")
    
    if signals:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(OUTPUT_DIR, f'bist100_vwap_signals_{timestamp}.csv')
        pd.DataFrame(signals).to_csv(filename, index=False)
        print(f"💾 Saved to {filename}")

# ==================== MAIN FUNCTION ====================
def main():
    print("🚀 BIST100 Master Strategy Scanner")
    print("="*50)
    print("📂 Output directory: tarama/")
    print("="*50)
    
    # Get stocks
    stocks = get_bist100_stocks()
    if not stocks:
        print("❌ No stocks to scan!")
        return
    
    print(f"📊 Scanning {len(stocks)} stocks across all strategies...")
    
    # Run all scanners
    run_52week_scanner(stocks)
    run_bollinger_scanner(stocks)
    run_classic_scanner(stocks)
    run_consolidation_scanner(stocks)
    run_macd_scanner(stocks)
    run_golden_cross_scanner(stocks)
    run_supertrend_scanner(stocks)
    run_vwap_scanner(stocks)
    
    print("\n" + "="*50)
    print("✅ All scans complete!")
    print(f"📁 Results saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("="*50)

if __name__ == "__main__":
    main()
