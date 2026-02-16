import yfinance as yf
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Türkçe font
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

print("=== BIST TEKNİK TARAMA (EMA20/50/200 Kırılım + RSI52 + MACD+ + Vol%130) ===")
print("Koşullar (SON GÜN):\n"
      "✅ Son 3 kapanış > EMA20\n"
      "✅ Close > EMA50 (kırılım)\n"
      "✅ RSI(14) >= 52\n"
      "✅ MACD > 0 (pozitif alan)\n"
      "✅ Close > EMA200 (uzun vadeli)\n"
      "✅ Hacim > 20g ort %130")

# 1. CSV OKU (.IS otomatik)
csv_file = 'bisttum.csv'
if not pd.io.common.file_exists(csv_file):
    print(f"❌ {csv_file} yok! Önceki mesajdaki CSV'yi kaydet.")
    exit()

tickers_raw = pd.read_csv(csv_file, header=None)[0].str.strip().tolist()
tickers = [t + '.IS' if not t.endswith('.IS') else t for t in tickers_raw]
print(f"📁 {len(tickers)} hisse taranıyor...")

# 2. Veri İndir (2Y: EMA200 için yeterli)
end_date = datetime.now().strftime('%Y-%m-%d')
recent_start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
print(f"Veri: {recent_start} → {end_date}")

print("Veri indiriliyor (batch, 1-2 dk)...")
data = yf.download(tickers, start=recent_start, end=end_date, group_by='ticker', progress=True, threads=True)

# 3. Tarama Koşulları
matches = []
for ticker in tickers:
    if ticker not in data or data[ticker].empty:
        continue
    
    df = data[ticker].dropna()
    if len(df) < 250:  # EMA200 + buffer
        continue
    
    # İndikatörler (talib)
    ema20 = talib.EMA(df['Close'], timeperiod=20)
    ema50 = talib.EMA(df['Close'], timeperiod=50)
    ema200 = talib.EMA(df['Close'], timeperiod=200)
    rsi = talib.RSI(df['Close'], timeperiod=14)
    macd, macdsignal, _ = talib.MACD(df['Close'])
    
    vol_ma20 = df['Volume'].rolling(20).mean()
    vol_ratio = df['Volume'] / vol_ma20
    
    # Son değerler (NaN temizle)
    last_idx = -1
    while pd.isna([ema200[last_idx], rsi[last_idx], macd[last_idx]]).any() and last_idx > -250:
        last_idx -= 1
    
    if last_idx <= -250:
        continue
    
    close = df['Close'].iloc[last_idx]
    ema20_v = ema20[last_idx]
    ema50_v = ema50[last_idx]
    ema200_v = ema200[last_idx]
    rsi_v = rsi[last_idx]
    macd_v = macd[last_idx]
    vol_r = vol_ratio.iloc[last_idx]
    
    # KOŞULLAR (TAM EŞLEŞME)
    cond1_3days = all(df['Close'].iloc[last_idx-2:last_idx+1] > ema20[last_idx-2:last_idx+1])  # Son 3 > EMA20
    cond2_ema50 = close > ema50_v  # EMA50 kırılımı (üstünde)
    cond3_rsi = rsi_v >= 52
    cond4_macd = macd_v > 0  # Pozitif alan
    cond5_ema200 = close > ema200_v
    cond6_vol = vol_r >= 1.30
    
    if all([cond1_3days, cond2_ema50, cond3_rsi, cond4_macd, cond5_ema200, cond6_vol]):
        # Ek metrikler
        pct_above_ema20 = (close - ema20_v) / ema20_v * 100
        pct_above_ema50 = (close - ema50_v) / ema50_v * 100
        pct_above_ema200 = (close - ema200_v) / ema200_v * 100
        matches.append({
            'Ticker': ticker,
            'Close': round(close, 2),
            '%>EMA20': round(pct_above_ema20, 1),
            '%>EMA50': round(pct_above_ema50, 1),
            '%>EMA200': round(pct_above_ema200, 1),
            'RSI': round(rsi_v, 1),
            'MACD': round(macd_v, 4),
            'Vol%': round(vol_r * 100, 0),
            'Score': round((pct_above_ema50 + pct_above_ema200 + (rsi_v-50)/5 + (vol_r-1)*10), 2)  # Basit skor
        })
        print(f"✅ {ticker}: RSI{rsi_v:.0f} Vol%{vol_r*100:.0f} %{pct_above_ema50:.1f}>EMA50")

print(f"\n🎯 EŞLEŞEN: {len(matches)} / {len(tickers)} hisse")

if matches:
    screen_df = pd.DataFrame(matches).sort_values('Score', ascending=False)
    print(f"\n📈 TOP TARAMA SONUÇLARI:\n{screen_df.round(2)}")
    
    # CSV Kaydet
    screen_df.to_csv('bist_ema_tum_scanner16.csv', index=False)
    print("\n💾 'bist_ema_scanner.csv' kaydedildi! (Excel aç)")
    
    # Grafik
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'BIST EMA Kırılım Tarama ({end_date}) | {len(matches)} Aday', fontsize=16)
    
    # Tablo-like bar (Top10 Score)
    top10 = screen_df.head(10)
    sns.barplot(data=top10, x='Score', y='Ticker', ax=axes[0], palette='viridis')
    axes[0].set_title('Top 10 Score (Yüksek → Güçlü Sinyal)')
    
    # RSI vs Vol scatter
    sns.scatterplot(data=screen_df, x='RSI', y='Vol%', size='Score', hue='Score', ax=axes[1], sizes=(50, 300), palette='coolwarm')
    axes[1].set_title('RSI vs Hacim% (Boyut=Score)')
    axes[1].axhline(130, color='red', ls='--', label='Vol%130')
    axes[1].axvline(52, color='green', ls='--', label='RSI52')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Öneriler
    top3 = screen_df.head(3)['Ticker'].tolist()
    print(f"\n🔥 TOP 3 AL ADAYI: {', '.join(top3)}")
    print("🟢 Güçlü: Score>5 + Vol%>150\n⚠️ DYOR + Stop-loss EMA20 altı!")
else:
    print("❌ Hiç hisse uymadı. Koşullar katı, yarın dene!")
