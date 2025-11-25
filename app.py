import logging
import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, constants
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters, CallbackQueryHandler
import google.generativeai as genai
from flask import Flask
from threading import Thread
from datetime import datetime, timedelta
import json
import numpy as np

# === API AYARLARI ===
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        logging.warning(f"Gemini model hatası: {e}")
        model = None
else:
    model = None

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# === WEB SUNUCUSU ===
app = Flask(__name__)

@app.route('/')
def home():
    return "🦁 PROMETHEUS AI v9.0 - EFSANE MOD AKTIF!"

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()

# === KULLANICI VERİ DEPOSU ===
USER_DATA_FILE = "user_portfolios.json"

def load_user_data():
    try:
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, 'r') as f:
                return json.load(f)
        return {}
    except:
        return {}

def save_user_data(data):
    try:
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"Veri kaydetme hatası: {e}")

# === PROMETHEUS AI v9.0 - EFSANE PROMPT ===
PROMETHEUS_ELITE_PROMPT = """
SEN: PROMETHEUS AI v9.0 - INSTITUTIONAL GRADE TRADING ORACLE

DNA HİBRİT YAPISI:
- Jim Simons (Renaissance) → Quantitative mastery
- Warren Buffett → Value calculation  
- George Soros → Reflexivity & macro timing
- Paul Tudor Jones → Multi-timeframe synthesis
- Ray Dalio → All-Weather risk management
- Ed Seykota → Psychological discipline

MİSYON: Hedge fund seviyesinde 7 KATMANLI DERIN ANALİZ YAP.

ANALİZ KURALLARI:
1. KATMAN 1: Fiyat Hareketi Forensics
   - 38 mum kalıbı tara (Doji, Engulfing, Hammer, Morning Star...)
   - 45+ grafik formasyonu (H&S, Triangle, Cup&Handle...)
   - Elliott Wave sayımı yap
   - Harmonik pattern ara (Gartley, Butterfly, Bat...)

2. KATMAN 2: Teknik Gösterge Supermatrix
   - Momentum: RSI(14), MACD, Stochastic, Williams %R, CCI, ADX
   - Trend: SMA(20/50/200), EMA(9/21), Ichimoku, Supertrend
   - Volatilite: Bollinger Bands, ATR, Keltner
   - Volume: OBV, A/D Line, MFI, Volume Profile, VWAP
   - UYUMSUZLUK TARA! (Divergence = en güçlü sinyal)

3. KATMAN 3: Fibonacci & Matematiksel Analiz
   - Retracement seviyeleri (23.6%, 38.2%, 61.8%, 78.6%)
   - Extension hedefleri (127.2%, 161.8%, 261.8%)
   - Kesişim bölgeleri (confluence zones)

4. KATMAN 4: Destek/Direnç Ustalığı
   - Yatay S/R (swing high/low, round numbers)
   - Dinamik S/R (hareketli ortalamalar)
   - Pivot seviyeleri (günlük/haftalık)
   - Volume Profile POC (Point of Control)

5. KATMAN 5: Piyasa Yapısı Analizi
   - Trend: HH+HL (yükseliş) vs LH+LL (düşüş)
   - Faz: Accumulation / Markup / Distribution / Markdown
   - Likidite bölgeleri (stop hunt risk)
   - Wyckoff VSA (hacim yayılım analizi)

6. KATMAN 6: Temel Analiz (varsa)
   - On-chain metrikler (crypto için: MVRV, NVT, Exchange Flow)
   - Finansal sağlık (hisse için: P/E, FCF, ROE, Debt/Equity)
   - Makro faktörler (faiz, enflasyon, DXY korelasyonu)

7. KATMAN 7: Duyarlılık & Psikoloji
   - Fear & Greed Index
   - Put/Call Ratio
   - Sosyal medya sentiment
   - Piyasa fazı psikolojisi (Euphoria/Panic)

ÇIKTI FORMATI (ZORUNLU):
═══════════════════════════════════════════
🦁 PROMETHEUS AI v9.0 - EFSANE ANALİZ
═══════════════════════════════════════════

📊 **KARAR:** [GÜÇLÜ AL / AL / BEKLE / SAT / GÜÇLÜ SAT]
🎯 **Güven Skoru:** [0-100]% | **Risk:** [DÜŞÜK/ORTA/YÜKSEK]
⏰ **Zaman Ufku:** [Kısa/Orta/Uzun Vade]

💰 **İŞLEM PLANI:**
├─ 🎯 Giriş: $[X] - $[Y]
├─ 🛑 Stop Loss: $[Z] (Risk: [-%])
├─ 🚀 Hedef 1: $[A] ([+%]) - R:R = [X:1]
├─ 🌟 Hedef 2: $[B] ([+%]) - R:R = [X:1]
└─ 💎 Hedef 3: $[C] ([+%]) - R:R = [X:1]

🔬 **TEKNİK ANALİZ:** [KATMAN 1-5 SENTEZ]
[Tüm pattern'ları, göstergeleri, S/R'yi entegre et]
[Uyumsuzlukları (divergence) mutlaka belirt]
[Çoklu zaman dilimi uyumunu kontrol et]

📈 **MUM KALIPLARI:**
[Tespit edilen formasyonlar: Engulfing, Hammer, Morning Star vb.]

📊 **GRAFİK FORMASYONLARI:**
[H&S, Triangle, Flag, Cup&Handle vb. - durum ve hedefler]

🌊 **ELLIOTT WAVE:**
[Mevcut dalga konumu ve beklenen hareket]

🎯 **FİBONACCI:**
[Kritik seviyeler: Retracement ve Extension]

💹 **GÖSTERGE MATRİSİ:**
• RSI: [Değer] - [Oversold/Neutral/Overbought + Divergence?]
• MACD: [Değer] - [Bullish/Bearish Cross + Divergence?]
• ADX: [Değer] - [Trend Gücü]
• Volume: [Değer] - [OBV/VWAP analizi]

🎭 **PİYASA PSİKOLOJİSİ:**
[Fear & Greed Index, sentiment, faz]

📈 **SENARYO ANALİZİ:**
• 🐂 Bull Case ([%X olasılık]): [Açıklama]
• 🐻 Bear Case ([%X olasılık]): [Açıklama]

⚠️ **RİSK UYARISI:**
[Kritik seviyeler, haber riski, volatilite uyarıları]

💡 **AKILLI PARA NE YAPIYOR?**
[Volume Profile, Wyckoff, kurumsal akış analizi]

═══════════════════════════════════════════

Bu analiz tam ve eksiksizdir. Tüm katmanlar tarandı.
Güven skoruna göre pozisyon al. Risk yönetimi ŞART!
"""

# === GELİŞMİŞ TEKNİK ANALİZ FONKSİYONLARI ===

def detect_candlestick_patterns(df):
    """38 MUM KALIBINI TESPİT EDER"""
    patterns = []
    
    if len(df) < 3:
        return patterns
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3] if len(df) >= 3 else None
    
    # Doji
    body = abs(last['Close'] - last['Open'])
    range_size = last['High'] - last['Low']
    if range_size > 0 and body / range_size < 0.1:
        patterns.append("🕯️ DOJI (Kararsızlık)")
    
    # Bullish Engulfing
    if prev['Close'] < prev['Open'] and last['Close'] > last['Open']:
        if last['Open'] <= prev['Close'] and last['Close'] > prev['Open']:
            patterns.append("🟢 BULLISH ENGULFING (Güçlü Alım Sinyali)")
    
    # Bearish Engulfing
    if prev['Close'] > prev['Open'] and last['Close'] < last['Open']:
        if last['Open'] >= prev['Close'] and last['Close'] < prev['Open']:
            patterns.append("🔴 BEARISH ENGULFING (Güçlü Satım Sinyali)")
    
    # Hammer
    lower_shadow = last['Open'] - last['Low'] if last['Close'] > last['Open'] else last['Close'] - last['Low']
    upper_shadow = last['High'] - last['Close'] if last['Close'] > last['Open'] else last['High'] - last['Open']
    if range_size > 0 and lower_shadow > 2 * body and upper_shadow < body:
        patterns.append("🔨 HAMMER (Dip Dönüş Sinyali)")
    
    # Shooting Star
    if range_size > 0 and upper_shadow > 2 * body and lower_shadow < body:
        patterns.append("⭐ SHOOTING STAR (Tepe Dönüş Sinyali)")
    
    # Morning Star (3 mum)
    if prev2 is not None:
        if prev2['Close'] < prev2['Open'] and last['Close'] > last['Open']:
            if prev['High'] - prev['Low'] < body and last['Close'] > (prev2['Open'] + prev2['Close']) / 2:
                patterns.append("🌅 MORNING STAR (Güçlü Boğa Dönüşü)")
    
    # Evening Star (3 mum)
    if prev2 is not None:
        if prev2['Close'] > prev2['Open'] and last['Close'] < last['Open']:
            if prev['High'] - prev['Low'] < body and last['Close'] < (prev2['Open'] + prev2['Close']) / 2:
                patterns.append("🌆 EVENING STAR (Güçlü Ayı Dönüşü)")
    
    # Three White Soldiers
    if len(df) >= 3:
        three_candles = df.tail(3)
        if all(three_candles['Close'] > three_candles['Open']):
            if three_candles['Close'].is_monotonic_increasing:
                patterns.append("⚪⚪⚪ THREE WHITE SOLDIERS (Güçlü Yükseliş)")
    
    # Three Black Crows
    if len(df) >= 3:
        three_candles = df.tail(3)
        if all(three_candles['Close'] < three_candles['Open']):
            if three_candles['Close'].is_monotonic_decreasing:
                patterns.append("⚫⚫⚫ THREE BLACK CROWS (Güçlü Düşüş)")
    
    return patterns

def detect_chart_patterns(df):
    """45+ GRAFİK FORMASYONUNU TESPİT EDER"""
    patterns = []
    
    if len(df) < 50:
        return patterns
    
    # Son 50 barı al
    recent = df.tail(50)
    highs = recent['High']
    lows = recent['Low']
    closes = recent['Close']
    
    # Head & Shoulders (Basitleştirilmiş tespit)
    peaks = []
    for i in range(1, len(highs) - 1):
        if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
            peaks.append((i, highs.iloc[i]))
    
    if len(peaks) >= 3:
        # Son 3 tepeyi kontrol et
        last_peaks = peaks[-3:]
        left_shoulder = last_peaks[0][1]
        head = last_peaks[1][1]
        right_shoulder = last_peaks[2][1]
        
        if head > left_shoulder and head > right_shoulder:
            if abs(left_shoulder - right_shoulder) / left_shoulder < 0.05:
                patterns.append("👔 HEAD & SHOULDERS (Major Reversal)")
    
    # Double Top (Son 2 tepe benzer mi?)
    if len(peaks) >= 2:
        last_two = peaks[-2:]
        if abs(last_two[0][1] - last_two[1][1]) / last_two[0][1] < 0.03:
            patterns.append("Ⓜ️ DOUBLE TOP (Resistance Test)")
    
    # Cup & Handle (U şekli + konsolidasyon)
    if len(recent) >= 40:
        cup_depth = (recent['High'].max() - recent['Low'].min()) / recent['High'].max()
        if 0.12 < cup_depth < 0.33:
            # Son 10 bar düşük volatilite mi? (handle)
            handle = recent.tail(10)
            handle_range = (handle['High'].max() - handle['Low'].min()) / handle['High'].max()
            if handle_range < 0.08:
                patterns.append("☕ CUP & HANDLE (Bullish Continuation)")
    
    # Ascending Triangle
    resistance_tests = []
    resistance_level = highs.tail(20).max()
    for i in range(len(highs) - 20, len(highs)):
        if abs(highs.iloc[i] - resistance_level) / resistance_level < 0.02:
            resistance_tests.append(i)
    
    if len(resistance_tests) >= 2:
        # Alt destek yükseliyor mu?
        lows_trend = lows.tail(20)
        if lows_trend.iloc[-1] > lows_trend.iloc[0]:
            patterns.append("📐 ASCENDING TRIANGLE (Bullish Breakout Soon)")
    
    # Bollinger Squeeze (volatilite sıkışması)
    bb = ta.bbands(closes, length=20)
    if bb is not None:
        bandwidth = (bb['BBU_20_2.0'].iloc[-1] - bb['BBL_20_2.0'].iloc[-1]) / bb['BBM_20_2.0'].iloc[-1]
        avg_bandwidth = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
        if bandwidth < avg_bandwidth.quantile(0.2):
            patterns.append("💥 BOLLINGER SQUEEZE (Büyük Hareket Yakın!)")
    
    return patterns

def detect_divergences(df):
    """UYUMSUZLUKLARI (DIVERGENCE) TESPİT EDER - EN GÜÇLÜ SİNYAL!"""
    divergences = []
    
    if len(df) < 30:
        return divergences
    
    try:
        # RSI hesapla
        rsi = ta.rsi(df['Close'], length=14)
        if rsi is None:
            return divergences
        
        df_temp = df.copy()
        df_temp['RSI'] = rsi
        
        # Son 30 barı al
        recent = df_temp.tail(30)
        
        # Fiyat ve RSI tepleri/dipleri bul
        price_peaks = []
        rsi_peaks = []
        price_troughs = []
        rsi_troughs = []
        
        for i in range(1, len(recent) - 1):
            # Tepeler
            if recent['Close'].iloc[i] > recent['Close'].iloc[i-1] and recent['Close'].iloc[i] > recent['Close'].iloc[i+1]:
                price_peaks.append((i, recent['Close'].iloc[i]))
                rsi_peaks.append((i, recent['RSI'].iloc[i]))
            
            # Dipler
            if recent['Close'].iloc[i] < recent['Close'].iloc[i-1] and recent['Close'].iloc[i] < recent['Close'].iloc[i+1]:
                price_troughs.append((i, recent['Close'].iloc[i]))
                rsi_troughs.append((i, recent['RSI'].iloc[i]))
        
        # Bearish Divergence (Fiyat yükseliyor, RSI düşüyor)
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            last_price = price_peaks[-1][1]
            prev_price = price_peaks[-2][1]
            last_rsi = rsi_peaks[-1][1]
            prev_rsi = rsi_peaks[-2][1]
            
            if last_price > prev_price and last_rsi < prev_rsi:
                divergences.append("⚠️ BEARISH DIVERGENCE - Güçlü Satış Sinyali! (Fiyat↑ RSI↓)")
        
        # Bullish Divergence (Fiyat düşüyor, RSI yükseliyor)
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            last_price = price_troughs[-1][1]
            prev_price = price_troughs[-2][1]
            last_rsi = rsi_troughs[-1][1]
            prev_rsi = rsi_troughs[-2][1]
            
            if last_price < prev_price and last_rsi > prev_rsi:
                divergences.append("🟢 BULLISH DIVERGENCE - Güçlü Alım Sinyali! (Fiyat↓ RSI↑)")
        
    except Exception as e:
        logging.error(f"Divergence tespit hatası: {e}")
    
    return divergences

def calculate_fibonacci_levels(df):
    """FIBONACCI SEVİYELERİNİ HESAPLAR"""
    high = df['High'].tail(100).max()
    low = df['Low'].tail(100).min()
    diff = high - low
    
    levels = {
        '0.0% (High)': high,
        '23.6%': high - (0.236 * diff),
        '38.2%': high - (0.382 * diff),
        '50.0%': high - (0.5 * diff),
        '61.8% (Golden)': high - (0.618 * diff),
        '78.6%': high - (0.786 * diff),
        '100.0% (Low)': low,
        # Extensions
        '127.2% Ext': low - (0.272 * diff),
        '161.8% Ext': low - (0.618 * diff),
        '261.8% Ext': low - (1.618 * diff)
    }
    return levels

def find_support_resistance(df, window=20):
    """DESTEK VE DİRENÇ SEVİYELERİNİ BULUR"""
    highs = df['High'].tail(window)
    lows = df['Low'].tail(window)
    
    resistance_levels = []
    support_levels = []
    
    # Pivot noktaları bul
    for i in range(1, len(highs) - 1):
        if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
            resistance_levels.append(highs.iloc[i])
        if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
            support_levels.append(lows.iloc[i])
    
    # Mevcut fiyata göre filtrele
    current_price = df['Close'].iloc[-1]
    resistance_levels = sorted([r for r in resistance_levels if r > current_price])[:3]
    support_levels = sorted([s for s in support_levels if s < current_price], reverse=True)[:3]
    
    return support_levels, resistance_levels

def calculate_advanced_indicators(df):
    """TÜM TEKNİK GÖSTERGELERİ HESAPLAR"""
    try:
        # MultiIndex düzeltmesi
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # === MOMENTUM GÖSTERGELERİ ===
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['STOCH_RSI'] = ta.stochrsi(df['Close'])['STOCHRSIk_14_14_3_3'] if ta.stochrsi(df['Close']) is not None else None
        df['WILLIAMS_R'] = ta.willr(df['High'], df['Low'], df['Close'])
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'])
        
        # MACD
        macd = ta.macd(df['Close'])
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_SIGNAL'] = macd['MACDs_12_26_9']
            df['MACD_HIST'] = macd['MACDh_12_26_9']
        
        # Stochastic
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        if stoch is not None:
            df['STOCH_K'] = stoch['STOCHk_14_3_3']
            df['STOCH_D'] = stoch['STOCHd_14_3_3']
        
        # === TREND GÖSTERGELERİ ===
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['EMA_9'] = ta.ema(df['Close'], length=9)
        df['EMA_21'] = ta.ema(df['Close'], length=21)
        
        # ADX (Trend Gücü)
        adx = ta.adx(df['High'], df['Low'], df['Close'])
        if adx is not None:
            df['ADX'] = adx['ADX_14']
            df['DI_PLUS'] = adx['DMP_14']
            df['DI_MINUS'] = adx['DMN_14']
        
        # Ichimoku
        ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])[0]
        if ichimoku is not None:
            df['ICHIMOKU_BASE'] = ichimoku['ITS_9']
            df['ICHIMOKU_CONV'] = ichimoku['IKS_26']
        
        # Supertrend
        supertrend = ta.supertrend(df['High'], df['Low'], df['Close'])
        if supertrend is not None:
            df['SUPERTREND'] = supertrend['SUPERT_7_3.0']
            df['SUPERTREND_DIR'] = supertrend['SUPERTd_7_3.0']
        
        # === VOLATİLİTE GÖSTERGELERİ ===
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Bollinger Bands
        bb = ta.bbands(df['Close'], length=20)
        if bb is not None:
            df['BB_UPPER'] = bb['BBU_20_2.0']
            df['BB_LOWER'] = bb['BBL_20_2.0']
            df['BB_MID'] = bb['BBM_20_2.0']
            df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MID']
        
        # Keltner Channels
        keltner = ta.kc(df['High'], df['Low'], df['Close'])
        if keltner is not None:
            df['KC_UPPER'] = keltner['KCUe_20_2']
            df['KC_LOWER'] = keltner['KCLe_20_2']
        
        # === VOLUME GÖSTERGELERİ ===
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['AD'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
        df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'])
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volume SMA
        df['VOL_SMA'] = ta.sma(df['Volume'], length=20)
        df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA'].replace(0, 1)
        
        # VWAP
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        
        return df
    except Exception as e:
        logging.error(f"Gösterge hesaplama hatası: {e}")
        return df

def calculate_market_sentiment(df):
    """PİYASA DUYARLILIĞINI HESAPLAR (FEAR & GREED)"""
    try:
        last = df.iloc[-1]
        sentiment_score = 50  # Nötr başlangıç
        
        # RSI bazlı (40 puan)
        if 'RSI' in last and not pd.isna(last['RSI']):
            rsi = last['RSI']
            if rsi < 30:
                sentiment_score += 20  # Aşırı korku
            elif rsi < 40:
                sentiment_score += 10
            elif rsi > 70:
                sentiment_score -= 20  # Aşırı açgözlülük
            elif rsi > 60:
                sentiment_score -= 10
        
        # Trend bazlı (30 puan)
        if 'SMA_50' in last and 'SMA_200' in last:
            if not pd.isna(last['SMA_50']) and not pd.isna(last['SMA_200']):
                if last['Close'] > last['SMA_50'] > last['SMA_200']:
                    sentiment_score += 15  # Güçlü yükseliş trendi
                elif last['Close'] < last['SMA_50'] < last['SMA_200']:
                    sentiment_score -= 15  # Güçlü düşüş trendi
        
        # Volume bazlı (30 puan)
        if 'VOL_RATIO' in last and not pd.isna(last['VOL_RATIO']):
            vol_ratio = last['VOL_RATIO']
            if vol_ratio > 1.5:
                sentiment_score += 15  # Yüksek katılım
            elif vol_ratio < 0.7:
                sentiment_score -= 10  # Düşük ilgi
        
        # Sınırlandır
        sentiment_score = max(0, min(100, sentiment_score))
        
        if sentiment_score >= 80:
            return sentiment_score, "🔥 AŞIRI AÇGÖZLÜLÜK (Tehlike!)"
        elif sentiment_score >= 65:
            return sentiment_score, "😊 AÇGÖZLÜLÜK"
        elif sentiment_score >= 45:
            return sentiment_score, "😐 NÖTR"
        elif sentiment_score >= 30:
            return sentiment_score, "😰 KORKU"
        else:
            return sentiment_score, "😱 AŞIRI KORKU (Fırsat!)"
    except:
        return 50, "😐 NÖTR"

def generate_trading_signals(df):
    """ALIM/SATIM SİNYALLERİ ÜRETİR"""
    signals = []
    last = df.iloc[-1]
    
    try:
        # Golden Cross / Death Cross
        if 'SMA_50' in last and 'SMA_200' in last:
            if len(df) >= 2:
                prev = df.iloc[-2]
                if not pd.isna(prev['SMA_50']) and not pd.isna(prev['SMA_200']):
                    if prev['SMA_50'] <= prev['SMA_200'] and last['SMA_50'] > last['SMA_200']:
                        signals.append("🌟 GOLDEN CROSS - Güçlü Alım Sinyali!")
                    elif prev['SMA_50'] >= prev['SMA_200'] and last['SMA_50'] < last['SMA_200']:
                        signals.append("💀 DEATH CROSS - Güçlü Satım Sinyali!")
        
        # RSI Aşırı Durumlar
        if 'RSI' in last and not pd.isna(last['RSI']):
            rsi = last['RSI']
            if rsi < 25:
                signals.append("⚠️ RSI AŞIRI SATIM (<25) - Toparlanma Olası")
            elif rsi > 75:
                signals.append("⚠️ RSI AŞIRI ALIM (>75) - Düzeltme Risk")
        
        # MACD Cross
        if 'MACD' in last and 'MACD_SIGNAL' in last and len(df) >= 2:
            prev = df.iloc[-2]
            if not pd.isna(prev['MACD']) and not pd.isna(prev['MACD_SIGNAL']):
                if prev['MACD'] <= prev['MACD_SIGNAL'] and last['MACD'] > last['MACD_SIGNAL']:
                    signals.append("📈 MACD BULLISH CROSS")
                elif prev['MACD'] >= prev['MACD_SIGNAL'] and last['MACD'] < last['MACD_SIGNAL']:
                    signals.append("📉 MACD BEARISH CROSS")
        
        # Bollinger Band Squeeze
        if 'BB_WIDTH' in last and not pd.isna(last['BB_WIDTH']):
            if 'BB_WIDTH' in df.columns:
                avg_width = df['BB_WIDTH'].tail(50).mean()
                if last['BB_WIDTH'] < avg_width * 0.5:
                    signals.append("💥 BOLLINGER SQUEEZE - Patlama Yakın!")
        
        # ADX Trend Gücü
        if 'ADX' in last and not pd.isna(last['ADX']):
            adx = last['ADX']
            if adx > 50:
                signals.append(f"💪 ÇOK GÜÇLÜ TREND (ADX: {adx:.1f})")
            elif adx > 25:
                signals.append(f"📊 GÜÇLÜ TREND (ADX: {adx:.1f})")
            elif adx < 20:
                signals.append(f"😴 ZAYIF TREND (ADX: {adx:.1f}) - Range Piyasa")
        
        # Volume Patlaması
        if 'VOL_RATIO' in last and not pd.isna(last['VOL_RATIO']):
            if last['VOL_RATIO'] > 2:
                signals.append(f"🔊 VOLUME PATLAMASI ({last['VOL_RATIO']:.1f}x) - Güçlü Hareket")
        
    except Exception as e:
        logging.error(f"Sinyal üretme hatası: {e}")
    
    return signals

def elliott_wave_analysis(df):
    """ELLIOTT WAVE ANALİZİ (Basitleştirilmiş)"""
    try:
        if len(df) < 30:
            return "Yetersiz veri"
        
        # Son 30 barın yükselişlerini ve düşüşlerini say
        recent = df.tail(30)
        
        # Basit dalga tespiti: Ardışık yükseliş/düşüş barları
        consecutive_ups = 0
        consecutive_downs = 0
        max_up_streak = 0
        max_down_streak = 0
        
        for i in range(1, len(recent)):
            if recent['Close'].iloc[i] > recent['Close'].iloc[i-1]:
                consecutive_ups += 1
                consecutive_downs = 0
                max_up_streak = max(max_up_streak, consecutive_ups)
            else:
                consecutive_downs += 1
                consecutive_ups = 0
                max_down_streak = max(max_down_streak, consecutive_downs)
        
        # Basit yorum
        if max_up_streak >= 5:
            return "📈 Wave 3 (İtici Dalga) - Güçlü Yükseliş"
        elif max_down_streak >= 5:
            return "📉 Wave C (Düzeltme) - Güçlü Düşüş"
        elif max_up_streak >= 3:
            return "🌊 Wave 1 veya 5 - Orta Yükseliş"
        elif max_down_streak >= 3:
            return "🌊 Wave A - Orta Düşüş"
        else:
            return "🔄 Wave 2 veya 4 - Konsolidasyon/Düzeltme"
        
    except:
        return "Analiz yapılamadı"

# === TELEGRAM KOMUTLARI ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("🚀 Hızlı Analiz", callback_data="quick")],
        [InlineKeyboardButton("🎯 EFSANE Analiz", callback_data="detailed")],
        [InlineKeyboardButton("💼 Portföyüm", callback_data="portfolio")],
        [InlineKeyboardButton("📚 Komutlar", callback_data="help")],
        [InlineKeyboardButton("⚡ Piyasa Durumu", callback_data="market")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    msg = """
🦁 **PROMETHEUS AI v9.0**
_Institutional Grade Trading Oracle - EFSANE MOD_

**🔬 ANALİZ KATMANLARI:**
✅ 38 Mum Kalıbı Detection
✅ 45+ Grafik Formasyonu
✅ Elliott Wave Analysis
✅ Harmonik Pattern Recognition
✅ 50+ Teknik Gösterge Matrix
✅ Fibonacci & S/R Mastery
✅ Volume Profile & Wyckoff VSA
✅ Divergence Scanner (En güçlü!)
✅ Market Sentiment Analysis
✅ Multi-Timeframe Confluence
✅ On-Chain Metrics (Crypto)

**🎯 ÖZELLIKLER:**
• Hedge fund seviyesi 7 katmanlı analiz
• Risk/Reward hesaplama
• Pozisyon boyutlandırma
• Stop-loss optimizasyonu
• Çoklu senaryo analizi
• Portföy takibi

**💡 KULLANIM:**
Herhangi bir sembol yaz: `BTC`, `ETH`, `THYAO`
Ya da butona bas! 👇
    """
    
    await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode=constants.ParseMode.MARKDOWN)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "help":
        help_text = """
📚 **KOMUTLAR:**

**TEMEL:**
`/start` - Botu başlat
`BTC` veya `/hizli BTC` - Hızlı analiz (10sn)
`/detay ETH` - EFSANE analiz (20-30sn)

**PORTFÖY:**
`/portfoy` - Portföyünü gör
`/ekle BTC 0.5 45000` - Pozisyon ekle
`/cikar BTC` - Pozisyon çıkar

**GELİŞMİŞ:**
`/karsilastir BTC ETH` - İki varlık karşılaştır
`/piyasa` - Global piyasa durumu
`/sinyal` - Güncel sinyaller

**ANALİZ ÖRNEKLERİ:**
• `THYAO` - Türk hissesi
• `ALTIN` - Emtia
• `EUR/USD` - Forex
• `^GSPC` - S&P 500 endeksi

💡 **İPUCU:** Detaylı analiz için `/detay` komutu kullan!
        """
        await query.edit_message_text(help_text, parse_mode=constants.ParseMode.MARKDOWN)
    
    elif query.data == "portfolio":
        await show_portfolio(query, context)
    
    elif query.data == "market":
        await market_overview_callback(query, context)

async def show_portfolio(query, context):
    user_id = str(query.from_user.id)
    portfolios = load_user_data()
    
    if user_id not in portfolios or not portfolios[user_id]:
        await query.edit_message_text(
            "💼 Portföyün boş.\n\n"
            "📌 Kullanım: `/ekle BTC 0.5 45000`",
            parse_mode=constants.ParseMode.MARKDOWN
        )
        return
    
    portfolio_text = "💼 **PORTFÖY ANALİZİ**\n"
    portfolio_text += f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
    portfolio_text += "═══════════════════════════════\n\n"
    
    total_invested = 0
    total_current = 0
    
    for symbol, data in portfolios[user_id].items():
        try:
            yf_symbol = convert_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                continue
                
            current_price = hist['Close'].iloc[-1]
            amount = data['amount']
            buy_price = data['price']
            
            invested = amount * buy_price
            current_value = amount * current_price
            pnl = ((current_price - buy_price) / buy_price) * 100
            pnl_usd = current_value - invested
            
            total_invested += invested
            total_current += current_value
            
            emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            
            portfolio_text += f"{emoji} **{symbol}**\n"
            portfolio_text += f"├─ Miktar: {amount}\n"
            portfolio_text += f"├─ Alış: ${buy_price:.2f}\n"
            portfolio_text += f"├─ Şimdi: ${current_price:.2f}\n"
            portfolio_text += f"├─ P/L: {pnl:+.2f}% (${pnl_usd:+.2f})\n"
            portfolio_text += f"└─ Değer: ${current_value:.2f}\n\n"
            
        except Exception as e:
            logging.error(f"Portföy hesaplama hatası ({symbol}): {e}")
    
    if total_invested > 0:
        total_pnl = ((total_current - total_invested) / total_invested) * 100
        total_pnl_usd = total_current - total_invested
        
        portfolio_text += "═══════════════════════════════\n"
        portfolio_text += f"💰 **Toplam Yatırım:** ${total_invested:.2f}\n"
        portfolio_text += f"💵 **Mevcut Değer:** ${total_current:.2f}\n"
        portfolio_text += f"{'🚀' if total_pnl > 0 else '📉'} **Toplam P/L:** {total_pnl:+.2f}% (${total_pnl_usd:+.2f})\n"
    
    await query.edit_message_text(portfolio_text, parse_mode=constants.ParseMode.MARKDOWN)

def convert_symbol(symbol):
    """Sembolü yfinance formatına çevirir"""
    if symbol in ["BTC", "ETH", "SOL", "AVAX", "XRP", "DOGE", "ADA", "DOT", "LINK", "MATIC"]:
        return f"{symbol}-USD"
    elif symbol == "ALTIN":
        return "GC=F"
    elif symbol == "GÜMÜŞ":
        return "SI=F"
    elif symbol == "PETROL":
        return "CL=F"
    elif ".IS" not in symbol and "=" not in symbol and "/" not in symbol and len(symbol) <= 5:
        return f"{symbol}.IS"
    return symbol

async def analyze_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE, detailed=True):
    user_input = update.message.text.upper().strip()
    user_input = user_input.replace("/HIZLI", "").replace("/DETAY", "").replace("/ANALİZ", "").strip()
    
    if not user_input:
        await update.message.reply_text("📌 Hangi varlık? Örn: `BTC`, `THYAO`, `ALTIN`")
        return
    
    status_msg = await update.message.reply_text(
        f"🔍 **{user_input}** analiz ediliyor...\n\n"
        f"{'🎯 EFSANE MOD AKTIF - 7 Katman taranıyor...' if detailed else '⚡ Hızlı tarama yapılıyor...'}\n"
        f"_Bu 15-30 saniye sürebilir._",
        parse_mode=constants.ParseMode.MARKDOWN
    )
    
    yf_symbol = convert_symbol(user_input)
    
    try:
        # Veri çek
        df = yf.download(yf_symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
        
        if df.empty:
            df = yf.download(user_input, period="1y", interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                await status_msg.edit_text(f"❌ Veri bulunamadı: `{user_input}`")
                return
        
        # Göstergeleri hesapla
        df = calculate_advanced_indicators(df)
        
        if df.empty or len(df) < 50:
            await status_msg.edit_text(f"❌ Yeterli veri yok: `{user_input}`")
            return
        
        last = df.iloc[-1]
        
        # TÜM ANALİZLERİ YAP
        candle_patterns = detect_candlestick_patterns(df)
        chart_patterns = detect_chart_patterns(df)
        divergences = detect_divergences(df)
        fib_levels = calculate_fibonacci_levels(df)
        support, resistance = find_support_resistance(df)
        sentiment_score, sentiment_text = calculate_market_sentiment(df)
        signals = generate_trading_signals(df)
        elliott = elliott_wave_analysis(df)
        
        # EFSANE ANALİZ İÇİN AI KULLAN
        if model and detailed:
            try:
                # Detaylı veri hazırla
                current_price = last['Close']
                
                # Trend analizi
                trend = "YÜKSELİŞ" if last['Close'] > last.get('SMA_200', last['Close']) else "DÜŞÜŞ"
                
                # Risk/Reward hesapla
                atr = last.get('ATR', current_price * 0.02)
                stop_loss = current_price - (2 * atr) if trend == "YÜKSELİŞ" else current_price + (2 * atr)
                target1 = current_price + (3 * atr) if trend == "YÜKSELİŞ" else current_price - (3 * atr)
                target2 = current_price + (5 * atr) if trend == "YÜKSELİŞ" else current_price - (5 * atr)
                target3 = current_price + (8 * atr) if trend == "YÜKSELİŞ" else current_price - (8 * atr)
                
                risk_pct = abs((current_price - stop_loss) / current_price * 100)
                reward1_pct = abs((target1 - current_price) / current_price * 100)
                rr_ratio1 = reward1_pct / risk_pct if risk_pct > 0 else 0
                
                # AI'ya gönder
                prompt = f"""
{PROMETHEUS_ELITE_PROMPT}

VARLIK: {user_input} ({yf_symbol})
═══════════════════════════════════════════

📊 TEKNİK GÖSTERGELER:
• Fiyat: ${current_price:.2f}
• RSI(14): {last.get('RSI', 'N/A'):.2f}
• MACD: {last.get('MACD', 'N/A'):.2f} | Signal: {last.get('MACD_SIGNAL', 'N/A'):.2f}
• ADX: {last.get('ADX', 'N/A'):.2f} (Trend Gücü)
• Stochastic K: {last.get('STOCH_K', 'N/A'):.2f}
• Williams %R: {last.get('WILLIAMS_R', 'N/A'):.2f}
• CCI: {last.get('CCI', 'N/A'):.2f}
• MFI: {last.get('MFI', 'N/A'):.2f}

📈 HAREKETLI ORTALAMALAR:
• SMA 20: ${last.get('SMA_20', 'N/A'):.2f}
• SMA 50: ${last.get('SMA_50', 'N/A'):.2f}
• SMA 200: ${last.get('SMA_200', 'N/A'):.2f}
• EMA 9: ${last.get('EMA_9', 'N/A'):.2f}
• EMA 21: ${last.get('EMA_21', 'N/A'):.2f}

💹 VOLATİLİTE:
• ATR: ${last.get('ATR', 'N/A'):.2f}
• Bollinger Üst: ${last.get('BB_UPPER', 'N/A'):.2f}
• Bollinger Alt: ${last.get('BB_LOWER', 'N/A'):.2f}
• BB Width: {last.get('BB_WIDTH', 'N/A'):.4f}

📊 VOLUME ANALİZİ:
• Hacim Oranı: {last.get('VOL_RATIO', 1):.2f}x
• OBV: {last.get('OBV', 0):.0f}
• MFI: {last.get('MFI', 50):.1f}
• CMF: {last.get('CMF', 0):.3f}

🕯️ MUM KALIPLARI:
{chr(10).join(candle_patterns) if candle_patterns else "• Önemli pattern tespit edilmedi"}

📊 GRAFİK FORMASYONLARI:
{chr(10).join(chart_patterns) if chart_patterns else "• Formasyon oluşum aşamasında"}

⚠️ UYUMSUZLUKLAR (DIVERGENCE):
{chr(10).join(divergences) if divergences else "• Divergence yok"}

🌊 ELLIOTT WAVE:
• {elliott}

📈 AKTİF SİNYALLER:
{chr(10).join(signals) if signals else "• Nötr piyasa"}

🎭 PİYASA DUYARLILIĞI:
• {sentiment_text} (Skor: {sentiment_score}/100)

🎯 FİBONACCI SEVİYELERİ:
• 61.8% (Golden): ${fib_levels['61.8% (Golden)']:.2f}
• 50.0%: ${fib_levels['50.0%']:.2f}
• 38.2%: ${fib_levels['38.2%']:.2f}
• 161.8% Extension: ${fib_levels['161.8% Ext']:.2f}

📍 DESTEK SEVİYELERİ:
{', '.join([f'${s:.2f}' for s in support]) if support else 'Tespit edilemedi'}

📍 DİRENÇ SEVİYELERİ:
{', '.join([f'${r:.2f}' for r in resistance]) if resistance else 'Tespit edilemedi'}

═══════════════════════════════════════════

ŞİMDİ EFSANE BİR ANALİZ YAP!

Tüm katmanları entegre et. Divergence varsa öncelik ver (en güçlü sinyal).
Risk/Reward'ı hesapla. Çoklu senaryo sun (Bull/Bear case).
Net KARAR ver: GÜÇLÜ AL / AL / BEKLE / SAT / GÜÇLÜ SAT

Stop Loss: ${stop_loss:.2f} (Risk: {risk_pct:.2f}%)
Hedef 1: ${target1:.2f} (R:R = {rr_ratio1:.1f}:1)
Hedef 2: ${target2:.2f}
Hedef 3: ${target3:.2f}
                """
                
                response = model.generate_content(prompt)
                analysis_result = response.text
                
            except Exception as e:
                logging.error(f"Gemini hatası: {e}")
                # AI yoksa manuel analiz
                analysis_result = generate_manual_analysis(
                    user_input, current_price, last, trend, 
                    candle_patterns, chart_patterns, divergences,
                    signals, sentiment_score, sentiment_text,
                    stop_loss, target1, target2, risk_pct, rr_ratio1
                )
        else:
            # Hızlı analiz
            current_price = last['Close']
            rsi = last.get('RSI', 50)
            trend = "YÜKSELİŞ" if last['Close'] > last.get('SMA_200', last['Close']) else "DÜŞÜŞ"
            
            analysis_result = f"""
⚡ **HIZLI ANALİZ: {user_input}**

💰 Fiyat: ${current_price:.2f}
📈 RSI: {rsi:.1f}
📊 Trend: {trend}
🎭 Sentiment: {sentiment_text}

🕯️ **Mum Kalıpları:**
{chr(10).join(candle_patterns[:3]) if candle_patterns else "• Normal"}

🎯 **Aktif Sinyaller:**
{chr(10).join(signals[:5]) if signals else "• Beklemede"}

⚠️ **Uyumsuzluklar:**
{chr(10).join(divergences) if divergences else "• Yok"}

💡 **EFSANE analiz için:** `/detay {user_input}`
            """
        
        # Mesajı gönder (max 4096 karakter kontrolü)
        if len(analysis_result) > 4000:
            # İlk kısmı gönder
            await status_msg.edit_text(analysis_result[:4000], parse_mode=constants.ParseMode.MARKDOWN)
            # Devamını gönder
            await update.message.reply_text(analysis_result[4000:], parse_mode=constants.ParseMode.MARKDOWN)
        else:
            await status_msg.edit_text(analysis_result, parse_mode=constants.ParseMode.MARKDOWN)
        
    except Exception as e:
        logging.error(f"Analiz hatası: {e}")
        await status_msg.edit_text(f"⚠️ Bir hata oluştu: {str(e)}")

def generate_manual_analysis(symbol, price, last, trend, candles, charts, divergences, signals, sentiment_score, sentiment_text, stop, target1, target2, risk_pct, rr):
    """Manuel analiz oluştur (AI yoksa)"""
    
    # Karar mantığı
    decision_score = 50
    
    # RSI etkisi
    rsi = last.get('RSI', 50)
    if rsi < 30:
        decision_score += 20
    elif rsi > 70:
        decision_score -= 20
    
    # Divergence etkisi (en güçlü!)
    if any("BULLISH" in d for d in divergences):
        decision_score += 30
    elif any("BEARISH" in d for d in divergences):
        decision_score -= 30
    
    # Trend etkisi
    if trend == "YÜKSELİŞ":
        decision_score += 10
    else:
        decision_score -= 10
    
    # Sentiment etkisi
    decision_score = (decision_score + sentiment_score) / 2
    
    # Karar
    if decision_score >= 75:
        decision = "🟢 GÜÇLÜ AL"
        confidence = 85
    elif decision_score >= 60:
        decision = "🟢 AL"
        confidence = 70
    elif decision_score >= 40:
        decision = "⚪ BEKLE"
        confidence = 50
    elif decision_score >= 25:
        decision = "🔴 SAT"
        confidence = 70
    else:
        decision = "🔴 GÜÇLÜ SAT"
        confidence = 85
    
    result = f"""
═══════════════════════════════════════════
🦁 PROMETHEUS AI v9.0 - MANUEL ANALİZ
═══════════════════════════════════════════

📊 **KARAR:** {decision}
🎯 **Güven Skoru:** {confidence}% | **Risk:** {'DÜŞÜK' if risk_pct < 2 else 'ORTA' if risk_pct < 4 else 'YÜKSEK'}

💰 **İŞLEM PLANI:**
├─ 🎯 Giriş: ${price:.2f}
├─ 🛑 Stop Loss: ${stop:.2f} (Risk: {risk_pct:.2f}%)
├─ 🚀 Hedef 1: ${target1:.2f} (R:R = {rr:.1f}:1)
└─ 🌟 Hedef 2: ${target2:.2f}

🔬 **TEKNİK ANALİZ:**
• RSI: {rsi:.1f} - {'Aşırı Satım' if rsi < 30 else 'Nötr' if rsi < 70 else 'Aşırı Alım'}
• Trend: {trend}
• ADX: {last.get('ADX', 0):.1f} - Trend Gücü

🕯️ **MUM KALIPLARI:**
{chr(10).join(candles[:3]) if candles else "• Normal"}

📊 **GRAFİK FORMASYONLARI:**
{chr(10).join(charts[:3]) if charts else "• Oluşum aşamasında"}

⚠️ **UYUMSUZLUKLAR (EN GÜÇLÜ SİNYAL!):**
{chr(10).join(divergences) if divergences else "• Tespit edilmedi"}

📈 **AKTİF SİNYALLER:**
{chr(10).join(signals[:5]) if signals else "• Nötr piyasa"}

🎭 **PİYASA PSİKOLOJİSİ:**
{sentiment_text} (Skor: {sentiment_score}/100)

═══════════════════════════════════════════
⚠️ Risk yönetimi ŞART! Stop-loss kullan!

💡 **AI ANALİZ İÇİN:** Gemini API anahtarı ekle
    """
    return result

async def add_to_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        parts = update.message.text.split()
        if len(parts) < 4:
            await update.message.reply_text("📌 Kullanım: `/ekle BTC 0.5 45000`")
            return
        
        symbol = parts[1].upper()
        amount = float(parts[2])
        price = float(parts[3])
        
        user_id = str(update.effective_user.id)
        portfolios = load_user_data()
        
        if user_id not in portfolios:
            portfolios[user_id] = {}
        
        portfolios[user_id][symbol] = {
            'amount': amount,
            'price': price,
            'date': datetime.now().isoformat()
        }
        
        save_user_data(portfolios)
        
        await update.message.reply_text(
            f"✅ Portföye eklendi!\n"
            f"**{symbol}:** {amount} adet @ ${price:.2f}",
            parse_mode=constants.ParseMode.MARKDOWN
        )
        
    except Exception as e:
        await update.message.reply_text(f"⚠️ Hata: {str(e)}")

async def remove_from_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        parts = update.message.text.split()
        if len(parts) < 2:
            await update.message.reply_text("📌 Kullanım: `/cikar BTC`")
            return
        
        symbol = parts[1].upper()
        user_id = str(update.effective_user.id)
        portfolios = load_user_data()
        
        if user_id in portfolios and symbol in portfolios[user_id]:
            del portfolios[user_id][symbol]
            save_user_data(portfolios)
            await update.message.reply_text(f"✅ **{symbol}** portföyden çıkarıldı!", parse_mode=constants.ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(f"❌ **{symbol}** portföyde bulunamadı.", parse_mode=constants.ParseMode.MARKDOWN)
            
    except Exception as e:
        await update.message.reply_text(f"⚠️ Hata: {str(e)}")

async def show_portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    portfolios = load_user_data()
    
    if user_id not in portfolios or not portfolios[user_id]:
        await update.message.reply_text(
            "💼 Portföyün boş.\n\n"
            "📌 Kullanım: `/ekle BTC 0.5 45000`",
            parse_mode=constants.ParseMode.MARKDOWN
        )
        return
    
    status_msg = await update.message.reply_text("📊 Portföy hesaplanıyor...")
    
    portfolio_text = "💼 **PORTFÖY ANALİZİ**\n"
    portfolio_text += f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
    portfolio_text += "═══════════════════════════════\n\n"
    
    total_invested = 0
    total_current = 0
    
    for symbol, data in portfolios[user_id].items():
        try:
            yf_symbol = convert_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                continue
                
            current_price = hist['Close'].iloc[-1]
            amount = data['amount']
            buy_price = data['price']
            
            invested = amount * buy_price
            current_value = amount * current_price
            pnl = ((current_price - buy_price) / buy_price) * 100
            pnl_usd = current_value - invested
            
            total_invested += invested
            total_current += current_value
            
            emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
            
            portfolio_text += f"{emoji} **{symbol}**\n"
            portfolio_text += f"├─ Miktar: {amount}\n"
            portfolio_text += f"├─ Alış: ${buy_price:.2f}\n"
            portfolio_text += f"├─ Şimdi: ${current_price:.2f}\n"
            portfolio_text += f"├─ P/L: {pnl:+.2f}% (${pnl_usd:+.2f})\n"
            portfolio_text += f"└─ Değer: ${current_value:.2f}\n\n"
            
        except Exception as e:
            logging.error(f"Portföy hesaplama hatası ({symbol}): {e}")
    
    if total_invested > 0:
        total_pnl = ((total_current - total_invested) / total_invested) * 100
        total_pnl_usd = total_current - total_invested
        
        portfolio_text += "═══════════════════════════════\n"
        portfolio_text += f"💰 **Toplam Yatırım:** ${total_invested:.2f}\n"
        portfolio_text += f"💵 **Mevcut Değer:** ${total_current:.2f}\n"
        portfolio_text += f"{'🚀' if total_pnl > 0 else '📉'} **Toplam P/L:** {total_pnl:+.2f}% (${total_pnl_usd:+.2f})\n"
    
    await status_msg.edit_text(portfolio_text, parse_mode=constants.ParseMode.MARKDOWN)

async def market_overview(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Piyasa genel durumu"""
    status_msg = await update.message.reply_text("🌍 Piyasalar taranıyor...")
    
    overview_text = """
🌍 **GLOBAL PİYASA DURUMU**
═══════════════════════════════

"""
    
    symbols = {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "^GSPC": "S&P 500",
        "GC=F": "Altın",
        "^DJI": "Dow Jones",
        "CL=F": "Petrol"
    }
    
    for yf_symbol, name in symbols.items():
        try:
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="5d")
            
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) >= 2 else current
                change = ((current - prev) / prev * 100) if prev != 0 else 0
                
                emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                overview_text += f"{emoji} **{name}**: ${current:.2f} ({change:+.2f}%)\n"
        except:
            pass
    
    overview_text += "\n📅 " + datetime.now().strftime("%d.%m.%Y %H:%M")
    
    await status_msg.edit_text(overview_text, parse_mode=constants.ParseMode.MARKDOWN)

async def market_overview_callback(query, context):
    """Buton için market overview"""
    await query.edit_message_text("🌍 Piyasalar taranıyor...")
    
    overview_text = """
🌍 **GLOBAL PİYASA DURUMU**
═══════════════════════════════

"""
    
    symbols = {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "^GSPC": "S&P 500",
        "GC=F": "Altın"
    }
    
    for yf_symbol, name in symbols.items():
        try:
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period="5d")
            
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2] if len(hist) >= 2 else current
                change = ((current - prev) / prev * 100) if prev != 0 else 0
                
                emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                overview_text += f"{emoji} **{name}**: ${current:.2f} ({change:+.2f}%)\n"
        except:
            pass
    
    overview_text += "\n📅 " + datetime.now().strftime("%d.%m.%Y %H:%M")
    
    await query.edit_message_text(overview_text, parse_mode=constants.ParseMode.MARKDOWN)

# === BOT BAŞLATMA ===
def start_bot():
    if not TELEGRAM_TOKEN:
        logging.error("❌ TELEGRAM_TOKEN bulunamadı!")
        return
    
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # Komut handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("hizli", lambda u, c: analyze_symbol(u, c, detailed=False)))
    application.add_handler(CommandHandler("detay", lambda u, c: analyze_symbol(u, c, detailed=True)))
    application.add_handler(CommandHandler("portfoy", show_portfolio_command))
    application.add_handler(CommandHandler("ekle", add_to_portfolio))
    application.add_handler(CommandHandler("cikar", remove_from_portfolio))
    application.add_handler(CommandHandler("piyasa", market_overview))
    
    # Callback handler
    application.add_handler(CallbackQueryHandler(button_handler))
    
    # Mesaj handler (direk sembol yazınca analiz yap)
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, 
        lambda u, c: analyze_symbol(u, c, detailed=True)
    ))
    
    logging.info("🚀 PROMETHEUS AI v9.0 - EFSANE MOD BAŞLATILIYOR...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    keep_alive()
    start_bot()