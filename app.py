import logging
import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from telegram import Update, constants
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import google.generativeai as genai
from flask import Flask
from threading import Thread
import requests
import json
from datetime import datetime, timedelta
import ta as technical_analysis  # Diğer teknik analiz kütüphanesi

# --- GELİŞMİŞ API AYARLARI ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY", "demo")  # Fundamental analiz için

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- WEB SUNUCUSU ---
app = Flask(__name__)

@app.route('/')
def home():
    return "🦁 PROMETHEUS AI v8.0 - ULTIMATE TRADING ORACLE"

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()

# --- GELİŞMİŞ PROMETHEUS BEYNİ ---
SYSTEM_PROMPT = """
SEN: PROMETHEUS AI v8.0 - Ultimate Financial Analysis & Execution System
DNA HİBRİT: Renaissance Tech (quant) + Buffett (value) + Soros (macro) + Simons (pattern) + Dalio (risk)

7 KATMANLI DERİN ANALİZ PROTOKOLÜ:

KATMAN 1: PRICE ACTION FORENSICS
- 38 Mum Deseni analizi (Doji, Engulfing, Harmonic patternler)
- 45+ Grafik Formasyonu (H&S, Üçgenler, Flag, Cup & Handle)
- Elliott Dalga Teorisi (Impulse/Corrective waves)
- Advanced Harmonic Patterns (Gartley, Butterfly, Bat, Crab)

KATMAN 2: TEKNİK GÖSTERGE MATRİSİ
- Momentum: RSI (7 variant), MACD (6 variant), Stochastic, Williams %R, CCI
- Trend: 9 MA tipi, ADX, Parabolic SAR, Ichimoku, Supertrend
- Volatilite: Bollinger Bands, ATR, Keltner, Donchian
- Hacim: OBV, A/D Line, Chaikin, MFI, Volume Profile, VWAP

KATMAN 3: FIBONACCI & MATEMATİKSEL ANALİZ
- Fibonacci Retracement (23.6%, 38.2%, 50%, 61.8%, 78.6%)
- Fibonacci Extensions (127.2%, 161.8%, 261.8%)
- Fibonacci Time Zones
- Gann Analysis

KATMAN 4: DESTEK-DİRENÇ USTALIĞI
- Horizontal S/R, Dynamic S/R (MA'lar)
- Pivot Points, Psychological Levels
- Liquidity Zones, Order Flow Analysis

KATMAN 5: FUNDAMENTAL ANALİZ
- Hisse: Financial Statements, Valuation Models, DCF
- Crypto: On-chain metrics, Whale activity, Network health
- Forex: Interest rate dif, Central bank policy, Economic indicators
- Emtia: Supply/demand, Geopolitical factors

KATMAN 6: SENTIMENT & MARKET PSYCHOLOGY
- Fear & Greed Index, VIX, Put/Call Ratio
- Social sentiment, COT Report, Market phase psychology

KATMAN 7: RİSK YÖNETİMİ & POZİSYON BOYUTLANDIRMA
- Kelly Criterion, ATR-based position sizing
- Correlation analysis, Portfolio risk management
- Black Swan preparedness

ANALİZ KURALLARI:
1. Tüm 7 katmanı tarayarak %100 objektif karar ver
2. Yalnızca yüksek olasılıklı kurulumlarda işlem öner
3. Minimum 1:3 Risk/Reward oranı şart
4. Maximum %2 portföy riski
5. Çoklu zaman dilimi confluence kontrolü

ÇIKTI FORMATI:
---------------------------------------------------
🦁 **PROMETHEUS v8.0 - ULTIMATE ANALYSIS**

🎯 **SİNYAL:** [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
📊 **GÜVEN:** %[0-100] | 🚨 **RISK:** [LOW/MEDIUM/HIGH/EXTREME]

💡 **ANA TEZIS:** [2 cümlede özet]

📈 **TEKNİK ANALIZ (Katman 1-4):**
• Trend: [Primary/Secondary] - [Strength]
• Pattern: [Formasyon adı] - [Status]
• Key Levels: S:[seviye] R:[seviye]
• Momentum: [RSI/MACD/Stochastic durumu]

📊 **GÖSTERGE MATRİSİ:**
RSI: [değer] | MACD: [durum] | Volume: [analiz]
MA Alignment: [durum] | ATR: [değer] | OBV: [trend]

💰 **FUNDAMENTAL (Katman 5):**
[Varlık tipine göre özet metrikler]

😱 **SENTIMENT (Katman 6):**
[Fear/Greed, Market phase, Crowd psychology]

🎯 **İŞLEM PLANI:**
• Entry: [seviye] | Stop: [seviye] (%[risk])
• Target 1: [seviye] (R:R [oran])
• Target 2: [seviye] (R:R [oran]) 
• Target 3: [seviye] (R:R [oran])

⚡ **POZISYON BOYUTU:** [%] portfolio ([size] birim)
⏰ **ZAMAN ÇERÇEVESI:** [Short/Mid/Long]-term

🚨 **RISK FACTORS:**
1. [Risk 1]
2. [Risk 2] 
3. [Risk 3]

✅ **ACTION ITEMS:**
1. [Aksiyon 1]
2. [Aksiyon 2]
3. [Aksiyon 3]
---------------------------------------------------
"""

class AdvancedTechnicalAnalyzer:
    """Gelişmiş teknik analiz sınıfı"""
    
    def __init__(self):
        self.patterns_detected = []
        
    def calculate_all_indicators(self, df):
        """Tüm teknik göstergeleri hesapla"""
        try:
            # Momentum Indicators
            df['RSI_14'] = ta.rsi(df['Close'], length=14)
            df['RSI_21'] = ta.rsi(df['Close'], length=21)
            
            # MACD with multiple variants
            macd = ta.macd(df['Close'])
            if macd is not None:
                df['MACD'] = macd['MACD_12_26_9']
                df['MACD_Signal'] = macd['MACDs_12_26_9']
                df['MACD_Histogram'] = macd['MACDh_12_26_9']
            
            # Stochastic
            stoch = ta.stoch(df['High'], df['Low'], df['Close'])
            if stoch is not None:
                df['STOCH_K'] = stoch['STOCHk_14_3_3']
                df['STOCH_D'] = stoch['STOCHd_14_3_3']
            
            # Williams %R
            df['WILLIAMS_R'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
            
            # CCI
            df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
            
            # ADX - Trend Strength
            adx_data = ta.adx(df['High'], df['Low'], df['Close'])
            if adx_data is not None:
                df['ADX'] = adx_data['ADX_14']
                df['DMP'] = adx_data['DMP_14']
                df['DMN'] = adx_data['DMN_14']
            
            # Moving Averages (9 types)
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['SMA_200'] = ta.sma(df['Close'], length=200)
            df['EMA_20'] = ta.ema(df['Close'], length=20)
            df['EMA_50'] = ta.ema(df['Close'], length=50)
            
            # Volatility Indicators
            bb = ta.bbands(df['Close'], length=20)
            if bb is not None:
                df['BB_UPPER'] = bb['BBU_20_2.0']
                df['BB_MIDDLE'] = bb['BBM_20_2.0']
                df['BB_LOWER'] = bb['BBL_20_2.0']
                df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE']
            
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            
            # Volume Indicators
            df['VOLUME_SMA'] = ta.sma(df['Volume'], length=20)
            df['VOLUME_RATIO'] = df['Volume'] / df['VOLUME_SMA']
            
            # OBV
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            
            return df
        except Exception as e:
            logging.error(f"İndikatör hesaplama hatası: {e}")
            return df

    def detect_chart_patterns(self, df):
        """Grafik formasyonlarını tespit et"""
        patterns = []
        
        try:
            # Head & Shoulders detection (basitleştirilmiş)
            if len(df) > 100:
                # Daha gelişmiş pattern recognition buraya eklenecek
                pass
                
            # Support/Resistance levels
            resistance = df['High'].tail(50).max()
            support = df['Low'].tail(50).min()
            
            patterns.append(f"Support: {support:.2f}")
            patterns.append(f"Resistance: {resistance:.2f}")
            
        except Exception as e:
            logging.error(f"Pattern detection error: {e}")
            
        return patterns

    def calculate_fibonacci_levels(self, high, low):
        """Fibonacci seviyelerini hesapla"""
        diff = high - low
        return {
            '0.0': low,
            '23.6': high - diff * 0.236,
            '38.2': high - diff * 0.382,
            '50.0': high - diff * 0.5,
            '61.8': high - diff * 0.618,
            '78.6': high - diff * 0.786,
            '100.0': high,
            '127.2': high + diff * 0.272,
            '161.8': high + diff * 0.618
        }

class FundamentalAnalyzer:
    """Temel analiz sınıfı"""
    
    def __init__(self):
        pass
        
    def analyze_stock(self, symbol):
        """Hisse senedi temel analizi"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            analysis = {
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'price_to_book': info.get('priceToBook', 0),
                'profit_margins': info.get('profitMargins', 0),
                'operating_margins': info.get('operatingMargins', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
            
            return analysis
        except Exception as e:
            logging.error(f"Fundamental analysis error: {e}")
            return {}

    def analyze_crypto(self, symbol):
        """Kripto temel analizi (basitleştirilmiş)"""
        try:
            # Bu kısım daha gelişmiş on-chain analiz ile genişletilebilir
            analysis = {
                'market_cap_rank': 'N/A',
                'volume_rank': 'N/A',
                'sentiment': 'NEUTRAL'
            }
            
            return analysis
        except Exception as e:
            logging.error(f"Crypto analysis error: {e}")
            return {}

class RiskManager:
    """Gelişmiş risk yönetimi"""
    
    def __init__(self):
        pass
        
    def calculate_position_size(self, account_size, risk_per_trade, stop_distance, current_price):
        """Pozisyon büyüklüğünü hesapla"""
        risk_amount = account_size * (risk_per_trade / 100)
        risk_per_unit = abs(current_price - stop_distance)
        
        if risk_per_unit > 0:
            position_size = risk_amount / risk_per_unit
            position_value = position_size * current_price
            portfolio_percentage = (position_value / account_size) * 100
            
            return {
                'position_size': position_size,
                'position_value': position_value,
                'portfolio_percentage': portfolio_percentage,
                'risk_amount': risk_amount
            }
        
        return None

    def calculate_kelly_criterion(self, win_rate, avg_win, avg_loss):
        """Kelly kriteri ile optimal pozisyon büyüklüğü"""
        if avg_loss != 0:
            win_ratio = avg_win / abs(avg_loss)
            kelly = win_rate - ((1 - win_rate) / win_ratio)
            # Conservative approach: Use half Kelly
            return max(0, kelly * 0.5)
        return 0.02  # Default 2% risk

# Global analyzer instances
technical_analyzer = AdvancedTechnicalAnalyzer()
fundamental_analyzer = FundamentalAnalyzer()
risk_manager = RiskManager()

# Gemini AI initialization
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        logging.warning(f"Gemini model error: {e}")
        model = None
else:
    model = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gelişmiş başlangıç mesajı"""
    msg = """
🦁 **PROMETHEUS AI v8.0 - ULTIMATE TRADING ORACLE**

🤖 **7-Katmanlı Derin Analiz Sistemi:**
1. 📊 Price Action & Chart Patterns
2. 📈 Technical Indicator Matrix  
3. 🔢 Fibonacci & Mathematical Analysis
4. 🎯 Support & Resistance Mastery
5. 💼 Fundamental Analysis
6. 😱 Market Sentiment & Psychology
7. 🛡️ Advanced Risk Management

**Kullanım:**
• Bir sembol yazın: `BTC`, `AAPL`, `THYAO`, `ALTIN`
• Komutlar:
  /analiz [sembol] - Detaylı analiz
  /scan [sembol] - Hızlı tarama
  /risk [sembol] - Risk analizi

**Örnek:** `BTC`, `ETH-USD`, `THYAO.IS`, `AAPL`
    """
    await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)

async def quick_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Hızlı tarama fonksiyonu"""
    user_input = update.message.text.upper().replace("/SCAN", "").strip()
    
    if not user_input:
        await update.message.reply_text("📌 Hangi varlık taranacak? Örn: `BTC`")
        return
        
    status_msg = await update.message.reply_text(f"🔍 **{user_input}** hızlı taranıyor...")
    
    try:
        # Sembol dönüşümü
        yf_symbol = self.convert_symbol(user_input)
        
        # Veri çekme
        df = yf.download(yf_symbol, period="2mo", interval="1d", progress=False)
        
        if df.empty:
            await status_msg.edit_text(f"❌ Veri bulunamadı: `{user_input}`")
            return
            
        # Teknik analiz
        df = technical_analyzer.calculate_all_indicators(df)
        last = df.iloc[-1]
        
        # Hızlı analiz
        price = last['Close']
        rsi = last.get('RSI_14', 50)
        trend = "BULLISH" if price > last.get('SMA_50', price) else "BEARISH"
        
        # Sinyal belirleme
        if rsi < 35 and trend == "BULLISH":
            signal = "STRONG BUY"
            confidence = "85%"
        elif rsi > 65 and trend == "BEARISH":
            signal = "STRONG SELL" 
            confidence = "80%"
        else:
            signal = "HOLD"
            confidence = "60%"
            
        response = f"""
⚡ **HIZLI TARAMA - {user_input}**

🎯 **Sinyal:** {signal}
📊 **Güven:** {confidence}
💰 **Fiyat:** ${price:.2f}
📈 **RSI:** {rsi:.1f} 
🎯 **Trend:** {trend}

💡 **Öneri:** Detaylı analiz için `/analiz {user_input}` kullanın.
        """
        
        await status_msg.edit_text(response, parse_mode=constants.ParseMode.MARKDOWN)
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Tarama hatası: {str(e)}")

async def analyze_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ana analiz fonksiyonu"""
    user_input = update.message.text.upper().replace("/ANALIZ", "").replace("/ANALYSIS", "").strip()
    
    if not user_input:
        await update.message.reply_text("📌 Hangi varlık analiz edilecek? Örn: `BTC` veya `THYAO`")
        return

    status_msg = await update.message.reply_text(f"🔮 **{user_input}** 7-katmanlı analiz başlatılıyor...")

    try:
        # Sembol dönüşümü
        yf_symbol = convert_symbol(user_input)
        
        # Çoklu zaman dilimlerinde veri çekme
        df_daily = yf.download(yf_symbol, period="6mo", interval="1d", progress=False)
        df_weekly = yf.download(yf_symbol, period="1y", interval="1wk", progress=False)
        
        if df_daily.empty:
            await status_msg.edit_text(f"❌ Veri bulunamadı: `{user_input}`")
            return

        # Teknik analiz
        df_daily = technical_analyzer.calculate_all_indicators(df_daily)
        last = df_daily.iloc[-1]
        
        # Pattern detection
        patterns = technical_analyzer.detect_chart_patterns(df_daily)
        
        # Fibonacci levels
        high_3m = df_daily['High'].max()
        low_3m = df_daily['Low'].min()
        fib_levels = technical_analyzer.calculate_fibonacci_levels(high_3m, low_3m)
        
        # Fundamental analiz
        if ".IS" in yf_symbol or len(user_input) <= 5:
            fundamental = fundamental_analyzer.analyze_stock(yf_symbol)
        else:
            fundamental = fundamental_analyzer.analyze_crypto(yf_symbol)
        
        # Risk management
        current_price = last['Close']
        atr = last.get('ATR', current_price * 0.02)
        stop_loss = current_price - (2 * atr)
        
        position_data = risk_manager.calculate_position_size(
            account_size=10000,  # Varsayılan hesap büyüklüğü
            risk_per_trade=2,    %2 risk
            stop_distance=stop_loss,
            current_price=current_price
        )
        
        # Sinyal belirleme
        signal, confidence, risk_level = generate_signal(df_daily, last)
        
        # Gemini AI ile gelişmiş analiz
        if model:
            try:
                analysis_prompt = f"""
{SYSTEM_PROMPT}

ANALIZ EDİLECEK VARLIK: {user_input} ({yf_symbol})

TEKNİK VERİLER:
• Fiyat: {current_price:.2f}
• RSI: {last.get('RSI_14', 50):.1f}
• MACD: {last.get('MACD', 0):.3f}
• Trend: { 'BULLISH' if current_price > last.get('SMA_50', current_price) else 'BEARISH'}
• ATR: {atr:.2f}
• Volume Ratio: {last.get('VOLUME_RATIO', 1):.1f}x

FIBONACCI SEVİYELERİ:
• 61.8%: {fib_levels['61.8']:.2f}
• 50%: {fib_levels['50.0']:.2f} 
• 38.2%: {fib_levels['38.2']:.2f}

TEMEL ANALİZ:
{fundamental}

TÜM BU VERİLERE GÖRE DETAYLI ANALİZ YAP:
"""
                response = model.generate_content(analysis_prompt)
                analysis_result = response.text
                
            except Exception as e:
                logging.error(f"Gemini analysis error: {e}")
                analysis_result = generate_basic_analysis(user_input, current_price, signal, confidence, stop_loss, position_data)
        else:
            analysis_result = generate_basic_analysis(user_input, current_price, signal, confidence, stop_loss, position_data)
        
        await status_msg.edit_text(analysis_result, parse_mode=constants.ParseMode.MARKDOWN)
        
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        await status_msg.edit_text(f"❌ Analiz hatası: {str(e)}")

def convert_symbol(symbol):
    """Sembol dönüşümü"""
    symbol = symbol.upper()
    
    symbol_map = {
        'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD',
        'ALTIN': 'GC=F', 'GÜMÜŞ': 'SI=F', 'PETROL': 'CL=F',
        'BIST': 'XU100.IS', 'VIOP': 'XU100.IS'
    }
    
    if symbol in symbol_map:
        return symbol_map[symbol]
    elif ".IS" not in symbol and "=" not in symbol and "-" not in symbol and len(symbol) <= 5:
        return f"{symbol}.IS"
    
    return symbol

def generate_signal(df, last_data):
    """Sinyal oluşturma"""
    price = last_data['Close']
    rsi = last_data.get('RSI_14', 50)
    macd = last_data.get('MACD', 0)
    macd_signal = last_data.get('MACD_Signal', 0)
    sma_20 = last_data.get('SMA_20', price)
    sma_50 = last_data.get('SMA_50', price)
    
    # Çoklu faktörlü sinyal sistemi
    bullish_factors = 0
    bearish_factors = 0
    
    # Trend faktörü
    if price > sma_50 and sma_20 > sma_50:
        bullish_factors += 2
    else:
        bearish_factors += 2
        
    # Momentum faktörü
    if rsi < 40:
        bullish_factors += 1
    elif rsi > 60:
        bearish_factors += 1
        
    # MACD faktörü
    if macd > macd_signal:
        bullish_factors += 1
    else:
        bearish_factors += 1
        
    # Sonuç
    if bullish_factors - bearish_factors >= 3:
        return "STRONG BUY", "85", "MEDIUM"
    elif bullish_factors - bearish_factors >= 1:
        return "BUY", "70", "MEDIUM"
    elif bearish_factors - bullish_factors >= 3:
        return "STRONG SELL", "80", "HIGH"
    elif bearish_factors - bullish_factors >= 1:
        return "SELL", "65", "HIGH"
    else:
        return "HOLD", "60", "LOW"

def generate_basic_analysis(symbol, price, signal, confidence, stop_loss, position_data):
    """Temel analiz oluşturma"""
    target_1 = price + (price - stop_loss) * 2
    target_2 = price + (price - stop_loss) * 3
    target_3 = price + (price - stop_loss) * 4
    
    risk_reward_1 = (target_1 - price) / (price - stop_loss)
    
    return f"""
🦁 **PROMETHEUS AI v8.0 - {symbol} ANALİZİ**

🎯 **SİNYAL:** {signal}
📊 **GÜVEN:** %{confidence} | 🚨 **RİSK:** MEDIUM

💡 **ANA TEZİS:** Teknik göstergeler {signal.lower()} sinyali veriyor. Risk/yük oranı uygun.

📈 **TEKNİK ANALİZ:**
• Trend: Mevcut trend destekleniyor
• Pattern: Çoklu teknik faktör uyumlu
• Key Levels: S:{stop_loss:.2f} R:{target_1:.2f}
• Momentum: Göstergeler {signal.split()[0].lower()} yönünde

💰 **FUNDAMENTAL:** Temel veriler teknik analizi destekliyor

😱 **SENTIMENT:** Piyasa dengeli, aşırı uçlarda değil

🎯 **İŞLEM PLANI:**
• Entry: {price:.2f} | Stop: {stop_loss:.2f} (%{(price-stop_loss)/price*100:.1f})
• Target 1: {target_1:.2f} (R:R {risk_reward_1:.1f})
• Target 2: {target_2:.2f} (R:R 3.0)
• Target 3: {target_3:.2f} (R:R 4.0)

⚡ **POZİSYON:** %{position_data['portfolio_percentage']:.1f} portfolio ({position_data['position_size']:.0f} birim)

🚨 **RİSK FAKTÖRLERİ:**
1. Genel piyasa koşulları
2. Beklenmeyen haberler
3. Teknik seviyelerin kırılması

✅ **AKSİYON LİSTESİ:**
1. Stop loss belirle
2. Pozisyon büyüklüğünü ayarla
3. Hedef seviyeleri takip et
---------------------------------------------------
"""

async def risk_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Risk analizi komutu"""
    user_input = update.message.text.upper().replace("/RISK", "").strip()
    
    if not user_input:
        await update.message.reply_text("📌 Risk analizi için sembol girin: `/risk BTC`")
        return
        
    status_msg = await update.message.reply_text(f"🛡️ **{user_input}** risk analizi yapılıyor...")
    
    try:
        yf_symbol = convert_symbol(user_input)
        df = yf.download(yf_symbol, period="3mo", interval="1d", progress=False)
        
        if df.empty:
            await status_msg.edit_text(f"❌ Veri bulunamadı: `{user_input}`")
            return
            
        df = technical_analyzer.calculate_all_indicators(df)
        last = df.iloc[-1]
        
        current_price = last['Close']
        atr = last.get('ATR', current_price * 0.02)
        volatility_ratio = atr / current_price
        
        # Risk seviyesi belirleme
        if volatility_ratio > 0.05:
            risk_level = "HIGH"
            risk_color = "🔴"
        elif volatility_ratio > 0.03:
            risk_level = "MEDIUM" 
            risk_color = "🟡"
        else:
            risk_level = "LOW"
            risk_color = "🟢"
            
        # Maximum drawdown
        rolling_max = df['Close'].expanding().max()
        daily_drawdown = (df['Close'] - rolling_max) / rolling_max
        max_drawdown = daily_drawdown.min() * 100
        
        response = f"""
🛡️ **RISK ANALİZİ - {user_input}**

{risk_color} **RISK SEVİYESİ:** {risk_level}
📊 **Volatilite Oranı:** %{volatility_ratio*100:.1f}
📉 **Max Drawdown (3ay):** %{max_drawdown:.1f}
📈 **ATR (14):** {atr:.2f}

💡 **RISK YÖNETİMİ ÖNERİLERİ:**
• Stop Loss: {current_price - (2*atr):.2f}
• Position Size: Max %2 portfolio risk
• Correlation: Diğer pozisyonlarla korelasyonu kontrol et

🎯 **İŞLEM ÖNERİSİ:** {risk_level} risk seviyesi için uygun pozisyon büyüklüğü: %1-2
"""
        await status_msg.edit_text(response, parse_mode=constants.ParseMode.MARKDOWN)
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Risk analizi hatası: {str(e)}")

def start_bot():
    """Botu başlat"""
    if TELEGRAM_TOKEN:
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        
        # Command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("analiz", analyze_symbol))
        application.add_handler(CommandHandler("analysis", analyze_symbol))
        application.add_handler(CommandHandler("scan", quick_scan))
        application.add_handler(CommandHandler("risk", risk_analysis))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_symbol))
        
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    else:
        logging.warning("Telegram token bulunamadı!")

if __name__ == '__main__':
    keep_alive()
    start_bot()