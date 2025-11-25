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

# --- GELİŞMİŞ API AYARLARI ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

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
            # DataFrame kontrolü
            if df.empty:
                return df
                
            # Momentum Indicators
            df['RSI_14'] = ta.rsi(df['Close'], length=14)
            df['RSI_21'] = ta.rsi(df['Close'], length=21)
            
            # MACD
            try:
                macd = ta.macd(df['Close'])
                if macd is not None:
                    df['MACD'] = macd.get('MACD_12_26_9', 0)
                    df['MACD_Signal'] = macd.get('MACDs_12_26_9', 0)
                    df['MACD_Histogram'] = macd.get('MACDh_12_26_9', 0)
            except Exception as e:
                logging.warning(f"MACD hesaplama hatası: {e}")
                df['MACD'] = 0
                df['MACD_Signal'] = 0
                df['MACD_Histogram'] = 0
            
            # Stochastic
            try:
                stoch = ta.stoch(df['High'], df['Low'], df['Close'])
                if stoch is not None:
                    df['STOCH_K'] = stoch.get('STOCHk_14_3_3', 50)
                    df['STOCH_D'] = stoch.get('STOCHd_14_3_3', 50)
            except Exception as e:
                logging.warning(f"Stochastic hesaplama hatası: {e}")
                df['STOCH_K'] = 50
                df['STOCH_D'] = 50
            
            # Williams %R
            df['WILLIAMS_R'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
            
            # CCI
            df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
            
            # ADX - Trend Strength
            try:
                adx_data = ta.adx(df['High'], df['Low'], df['Close'])
                if adx_data is not None:
                    df['ADX'] = adx_data.get('ADX_14', 20)
                    df['DMP'] = adx_data.get('DMP_14', 0)
                    df['DMN'] = adx_data.get('DMN_14', 0)
            except Exception as e:
                logging.warning(f"ADX hesaplama hatası: {e}")
                df['ADX'] = 20
                df['DMP'] = 0
                df['DMN'] = 0
            
            # Moving Averages
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['SMA_200'] = ta.sma(df['Close'], length=200)
            df['EMA_20'] = ta.ema(df['Close'], length=20)
            df['EMA_50'] = ta.ema(df['Close'], length=50)
            
            # Volatility Indicators
            try:
                bb = ta.bbands(df['Close'], length=20)
                if bb is not None:
                    df['BB_UPPER'] = bb.get('BBU_20_2.0', df['Close'])
                    df['BB_MIDDLE'] = bb.get('BBM_20_2.0', df['Close'])
                    df['BB_LOWER'] = bb.get('BBL_20_2.0', df['Close'])
                    df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE']
            except Exception as e:
                logging.warning(f"Bollinger Bands hatası: {e}")
                df['BB_UPPER'] = df['Close']
                df['BB_MIDDLE'] = df['Close']
                df['BB_LOWER'] = df['Close']
                df['BB_WIDTH'] = 0
            
            # ATR
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            
            # Volume Indicators
            df['VOLUME_SMA'] = ta.sma(df['Volume'], length=20)
            df['VOLUME_RATIO'] = df['Volume'] / df['VOLUME_SMA'].replace(0, 1)
            
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
            if len(df) < 50:
                return ["Yeterli veri yok"]
                
            # Basit destek/direnç seviyeleri
            resistance = df['High'].tail(50).max()
            support = df['Low'].tail(50).min()
            
            patterns.append(f"Destek: {support:.2f}")
            patterns.append(f"Direnç: {resistance:.2f}")
            
            # Trend analizi
            current_price = df['Close'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df else current_price
            
            if current_price > sma_50:
                patterns.append("Trend: YÜKSELİŞ")
            else:
                patterns.append("Trend: DÜŞÜŞ")
                
        except Exception as e:
            logging.error(f"Pattern detection error: {e}")
            patterns = ["Pattern analiz hatası"]
            
        return patterns

    def calculate_fibonacci_levels(self, high, low):
        """Fibonacci seviyelerini hesapla"""
        try:
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
        except:
            return {}

class FundamentalAnalyzer:
    """Temel analiz sınıfı"""
    
    def analyze_stock(self, symbol):
        """Hisse senedi temel analizi"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'market_cap': self.format_number(info.get('marketCap', 0)),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'price_to_book': info.get('priceToBook', 'N/A'),
                'profit_margins': f"%{info.get('profitMargins', 0)*100:.1f}" if info.get('profitMargins') else 'N/A',
                'dividend_yield': f"%{info.get('dividendYield', 0)*100:.2f}" if info.get('dividendYield') else 'N/A'
            }
        except Exception as e:
            logging.error(f"Fundamental analysis error: {e}")
            return {'error': 'Temel analiz yapılamadı'}

    def analyze_crypto(self, symbol):
        """Kripto temel analizi"""
        try:
            return {
                'type': 'CRYPTO',
                'analysis': 'On-chain analiz mevcut değil',
                'market_sentiment': 'NÖTR'
            }
        except Exception as e:
            logging.error(f"Crypto analysis error: {e}")
            return {'error': 'Kripto analiz hatası'}

    def format_number(self, num):
        """Sayıları formatla"""
        if num >= 1e9:
            return f"${num/1e9:.2f}B"
        elif num >= 1e6:
            return f"${num/1e6:.2f}M"
        else:
            return f"${num:.2f}"

class RiskManager:
    """Gelişmiş risk yönetimi"""
    
    def calculate_position_size(self, account_size, risk_per_trade, stop_distance, current_price):
        """Pozisyon büyüklüğünü hesapla"""
        try:
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
        except Exception as e:
            logging.error(f"Position size calculation error: {e}")
        
        return {
            'position_size': 0,
            'position_value': 0,
            'portfolio_percentage': 0,
            'risk_amount': 0
        }

# Global analyzer instances
technical_analyzer = AdvancedTechnicalAnalyzer()
fundamental_analyzer = FundamentalAnalyzer()
risk_manager = RiskManager()

# Gemini AI initialization
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        logging.warning(f"Gemini model error: {e}")
        model = None
else:
    model = None
    logging.warning("Gemini API key bulunamadı - Temel analiz kullanılacak")

def convert_symbol(symbol):
    """Sembol dönüşümü"""
    symbol = symbol.upper().strip()
    
    symbol_map = {
        'BTC': 'BTC-USD', 
        'ETH': 'ETH-USD', 
        'SOL': 'SOL-USD',
        'ALTIN': 'GC=F', 
        'GÜMÜŞ': 'SI=F', 
        'PETROL': 'CL=F',
        'BIST': 'XU100.IS', 
        'VIOP': 'XU100.IS'
    }
    
    if symbol in symbol_map:
        return symbol_map[symbol]
    elif ".IS" not in symbol and "=" not in symbol and "-" not in symbol and len(symbol) <= 5:
        return f"{symbol}.IS"
    
    return symbol

def generate_signal(df, last_data):
    """Sinyal oluşturma"""
    try:
        price = last_data['Close']
        rsi = last_data.get('RSI_14', 50)
        macd = last_data.get('MACD', 0)
        macd_signal = last_data.get('MACD_Signal', 0)
        sma_50 = last_data.get('SMA_50', price)
        
        bullish_factors = 0
        bearish_factors = 0
        
        # Trend faktörü
        if price > sma_50:
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
    except Exception as e:
        logging.error(f"Signal generation error: {e}")
        return "HOLD", "50", "MEDIUM"

def generate_basic_analysis(symbol, price, signal, confidence, stop_loss, position_data, patterns, fundamental):
    """Temel analiz oluşturma"""
    try:
        target_1 = price + (price - stop_loss) * 2
        target_2 = price + (price - stop_loss) * 3
        target_3 = price + (price - stop_loss) * 4
        
        risk_reward_1 = (target_1 - price) / (price - stop_loss) if (price - stop_loss) > 0 else 0
        
        patterns_text = "\n".join(patterns[:3]) if patterns else "Pattern analiz edilemedi"
        
        fundamental_text = ""
        if fundamental and 'error' not in fundamental:
            for key, value in list(fundamental.items())[:3]:
                fundamental_text += f"{key}: {value}\n"
        
        return f"""
🦁 **PROMETHEUS AI v8.0 - {symbol} ANALİZİ**

🎯 **SİNYAL:** {signal}
📊 **GÜVEN:** %{confidence} | 🚨 **RİSK:** MEDIUM

💡 **ANA TEZİS:** Teknik göstergeler {signal.lower()} sinyali veriyor.

📈 **TEKNİK ANALİZ:**
{patterns_text}

📊 **GÖSTERGE MATRİSİ:**
• RSI: {price:.1f} | Trend: {signal.split()[0]}
• Key Levels: S:{stop_loss:.2f} R:{target_1:.2f}

💰 **FUNDAMENTAL:**
{fundamental_text if fundamental_text else 'Temel analiz mevcut değil'}

🎯 **İŞLEM PLANI:**
• Entry: {price:.2f} | Stop: {stop_loss:.2f}
• Target 1: {target_1:.2f} (R:R {risk_reward_1:.1f})
• Target 2: {target_2:.2f}
• Target 3: {target_3:.2f}

⚡ **POZİSYON:** %{position_data['portfolio_percentage']:.1f} portfolio

🚨 **RİSK FAKTÖRLERİ:**
1. Piyasa volatilitesi
2. Beklenmeyen haberler
3. Teknik seviye kırılmaları

✅ **AKSİYON LİSTESİ:**
1. Stop loss belirle
2. Pozisyon büyüklüğünü ayarla
3. Hedefleri takip et
---------------------------------------------------
"""
    except Exception as e:
        logging.error(f"Basic analysis generation error: {e}")
        return f"❌ Analiz oluşturulurken hata: {str(e)}"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Başlangıç mesajı"""
    msg = """
🦁 **PROMETHEUS AI v8.0 - ULTIMATE TRADING ORACLE**

🤖 **7-Katmanlı Derin Analiz Sistemi**
📊 Teknik Analiz • 💼 Fundamental • 🛡️ Risk Management

**Kullanım:**
• Bir sembol yazın: `BTC`, `AAPL`, `THYAO`, `ALTIN`
• Komutlar:
  /analiz [sembol] - Detaylı analiz
  /scan [sembol] - Hızlı tarama
  /risk [sembol] - Risk analizi

**Örnek:** `BTC`, `THYAO.IS`, `AAPL`
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
        yf_symbol = convert_symbol(user_input)
        df = yf.download(yf_symbol, period="1mo", interval="1d", progress=False)
        
        if df.empty:
            await status_msg.edit_text(f"❌ Veri bulunamadı: `{user_input}`")
            return
            
        df = technical_analyzer.calculate_all_indicators(df)
        last = df.iloc[-1]
        
        price = last['Close']
        rsi = last.get('RSI_14', 50)
        trend = "BULLISH" if price > last.get('SMA_50', price) else "BEARISH"
        
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

💡 **Öneri:** Detaylı analiz için `/analiz {user_input}`
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
        yf_symbol = convert_symbol(user_input)
        df_daily = yf.download(yf_symbol, period="3mo", interval="1d", progress=False)
        
        if df_daily.empty:
            await status_msg.edit_text(f"❌ Veri bulunamadı: `{user_input}`")
            return

        df_daily = technical_analyzer.calculate_all_indicators(df_daily)
        last = df_daily.iloc[-1]
        
        patterns = technical_analyzer.detect_chart_patterns(df_daily)
        
        current_price = last['Close']
        atr = last.get('ATR', current_price * 0.02)
        stop_loss = current_price - (2 * atr)
        
        position_data = risk_manager.calculate_position_size(
            account_size=10000,
            risk_per_trade=2,
            stop_distance=stop_loss,
            current_price=current_price
        )
        
        signal, confidence, risk_level = generate_signal(df_daily, last)
        
        if ".IS" in yf_symbol or len(user_input) <= 5:
            fundamental = fundamental_analyzer.analyze_stock(yf_symbol)
        else:
            fundamental = fundamental_analyzer.analyze_crypto(yf_symbol)
        
        if model:
            try:
                analysis_prompt = f"""
{SYSTEM_PROMPT}

ANALIZ EDİLECEK VARLIK: {user_input} ({yf_symbol})

TEKNİK VERİLER:
• Fiyat: {current_price:.2f}
• RSI: {last.get('RSI_14', 50):.1f}
• Trend: {'BULLISH' if current_price > last.get('SMA_50', current_price) else 'BEARISH'}
• ATR: {atr:.2f}

PATTERNS: {patterns}

TEMEL ANALİZ: {fundamental}

DETAYLI ANALİZ YAP:
"""
                response = model.generate_content(analysis_prompt)
                analysis_result = response.text
            except Exception as e:
                logging.error(f"Gemini analysis error: {e}")
                analysis_result = generate_basic_analysis(user_input, current_price, signal, confidence, stop_loss, position_data, patterns, fundamental)
        else:
            analysis_result = generate_basic_analysis(user_input, current_price, signal, confidence, stop_loss, position_data, patterns, fundamental)
        
        await status_msg.edit_text(analysis_result, parse_mode=constants.ParseMode.MARKDOWN)
        
    except Exception as e:
        logging.error(f"Analysis error: {e}")
        await status_msg.edit_text(f"❌ Analiz hatası: {str(e)}")

async def risk_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Risk analizi"""
    user_input = update.message.text.upper().replace("/RISK", "").strip()
    
    if not user_input:
        await update.message.reply_text("📌 Risk analizi için sembol girin: `/risk BTC`")
        return
        
    status_msg = await update.message.reply_text(f"🛡️ **{user_input}** risk analizi...")
    
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
        
        if volatility_ratio > 0.05:
            risk_level = "HIGH"
        elif volatility_ratio > 0.03:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        response = f"""
🛡️ **RISK ANALİZİ - {user_input}**

📊 **Risk Seviyesi:** {risk_level}
📈 **Volatilite:** %{volatility_ratio*100:.1f}
🎯 **ATR:** {atr:.2f}

💡 **Öneriler:**
• Stop Loss: {current_price - (2*atr):.2f}
• Position Size: Max %2 risk
• Dikkatli izleme önerilir
"""
        await status_msg.edit_text(response, parse_mode=constants.ParseMode.MARKDOWN)
        
    except Exception as e:
        await status_msg.edit_text(f"❌ Risk analizi hatası: {str(e)}")

def start_bot():
    """Botu başlat"""
    if not TELEGRAM_TOKEN:
        logging.error("TELEGRAM_TOKEN bulunamadı!")
        return
        
    try:
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("analiz", analyze_symbol))
        application.add_handler(CommandHandler("analysis", analyze_symbol))
        application.add_handler(CommandHandler("scan", quick_scan))
        application.add_handler(CommandHandler("risk", risk_analysis))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_symbol))
        
        logging.info("Bot başlatılıyor...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        logging.error(f"Bot başlatma hatası: {e}")

if __name__ == '__main__':
    keep_alive()
    start_bot()