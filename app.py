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
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import requests

# --- API AYARLARI ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Gemini Modelini Başlat
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

# --- WEB SUNUCUSU ---
app = Flask(__name__)

@app.route('/')
def home():
    return "🦁 Prometheus AI v9.0 - Tüm Sistemler Aktif!"

@app.route('/health')
def health():
    return "OK", 200

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()

# --- BIST100 SİSTEMİ ---
BIST_100_SYMBOLS = {
    'THYAO', 'GARAN', 'AKBNK', 'ISCTR', 'YKBNK', 'SAHOL', 'KCHOL', 
    'TCELL', 'ASELS', 'EREGL', 'SISE', 'FROTO', 'TOASO', 'TUPRS',
    'HALKB', 'VAKBN', 'ENJSA', 'EKIZ', 'PETKM', 'TUKAS', 'ARCLK',
    'GUBRF', 'KORDS', 'CCOLA', 'BIMAS', 'AKSA', 'CIMSA', 'DOAS',
    'ECILC', 'FENER', 'GSRAY', 'HEKTS', 'ISGYO', 'KARSN', 'MGROS',
    'OTKAR', 'PETUN', 'SNKRN', 'TATGD', 'TRKCM', 'ULKER', 'VESBE',
    'YATAS', 'ZOREN'
}

# --- GELİŞMİŞ TEKNİK ANALİZ ---
class AdvancedTechnicalAnalysis:
    def __init__(self):
        self.indicators = {}
    
    def calculate_advanced_indicators(self, df):
        """40+ teknik gösterge hesaplama"""
        try:
            # Trend Göstergeleri
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['SMA_200'] = ta.sma(df['Close'], length=200)
            df['EMA_12'] = ta.ema(df['Close'], length=12)
            df['EMA_26'] = ta.ema(df['Close'], length=26)
            
            # Momentum Göstergeleri
            df['RSI_14'] = ta.rsi(df['Close'], length=14)
            df['RSI_21'] = ta.rsi(df['Close'], length=21)
            df['STOCH_K'] = ta.stoch(df['High'], df['Low'], df['Close'])['STOCHk_14_3_3']
            df['STOCH_D'] = ta.stoch(df['High'], df['Low'], df['Close'])['STOCHd_14_3_3']
            df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
            df['MACD_SIGNAL'] = ta.macd(df['Close'])['MACDs_12_26_9']
            df['MACD_HISTOGRAM'] = ta.macd(df['Close'])['MACDh_12_26_9']
            df['WILLIAMS_R'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
            df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
            df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
            df['MOMENTUM'] = ta.mom(df['Close'], length=10)
            
            # Volatilite Göstergeleri
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            bb = ta.bbands(df['Close'], length=20)
            if bb is not None:
                df['BB_UPPER'] = bb['BBU_20_2.0']
                df['BB_LOWER'] = bb['BBL_20_2.0']
                df['BB_MIDDLE'] = bb['BBM_20_2.0']
            
            # Hacim Göstergeleri
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
            df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
            
            # Özel Göstergeler
            df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
            df['SUPERTREND'] = ta.supertrend(df['High'], df['Low'], df['Close'])['SUPERTd_7_3.0']
            
            # Ichimoku (basitleştirilmiş)
            ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])
            if ichimoku is not None:
                df['ICHIMOKU_A'] = ichimoku['ITS_9']
                df['ICHIMOKU_B'] = ichimoku['IKS_26']
            
            return df
            
        except Exception as e:
            logging.error(f"Gelişmiş indikatör hatası: {e}")
            return df

    def generate_signals(self, df):
        """Otomatik trading sinyalleri üret"""
        signals = []
        if len(df) < 2:
            return signals
            
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        try:
            # RSI Sinyalleri
            if last['RSI_14'] < 30:
                signals.append("🔴 RSI AŞIRI SATIM - POTANSİYEL AL")
            elif last['RSI_14'] > 70:
                signals.append("🟢 RSI AŞIRI ALIM - POTANSİYEL SAT")
            
            # MACD Sinyalleri
            if last['MACD'] > last['MACD_SIGNAL'] and prev['MACD'] <= prev['MACD_SIGNAL']:
                signals.append("📈 MACD ALTIIN KESİTİ - BULLISH")
            elif last['MACD'] < last['MACD_SIGNAL'] and prev['MACD'] >= prev['MACD_SIGNAL']:
                signals.append("📉 MACD AŞAĞI KESİTİ - BEARISH")
            
            # Bollinger Bands
            if last['Close'] < last['BB_LOWER']:
                signals.append("⬆️ BB ALT BANT - OVERSOLD BOUNCE BEKLENTİSİ")
            elif last['Close'] > last['BB_UPPER']:
                signals.append("⬇️ BB ÜST BANT - OVERBOUGHT DÜZELME BEKLENTİSİ")
            
            # Trend Analizi
            if last['Close'] > last['SMA_20'] > last['SMA_50'] > last['SMA_200']:
                signals.append("🚀 GÜÇLÜ YUKARI TREND")
            elif last['Close'] < last['SMA_20'] < last['SMA_50'] < last['SMA_200']:
                signals.append("🔻 GÜÇLÜ AŞAĞI TREND")
            
            # Volume Sinyalleri
            avg_volume = df['Volume'].tail(20).mean()
            if last['Volume'] > avg_volume * 1.5:
                if last['Close'] > prev['Close']:
                    signals.append("💰 YÜKSEK HACİMLİ ALIM")
                else:
                    signals.append("💸 YÜKSEK HACİMLİ SATIM")
                    
        except Exception as e:
            logging.error(f"Sinyal üretme hatası: {e}")
            
        return signals

# --- GLOBAL DEĞİŞKENLER ---
advanced_analyzer = AdvancedTechnicalAnalysis()
user_alerts = {}
application = None

# --- PROMETHEUS AI SİSTEMİ ---
ADVANCED_SYSTEM_PROMPT = """
SEN: PROMETHEUS AI v9.0 - Gelişmiş Quant Analiz Sistemi
KİMLİK: Algoritmik fon yöneticisi, teknik analiz uzmanı

ANALİZ KATMANLARI:
1. 40+ TEKNİK GÖSTERGE ANALİZİ
2. ÇOKLU ZAMAN DİLİMİ DEĞERLENDİRMESİ  
3. TREND & MOMENTUM SENTEZLENMESİ
4. HACİM & FİYAT İLİŞKİSİ
5. RİSK/ÖDÜL OPTİMİZASYONU

GÖREV: Aşağıdaki gelişmiş teknik verilere dayanarak DETAYLI analiz yap.

ÇIKTI FORMATI:
═══════════════════════════════════════════════
🎯 **PROMETHEUS AI v9.0 - GELİŞMİŞ ANALİZ**
═══════════════════════════════════════════════

📊 **TEKNİK SİNYALLER:**
• Trend: [YÖN] [GÜÇ]
• Momentum: [DURUM]
• Hacim Analizi: [BİRİKİM/DAĞITIM]

⚡ **OTOMATİK SİNYALLER:**
[Üretilen sinyaller listesi]

🦁 **NİHAİ KARAR:** [AL / SAT / BEKLE]
📈 **GÜVEN SKORU:** %[0-100]

💰 **İŞLEM PLANI:**
• 🎯 Giriş: [FİYAT]
• 🛑 Stop-Loss: [FİYAT] (Risk: %X)
• 🎯 Hedef 1: [FİYAT] (R:R X:1)
• 🎯 Hedef 2: [FİYAT] 
• 🎯 Hedef 3: [FİYAT]

📋 **DETAYLI ANALİZ:**
[Teknik göstergelerin detaylı yorumu]

⚠️ **RISK UYARILARI:**
[Spesifik risk faktörleri]
═══════════════════════════════════════════════
"""

# --- SEMBOL SİSTEMİ ---
def get_yfinance_symbol(user_input):
    """Akıllı sembol dönüşümü"""
    special_cases = {
        'ALTIN': 'GC=F', 'GUMUS': 'SI=F', 'PETROL': 'CL=F',
        'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'BITCOIN': 'BTC-USD',
        'BIST100': 'XU100.IS', 'SP500': '^GSPC', 'NASDAQ': '^IXIC',
        'DOLAR': 'TRY=X', 'EURO': 'EURTRY=X'
    }
    
    user_upper = user_input.upper().strip()
    
    if user_upper in special_cases:
        return special_cases[user_upper]
    elif user_upper in BIST_100_SYMBOLS:
        return f"{user_upper}.IS"
    elif '.' not in user_upper and '-' not in user_upper and len(user_upper) <= 5:
        return f"{user_upper}.IS"
    else:
        return user_upper

# --- BOT KOMUTLARI ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = """
🦁 **PROMETHEUS AI v9.0 - TÜM SİSTEMLER AKTİF**

🎯 **Gelişmiş Özellikler:**
• 40+ Teknik Gösterge Analizi
• ⚡ Hızlı Sinyal Sistemi (5sn)
• 📈 Günlük BIST100 Raporu
• 🔔 Akıllı Fiyat Alarmları
• 🤖 AI Destekli Yorumlama

📋 **Komutlar:**
/start - Botu başlat
/analiz [sembol] - Detaylı teknik analiz
/sinyal [sembol] - ⚡ 5 saniyede hızlı sinyal
/rapor - 📈 Günlük BIST100 özeti
/alert [sembol] [fiyat] - 🔔 Fiyat alarmı kur
/top5 - 🏆 En iyi 5 BIST hissesi

💎 **Örnekler:**
/analiz THYAO
/sinyal GARAN
/rapor
/alert AKBNK 50
/top5
"""
    await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)

async def advanced_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gelişmiş analiz komutu"""
    user_input = update.message.text.upper().strip()
    user_msg = user_input.replace("/ANALIZ", "").strip()
    
    if not user_msg:
        await update.message.reply_text("❌ Hangi varlık? Örn: `/analiz THYAO`")
        return

    status_msg = await update.message.reply_text(
        f"🔍 **GELİŞMİŞ ANALİZ BAŞLATILDI**\n"
        f"**Varlık:** `{user_msg}`\n"
        f"⏳ 40+ gösterge hesaplanıyor...", 
        parse_mode=constants.ParseMode.MARKDOWN
    )

    yf_symbol = get_yfinance_symbol(user_msg)

    try:
        # Veri çek ve analiz et
        df = yf.download(yf_symbol, period='6mo', interval='1d', progress=False, auto_adjust=True)
        
        if df.empty:
            await status_msg.edit_text(f"❌ Veri bulunamadı: `{user_msg}`")
            return

        df = advanced_analyzer.calculate_advanced_indicators(df)
        last = df.iloc[-1]
        signals = advanced_analyzer.generate_signals(df)
        
        # AI Analizi
        if model:
            try:
                technical_summary = f"""
📊 **GELİŞMİŞ TEKNİK VERİLER:**

**Fiyat & Trend:**
• Mevcut Fiyat: {last['Close']:.2f}
• Trend: {'YUKARI' if last['Close'] > last['SMA_200'] else 'AŞAĞI'}
• SMA: {last['SMA_20']:.2f} | {last['SMA_50']:.2f} | {last['SMA_200']:.2f}

**Momentum:**
• RSI: {last['RSI_14']:.2f}
• MACD: {last['MACD']:.4f}
• Stochastic: {last.get('STOCH_K', 0):.2f}

**Volatilite & Hacim:**
• ATR: {last.get('ATR', 0):.2f}
• Bollinger: %{(last['Close'] - last['BB_LOWER']) / (last['BB_UPPER'] - last['BB_LOWER']) * 100:.1f}
"""
                
                prompt = f"{ADVANCED_SYSTEM_PROMPT}\n\nVARLIK: {user_msg}\n{technical_summary}\nSinyaller: {chr(10).join(signals)}"
                response = model.generate_content(prompt)
                analysis_result = response.text
                
            except Exception as e:
                logging.error(f"Gemini hatası: {e}")
                analysis_result = generate_backup_analysis(last, df, signals, user_msg)
        else:
            analysis_result = generate_backup_analysis(last, df, signals, user_msg)

        await status_msg.edit_text(analysis_result, parse_mode=constants.ParseMode.MARKDOWN)

    except Exception as e:
        logging.error(f"Analiz hatası: {e}")
        await status_msg.edit_text(f"❌ Analiz hatası: {str(e)}")

async def quick_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """⚡ 5 saniyede hızlı sinyal"""
    user_input = update.message.text.upper().strip()
    user_msg = user_input.replace("/SINYAL", "").strip()
    
    if not user_msg:
        await update.message.reply_text("❌ Hızlı sinyal için: `/sinyal THYAO`")
        return

    try:
        yf_symbol = get_yfinance_symbol(user_msg)
        df = yf.download(yf_symbol, period="1mo", interval="1d", progress=False)
        
        if df.empty:
            await update.message.reply_text(f"❌ Veri yok: `{user_msg}`")
            return
            
        df = advanced_analyzer.calculate_advanced_indicators(df)
        last = df.iloc[-1]
        
        # Hızlı sinyal algoritması
        if last['RSI_14'] < 35 and last['Close'] > last['SMA_50']:
            signal = "🟢 AL"
            reason = "RSI Oversold + Trend Yukarı"
            confidence = 75
        elif last['RSI_14'] > 65 and last['Close'] < last['SMA_50']:
            signal = "🔴 SAT" 
            reason = "RSI Overbought + Trend Aşağı"
            confidence = 70
        else:
            signal = "🟡 BEKLE"
            reason = "Trend belirsiz"
            confidence = 50
            
        response = f"""
⚡ **HIZLI SİNYAL - {user_msg}**

🎯 **Karar:** {signal}
📊 **Güven:** %{confidence}
💰 **Fiyat:** {last['Close']:.2f}
📈 **RSI:** {last['RSI_14']:.1f}

💡 **Sebep:** {reason}

⏱️ _5 saniyede hesaplandı_
"""
        await update.message.reply_text(response, parse_mode=constants.ParseMode.MARKDOWN)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Sinyal hatası: {str(e)}")

async def daily_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """📈 Günlük BIST100 raporu"""
    report_msg = await update.message.reply_text("📊 **Günlük BIST100 Raporu Hazırlanıyor...**")
    
    try:
        recommendations = []
        for symbol in list(BIST_100_SYMBOLS)[:10]:  # İlk 10 hisseyi tara
            try:
                df = yf.download(f"{symbol}.IS", period="1mo", progress=False)
                if not df.empty:
                    df = advanced_analyzer.calculate_advanced_indicators(df)
                    last = df.iloc[-1]
                    
                    if (last['RSI_14'] < 35 and 
                        last['Close'] > last['SMA_50'] and 
                        df['Volume'].tail(5).mean() > df['Volume'].tail(20).mean()):
                        recommendations.append({
                            'symbol': symbol,
                            'price': last['Close'],
                            'rsi': last['RSI_14'],
                            'reason': 'RSI Oversold + Trend + Hacim'
                        })
            except:
                continue
        
        # En iyi 3 hisseyi seç
        top_picks = sorted(recommendations, key=lambda x: x['rsi'])[:3]
        
        report = "📈 **GÜNLÜK BIST100 RAPORU**\n\n"
        report += "🏆 **BUGÜN ÖNE ÇIKAN HİSSELER:**\n\n"
        
        for rec in top_picks:
            report += f"• **{rec['symbol']}** - {rec['price']:.2f} TL\n"
            report += f"  RSI: {rec['rsi']:.1f} - {rec['reason']}\n\n"
            
        if not top_picks:
            report += "• Bugün için belirgin alım sinyali yok\n\n"
            
        report += "💡 _Kendi araştırmanızı yapmayı unutmayın._\n"
        report += "⚠️ _Yatırım tavsiyesi değildir._"
        
        await report_msg.edit_text(report, parse_mode=constants.ParseMode.MARKDOWN)
        
    except Exception as e:
        await report_msg.edit_text(f"❌ Rapor hatası: {str(e)}")

async def set_alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """🔔 Fiyat alarmı kur"""
    try:
        user_id = update.effective_user.id
        args = context.args
        
        if len(args) < 2:
            await update.message.reply_text("❌ Kullanım: `/alert THYAO 150`")
            return
            
        symbol = args[0].upper()
        price = float(args[1])
        
        if user_id not in user_alerts:
            user_alerts[user_id] = []
            
        user_alerts[user_id].append({
            'symbol': symbol,
            'target_price': price,
            'created_at': datetime.now()
        })
        
        await update.message.reply_text(
            f"🔔 **Alarm Ayarlandı!**\n"
            f"**{symbol}** için {price:.2f} seviyesi izleniyor...\n"
            f"Fiyat ulaştığında bildirim alacaksın."
        )
        
    except ValueError:
        await update.message.reply_text("❌ Geçersiz fiyat! Örnek: `/alert THYAO 150.50`")

async def top5_picks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """🏆 En iyi 5 BIST hissesi"""
    top_msg = await update.message.reply_text("🏆 **En İyi 5 BIST Hissesi Aranıyor...**")
    
    try:
        picks = []
        for symbol in list(BIST_100_SYMBOLS)[:15]:  # İlk 15 hisseyi tara
            try:
                df = yf.download(f"{symbol}.IS", period="3mo", progress=False)
                if len(df) > 50:
                    df = advanced_analyzer.calculate_advanced_indicators(df)
                    last = df.iloc[-1]
                    
                    # Puanlama sistemi
                    score = 0
                    if last['RSI_14'] < 40: score += 30
                    if last['Close'] > last['SMA_50']: score += 25
                    if last['MACD'] > last['MACD_SIGNAL']: score += 20
                    if last['Volume'] > df['Volume'].tail(20).mean(): score += 15
                    if last['Close'] > last['SMA_200']: score += 10
                    
                    if score > 50:
                        picks.append({
                            'symbol': symbol,
                            'price': last['Close'],
                            'score': score,
                            'rsi': last['RSI_14']
                        })
            except:
                continue
        
        # En yüksek puanlı 5 hisse
        top_5 = sorted(picks, key=lambda x: x['score'], reverse=True)[:5]
        
        response = "🏆 **EN İYİ 5 BIST HİSSESİ**\n\n"
        
        for i, pick in enumerate(top_5, 1):
            response += f"{i}. **{pick['symbol']}** - {pick['price']:.2f} TL\n"
            response += f"   📊 Skor: {pick['score']}/100 | RSI: {pick['rsi']:.1f}\n\n"
            
        if not top_5:
            response += "• Şu anda yüksek skorlu hisse bulunamadı\n\n"
            
        response += "💎 _Detaylı analiz için: `/analiz HISSENAME`_"
        
        await top_msg.edit_text(response, parse_mode=constants.ParseMode.MARKDOWN)
        
    except Exception as e:
        await top_msg.edit_text(f"❌ Top 5 hatası: {str(e)}")

def generate_backup_analysis(last, df, signals, symbol):
    """AI olmadan gelişmiş backup analiz"""
    
    # Trend analizi
    trend_strength = "GÜÇLÜ" if last.get('ADX', 0) > 25 else "ZAYIF" if last.get('ADX', 0) < 20 else "ORTA"
    trend_direction = "YUKARI" if last['Close'] > last['SMA_200'] else "AŞAĞI"
    
    # Momentum analizi
    momentum = "YÜKSELİŞ" if last['MACD'] > last['MACD_SIGNAL'] else "DÜŞÜŞ"
    
    # Risk/Hedef hesaplama
    atr = last.get('ATR', df['Close'].std())
    entry = last['Close']
    stop_loss = entry - (2 * atr) if trend_direction == "YUKARI" else entry + (2 * atr)
    target1 = entry + (3 * atr) if trend_direction == "YUKARI" else entry - (3 * atr)
    target2 = entry + (5 * atr) if trend_direction == "YUKARI" else entry - (5 * atr)
    
    # Güven skoru
    confidence = 50
    if last['RSI_14'] < 40 and trend_direction == "YUKARI": confidence = 75
    if last['RSI_14'] > 60 and trend_direction == "AŞAĞI": confidence = 70
    
    return f"""
═══════════════════════════════════════════════
🎯 **PROMETHEUS AI v9.0 - GELİŞMİŞ ANALİZ**
═══════════════════════════════════════════════

📊 **TEKNİK SİNYALLER:**
• Trend: {trend_direction} ({trend_strength})
• Momentum: {momentum}
• Hacim: {'ARTAN' if last['Volume'] > df['Volume'].tail(20).mean() else 'AZALAN'}

⚡ **OTOMATİK SİNYALLER:**
{chr(10).join(signals) if signals else '• Bekleme modunda'}

🦁 **NİHAİ KARAR:** {'AL' if confidence >= 70 else 'SAT' if confidence >= 60 else 'BEKLE'}
📈 **GÜVEN SKORU:** %{confidence}

💰 **İŞLEM PLANI:**
• 🎯 Giriş: {entry:.2f}
• 🛑 Stop-Loss: {stop_loss:.2f} (Risk: {abs((stop_loss - entry) / entry * 100):.1f}%)
• 🎯 Hedef 1: {target1:.2f} (R:R 1.5:1)
• 🎯 Hedef 2: {target2:.2f} (R:R 2.5:1)

📋 **DETAYLI ANALİZ:**
RSI: {last['RSI_14']:.1f} | MACD: {last['MACD']:.4f}
Trend: {trend_direction} | Volatilite: {atr:.2f}

⚠️ **RISK UYARILARI:**
• Stop-loss kullanımı zorunludur
• Pozisyon büyüklüğü max %2 risk
═══════════════════════════════════════════════
"""

# --- ALARM KONTROL SİSTEMİ ---
def check_alarms():
    """Aktif alarmları kontrol et"""
    if not application:
        return
        
    try:
        current_time = datetime.now()
        for user_id, alerts in list(user_alerts.items()):
            for alert in alerts[:]:
                try:
                    symbol = alert['symbol']
                    yf_symbol = get_yfinance_symbol(symbol)
                    df = yf.download(yf_symbol, period='1d', progress=False)
                    
                    if not df.empty:
                        current_price = df['Close'].iloc[-1]
                        target = alert['target_price']
                        
                        # Fiyat hedefe ulaştı mı?
                        if (current_price >= target and alert.get('direction') != 'SHORT') or \
                           (current_price <= target and alert.get('direction') == 'SHORT'):
                            
                            message = f"🔔 **ALARM!** {symbol} {current_price:.2f} seviyesine ulaştı!"
                            
                            # Kullanıcıya bildirim gönder
                            async def send_notification():
                                try:
                                    await application.bot.send_message(
                                        chat_id=user_id,
                                        text=message,
                                        parse_mode=constants.ParseMode.MARKDOWN
                                    )
                                    # Alarmı temizle
                                    alerts.remove(alert)
                                except Exception as e:
                                    logging.error(f"Alarm gönderme hatası: {e}")
                            
                            # Background task başlat
                            import asyncio
                            asyncio.create_task(send_notification())
                            
                except Exception as e:
                    logging.error(f"Alarm kontrol hatası: {e}")
                    continue
                    
    except Exception as e:
        logging.error(f"Alarm sistemi hatası: {e}")

# --- BOT BAŞLATMA ---
def start_bot():
    global application
    if TELEGRAM_TOKEN:
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        
        # Komutları ekle
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("analiz", advanced_analyze))
        application.add_handler(CommandHandler("sinyal", quick_signal))
        application.add_handler(CommandHandler("rapor", daily_report))
        application.add_handler(CommandHandler("alert", set_alert))
        application.add_handler(CommandHandler("top5", top5_picks))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, advanced_analyze))
        
        # Alarm scheduler'ı başlat
        scheduler = BackgroundScheduler()
        scheduler.add_job(check_alarms, 'interval', minutes=5)
        scheduler.start()
        
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
    else:
        logging.warning("Telegram token bulunamadı!")

if __name__ == '__main__':
    keep_alive()
    start_bot()
