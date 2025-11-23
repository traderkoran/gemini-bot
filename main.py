import logging
import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from telegram import Update, constants
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import google.generativeai as genai
from flask import Flask
from threading import Thread

# --- API AYARLARI ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Gemini Modelini Başlat - GÜNCEL MODEL
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        # GÜNCEL MODEL İSMİ
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
    return "🦁 Prometheus Bot Aktif!"

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()

# --- PROMETHEUS BEYNİ ---
SYSTEM_PROMPT = """
SEN: PROMETHEUS AI v7.3 (Yatırım Danışmanı).
KİMLİK: Duygusuz, profesyonel fon yöneticisi.
GÖREV: Verilen teknik verilere göre AL / SAT / BEKLE kararı ver.

ANALİZ KURALLARI:
1. Trend (SMA 200) yönüne bak.
2. RSI (14) aşırı alım/satım bölgesinde mi?
3. ATR değerine göre mantıklı bir Stop-Loss belirle.
4. Hacim (Volume) fiyatı destekliyor mu?

ÇIKTI FORMATI:
---------------------------------------------------
🦁 **PROMETHEUS KARARI:** [AL / SAT / BEKLE]
Güven: %[0-100]

📉 **İŞLEM PLANI:**
• Giriş: [Güncel Fiyat]
• 🛑 Stop-Loss: [Fiyat] (ATR bazlı)
• 🎯 Hedef: [Fiyat] (R:R 1:2)

🧠 **ANALİZ:**
[Teknik verileri ve hacmi yorumla. Akıllı para ne yapıyor?]

⚠️ _Risk Notu: Stopsuz işlem kumardır._
---------------------------------------------------
"""

def calculate_technicals(df):
    """Teknik indikatörleri hesaplar"""
    try:
        # Yfinance MultiIndex Düzeltmesi
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # RSI hesapla
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # ATR hesapla - pandas_ta formatına uygun
        atr_result = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        if atr_result is not None:
            df['ATR'] = atr_result
        else:
            df['ATR'] = 0
            
        # MACD hesapla
        macd_result = ta.macd(df['Close'])
        if macd_result is not None:
            df['MACD'] = macd_result['MACD_12_26_9']
            df['MACD_SIGNAL'] = macd_result['MACDs_12_26_9']
        else:
            df['MACD'] = 0
            df['MACD_SIGNAL'] = 0
        
        # Bollinger Bands
        bb_result = ta.bbands(df['Close'], length=20)
        if bb_result is not None:
            df['BB_UPPER'] = bb_result['BBU_20_2.0']
            df['BB_LOWER'] = bb_result['BBL_20_2.0']
        else:
            df['BB_UPPER'] = df['Close']
            df['BB_LOWER'] = df['Close']
        
        # SMA 200
        if len(df) >= 200:
            sma_result = ta.sma(df['Close'], length=200)
            if sma_result is not None:
                df['SMA_200'] = sma_result
            else:
                df['SMA_200'] = df['Close']
        else:
            df['SMA_200'] = df['Close']

        # Volume analizi
        volume_sma = ta.sma(df['Volume'], length=20)
        if volume_sma is not None:
            df['VOL_SMA'] = volume_sma
            df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA'].replace(0, 1)
        else:
            df['VOL_SMA'] = df['Volume']
            df['VOL_RATIO'] = 1
        
        return df
    except Exception as e:
        logging.error(f"İndikatör Hatası: {e}")
        return df

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = """
    🦁 **PROMETHEUS DEVREDE**
    
    Bana bir sembol yaz, analiz edeyim.
    Örnek: `BTC`, `ETH`, `THYAO`, `ALTIN`
    
    Komutlar:
    /start - Botu başlat
    /analiz [sembol] - Teknik analiz yap
    """
    await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.upper().strip()
    
    # Komutları temizle
    user_msg = user_input.replace("/ANALIZ", "").replace("/ANALIZ ", "").strip()
    
    if not user_msg:
        await update.message.reply_text("Hangi varlık? Örn: `BTC` veya `THYAO`")
        return

    status_msg = await update.message.reply_text(f"🔍 **{user_msg}** verileri taranıyor...", parse_mode=constants.ParseMode.MARKDOWN)

    # Sembol dönüşümü
    yf_symbol = user_msg
    if user_msg in ["BTC", "ETH", "SOL", "AVAX", "XRP", "DOGE", "ADA", "DOT"]:
        yf_symbol = f"{user_msg}-USD"
    elif user_msg == "ALTIN":
        yf_symbol = "GC=F"
    elif user_msg == "GÜMÜŞ":
        yf_symbol = "SI=F"
    elif user_msg == "PETROL":
        yf_symbol = "CL=F"
    elif ".IS" not in user_msg and "=" not in user_msg and len(user_msg) <= 5:
        yf_symbol = f"{user_msg}.IS"

    try:
        # Veri Çekme
        df = yf.download(yf_symbol, period="6mo", interval="1d", progress=False, auto_adjust=True)
        
        if df.empty:
            # Alternatif sembol dene
            df = yf.download(user_msg, period="6mo", interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                await status_msg.edit_text(f"❌ Veri bulunamadı: `{user_msg}`")
                return

        df = calculate_technicals(df)
        
        if df.empty:
            await status_msg.edit_text(f"❌ Analiz için yeterli veri yok: `{user_msg}`")
            return
            
        last = df.iloc[-1]
        
        # Trend analizi
        if 'SMA_200' in df and not pd.isna(last['SMA_200']):
            trend = "YÜKSELİŞ" if last['Close'] > last['SMA_200'] else "DÜŞÜŞ"
            trend_strength = abs((last['Close'] - last['SMA_200']) / last['SMA_200'] * 100)
        else:
            trend = "NÖTR"
            trend_strength = 0

        # Değerleri formatla
        def safe_get(col, default="N/A"):
            try:
                if col in last and not pd.isna(last[col]):
                    if col in ['RSI', 'MACD', 'MACD_SIGNAL', 'VOL_RATIO']:
                        return f"{last[col]:.2f}"
                    elif col in ['Close', 'ATR', 'BB_UPPER', 'BB_LOWER', 'SMA_200']:
                        return f"{last[col]:.2f}"
                return default
            except:
                return default

        # Gemini AI analizi
        if model:
            try:
                prompt = f"""
                {SYSTEM_PROMPT}
                
                VARLIK: {user_msg} ({yf_symbol})
                Fiyat: {safe_get('Close')}
                RSI: {safe_get('RSI')}
                MACD: {safe_get('MACD')}
                Trend: {trend} (%{trend_strength:.1f})
                Bollinger Üst: {safe_get('BB_UPPER')} / Alt: {safe_get('BB_LOWER')}
                ATR: {safe_get('ATR')}
                Hacim Oranı: {safe_get('VOL_RATIO')}x
                
                Karar ver:
                """
                
                response = model.generate_content(prompt)
                analysis_result = response.text
                
            except Exception as e:
                logging.error(f"Gemini hatası: {e}")
                # Gemini olmadan basit analiz
                current_price = safe_get('Close', '0')
                atr_val = float(safe_get('ATR', '0'))
                rsi_val = float(safe_get('RSI', '50'))
                
                if rsi_val < 30 and trend == "YÜKSELİŞ":
                    signal = "AL"
                    confidence = "75"
                elif rsi_val > 70 and trend == "DÜŞÜŞ":
                    signal = "SAT"
                    confidence = "70"
                else:
                    signal = "BEKLE"
                    confidence = "60"
                    
                stop_loss = float(current_price) - (2 * atr_val) if signal == "AL" else float(current_price) + (2 * atr_val)
                target = float(current_price) + (4 * atr_val) if signal == "AL" else float(current_price) - (4 * atr_val)
                
                analysis_result = f"""
---------------------------------------------------
🦁 **PROMETHEUS KARARI:** {signal}
Güven: %{confidence}

📉 **İŞLEM PLANI:**
• Giriş: {current_price}
• 🛑 Stop-Loss: {stop_loss:.2f} (ATR bazlı)
• 🎯 Hedef: {target:.2f} (R:R 1:2)

🧠 **ANALİZ:**
RSI: {rsi_val:.1f}, Trend: {trend}
Temel teknik göstergelere göre karar verildi.

⚠️ _Risk Notu: Stopsuz işlem kumardır._
---------------------------------------------------
                """
        else:
            # API anahtarı yoksa basit analiz
            analysis_result = "⚠️ Gemini AI anahtarı eksik. Temel analiz yapılamıyor."

        await status_msg.edit_text(analysis_result, parse_mode=constants.ParseMode.MARKDOWN)

    except Exception as e:
        logging.error(f"Genel hata: {e}")
        await status_msg.edit_text(f"⚠️ İşlem hatası: {str(e)}")

# Telegram Bot Başlatma
def start_bot():
    if TELEGRAM_TOKEN:
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("analiz", analyze))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze))
        
        # Webhook kullanarak conflict hatasını önle
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    else:
        logging.warning("Telegram token bulunamadı!")

if __name__ == '__main__':
    keep_alive()
    start_bot()
