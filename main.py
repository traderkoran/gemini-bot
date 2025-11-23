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

# Gemini Modelini Başlat
# Hata almamak için en standart model olan 'gemini-pro' veya 'gemini-1.5-flash' kullanıyoruz.
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        # Önce Flash'ı dene, olmazsa Pro'ya düş
        model = genai.GenerativeModel('gemini-1.5-flash')
    except:
        model = genai.GenerativeModel('gemini-pro')
else:
    model = None

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- WEB SUNUCUSU ---
app = Flask('')

@app.route('/')
def home():
    return "Prometheus Bot Aktif!"

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
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
            
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        macd = ta.macd(df['Close'])
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_SIGNAL'] = macd['MACDs_12_26_9']
        
        bb = ta.bbands(df['Close'], length=20)
        if bb is not None:
            df['BB_UPPER'] = bb['BBU_20_2.0']
            df['BB_LOWER'] = bb['BBL_20_2.0']
        
        if len(df) >= 200:
            df['SMA_200'] = ta.sma(df['Close'], length=200)
        else:
            df['SMA_200'] = None 

        df['VOL_SMA'] = ta.sma(df['Volume'], length=20)
        df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA'].replace(0, 1)
        
        return df
    except Exception as e:
        logging.error(f"İndikatör Hatası: {e}")
        return df

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = """
    🦁 **PROMETHEUS DEVREDE**
    
    Bana bir sembol yaz, analiz edeyim.
    Örnek: `BTC`, `ETH`, `THYAO`, `ALTIN`
    """
    await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_msg = update.message.text.upper().replace("/ANALIZ", "").strip()
    
    if not user_msg:
        await update.message.reply_text("Hangi varlık? Örn: `BTC`")
        return

    status = await update.message.reply_text(f"🔍 **{user_msg}** verileri taranıyor...", parse_mode=constants.ParseMode.MARKDOWN)

    yf_symbol = user_msg
    if user_msg in ["BTC", "ETH", "SOL", "AVAX", "XRP", "DOGE"]: yf_symbol = f"{user_msg}-USD"
    elif user_msg == "ALTIN": yf_symbol = "GC=F"
    elif ".IS" not in user_msg and "=" not in user_msg and len(user_msg) <= 5:
        yf_symbol = f"{user_msg}.IS" # Varsayılan BIST varsay

    try:
        # Veri Çekme (Hata yakalamalı)
        try:
            df = yf.download(yf_symbol, period="2y", interval="1d", progress=False, auto_adjust=False)
        except:
             # Eğer hata verirse .IS'siz dene (Belki ABD hissesidir)
             df = yf.download(user_msg, period="2y", interval="1d", progress=False, auto_adjust=False)

        if df.empty:
            await status.edit_text(f"❌ Veri bulunamadı: `{user_msg}`")
            return

        df = calculate_technicals(df)
        last = df.iloc[-1]
        
        if 'SMA_200' in df and not pd.isna(last['SMA_200']):
            trend = "YÜKSELİŞ" if last['Close'] > last['SMA_200'] else "DÜŞÜŞ"
        else:
            trend = "Bilinmiyor"

        def get_val(col):
            try:
                val = last[col]
                return "N/A" if pd.isna(val) else "{:.2f}".format(val)
            except: return "N/A"

        prompt = f"""
        {SYSTEM_PROMPT}
        VARLIK: {yf_symbol}
        Fiyat: {get_val('Close')}
        RSI: {get_val('RSI')}
        MACD: {get_val('MACD')}
        Trend: {trend}
        Bollinger: {get_val('BB_UPPER')} / {get_val('BB_LOWER')}
        ATR: {get_val('ATR')}
        Hacim Oranı: {get_val('VOL_RATIO')}
        Karar ver.
        """
        
        if model:
            # Hata korumalı Gemini isteği
            try:
                response = model.generate_content(prompt)
                await status.edit_text(response.text, parse_mode=constants.ParseMode.MARKDOWN)
            except Exception as e:
                 # Eğer 1.5-flash hata verirse kullanıcıya bildir
                 await status.edit_text(f"⚠️ Yapay Zeka Hatası: {str(e)}\nAPI Key'i kontrol et veya model desteklenmiyor.")
        else:
            await status.edit_text("⚠️ API Anahtarı eksik.")

    except Exception as e:
        await status.edit_text(f"⚠️ Hata: {str(e)}")

if __name__ == '__main__':
    keep_alive()
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('analiz', analyze))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), analyze))
    application.run_polling()
