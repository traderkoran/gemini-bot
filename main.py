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
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- WEB SUNUCUSU (Render için) ---
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
SEN: PROMETHEUS AI v7.1 (Yatırım Danışmanı).
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
    """Teknik indikatörleri hesaplar (Hata korumalı)"""
    try:
        # Temel indikatörler
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # MACD
        macd = ta.macd(df['Close'])
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_SIGNAL'] = macd['MACDs_12_26_9']
        
        # Bollinger
        bb = ta.bbands(df['Close'], length=20)
        if bb is not None:
            df['BB_UPPER'] = bb['BBU_20_2.0']
            df['BB_LOWER'] = bb['BBL_20_2.0']
        
        # SMA 200 (Veri yeterliyse hesapla)
        if len(df) >= 200:
            df['SMA_200'] = ta.sma(df['Close'], length=200)
        else:
            # Veri azsa SMA 50 kullan veya None ata
            df['SMA_200'] = None 

        # Hacim
        df['VOL_SMA'] = ta.sma(df['Volume'], length=20)
        df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA']
        
        return df
    except Exception as e:
        logging.error(f"İndikatör Hatası: {e}")
        return df

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = """
    🦁 **PROMETHEUS DEVREDE**
    
    Bana bir sembol yaz, analiz edeyim.
    
    Örnekler:
    `BTC`
    `ETH`
    `THYAO`
    `ASELS`
    `ALTIN`
    """
    await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_msg = update.message.text.upper().replace("/ANALIZ", "").strip()
    
    if not user_msg:
        await update.message.reply_text("Hangi varlık? Örn: `BTC`")
        return

    status = await update.message.reply_text(f"🔍 **{user_msg}** verileri taranıyor...", parse_mode=constants.ParseMode.MARKDOWN)

    # --- AKILLI SEMBOL BULUCU ---
    yf_symbol = user_msg
    
    # Kripto düzeltmesi
    if user_msg in ["BTC", "ETH", "SOL", "AVAX", "XRP", "DOGE", "PEPE"]: 
        yf_symbol = f"{user_msg}-USD"
    elif user_msg == "ALTIN": 
        yf_symbol = "GC=F"
    
    # BIST Düzeltmesi (Eğer kripto/altın değilse ve .IS yoksa sonuna eklemeyi dene)
    elif ".IS" not in user_msg and "=" not in user_msg and len(user_msg) <= 5:
        # Varsayılan olarak BIST hissesi varsayıp .IS ekleyelim, değilse aşağıda kontrol edeceğiz
        possible_bist = f"{user_msg}.IS"
        
    try:
        # 1. VERİ ÇEKME (Önce normal dene)
        # period="2y" yaptık ki SMA 200 hesaplanabilsin
        df = yf.download(yf_symbol, period="2y", interval="1d", progress=False)
        
        # Eğer veri boş geldiyse ve BIST olma ihtimali varsa .IS ekleyip tekrar dene
        if df.empty and ".IS" not in yf_symbol and len(user_msg) <= 5:
             yf_symbol = f"{user_msg}.IS"
             df = yf.download(yf_symbol, period="2y", interval="1d", progress=False)

        if df.empty:
            await status.edit_text(f"❌ Veri bulunamadı: `{user_msg}`\nLütfen sembolü doğru yazdığından emin ol. (Örn: THYAO, BTC, USDTRY=X)")
            return

        # 2. HESAPLAMA
        df = calculate_technicals(df)
        last = df.iloc[-1]
        
        # Trend Kontrolü (Hata vermemesi için)
        if 'SMA_200' in df and not pd.isna(last['SMA_200']):
            trend = "YÜKSELİŞ (SMA200 Üstü)" if last['Close'] > last['SMA_200'] else "DÜŞÜŞ (SMA200 Altı)"
        else:
            trend = "Bilinmiyor (Veri Yetersiz)"

        # Güvenli Veri Çekme (NaN hatası olmasın diye)
        def get_val(col, fmt="{:.2f}"):
            try:
                val = last[col]
                if pd.isna(val): return "N/A"
                return fmt.format(val)
            except: return "N/A"

        # 3. AI SORGUSU
        prompt = f"""
        {SYSTEM_PROMPT}
        
        ANALİZ EDİLECEK VARLIK: {yf_symbol}
        
        TEKNİK VERİLER:
        - Fiyat: {get_val('Close')}
        - RSI (14): {get_val('RSI')}
        - MACD: {get_val('MACD', '{:.4f}')}
        - Trend Durumu: {trend}
        - Bollinger Bantları: Üst {get_val('BB_UPPER')} / Alt {get_val('BB_LOWER')}
        - ATR (Volatilite): {get_val('ATR', '{:.4f}')}
        - Hacim Oranı: {get_val('VOL_RATIO')} (1.0 üstü hacimli)
        
        Bu verilere dayanarak profesyonel kararını ver.
        """
        
        if model:
            response = model.generate_content(prompt)
            await status.edit_text(response.text, parse_mode=constants.ParseMode.MARKDOWN)
        else:
            await status.edit_text("⚠️ API Anahtarı Hatası. Render ayarlarını kontrol et.")

    except Exception as e:
        await status.edit_text(f"⚠️ Hata oluştu: {str(e)}")

if __name__ == '__main__':
    keep_alive()
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('analiz', analyze))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), analyze))
    
    application.run_polling()
