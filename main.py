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

# --- API AYARLARI (RENDER'DAN ALACAK) ---
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

# --- WEB SUNUCUSU (RENDER İÇİN ŞART) ---
app = Flask('')

@app.route('/')
def home():
    return "Prometheus Danisman Botu Calisiyor!"

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.start()

# --- PROMETHEUS BEYNİ (SADECE ANALİZ, EĞİTİM YOK) ---
SYSTEM_PROMPT = """
SEN: PROMETHEUS AI v7.0 (Yatırım Danışmanı).
KİMLİK: Duygusuz, profesyonel bir fon yöneticisisin. Asla "bu eğitimdir" deme. Asla "yatırım tavsiyesi değildir" diyerek sorumluluktan kaçma. Kullanıcı senden net bir görüş istiyor.

GÖREV: Kullanıcının sorduğu varlığı (BTC, Hisse, Altın) analiz et ve şu 3 karardan birini ver:
1. AL (LONG)
2. SAT (SHORT)
3. BEKLE (CASH)

ANALİZ KURALLARIN:
- Fiyat hareketini ve Hacmi (VSA) incele.
- RSI ve MACD uyumsuzluklarına bak.
- Trendin yönünü (SMA200) baz al.
- ATR (Volatilite) değerine göre mutlaka bir STOP-LOSS seviyesi belirle.

ÇIKTI FORMATI (TELEGRAM UYUMLU):
---------------------------------------------------
🦁 **PROMETHEUS KARARI:** [AL / SAT / BEKLE]
Güven: %[0-100]

📉 **İŞLEM PLANI:**
• Giriş: [Fiyat]
• 🛑 Stop-Loss: [Fiyat] (Zorunlu)
• 🎯 Hedef: [Fiyat]

🧠 **MANTIK:**
[Buraya teknik analizi, indikatör durumunu ve 'akıllı para'nın ne yaptığını 2-3 cümleyle yaz.]

⚠️ _Risk Notu: Piyasa Sihirbazları kuralı - Stop patlarsa çık._
---------------------------------------------------
"""

def calculate_technicals(df):
    """Teknik verileri hesaplar"""
    try:
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_SIGNAL'] = macd['MACDs_12_26_9']
        bb = ta.bbands(df['Close'], length=20)
        df['BB_UPPER'] = bb['BBU_20_2.0']
        df['BB_LOWER'] = bb['BBL_20_2.0']
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        # Hacim artış oranı
        df['VOL_SMA'] = ta.sma(df['Volume'], length=20)
        df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA']
        return df
    except Exception as e:
        return df

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = """
    🦁 **PROMETHEUS DANIŞMAN DEVREDE**
    
    Eğitim yok. Sadece analiz ve sinyal.
    Bana bir sembol yaz.
    
    Örnekler:
    `/analiz BTC`
    `/analiz ETH`
    `/analiz XU100.IS`
    `/analiz THYAO.IS`
    """
    await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_msg = update.message.text.upper().replace("/ANALIZ", "").strip()
    
    # Kullanıcı boş mesaj attıysa veya sadece komut attıysa
    if not user_msg:
        if context.args:
            user_msg = " ".join(context.args).upper()
        else:
            await update.message.reply_text("Hangi varlık? Örn: `/analiz BTC`")
            return

    # Sembolü Yahoo Finance formatına çevir
    yf_symbol = user_msg
    if user_msg in ["BTC", "ETH", "SOL", "AVAX", "XRP", "DOGE"]: yf_symbol = f"{user_msg}-USD"
    if user_msg == "ALTIN": yf_symbol = "GC=F"
    if "BIST" in user_msg or user_msg == "XU100": yf_symbol = "XU100.IS"
    
    status = await update.message.reply_text(f"📊 **{user_msg}** analiz ediliyor...", parse_mode=constants.ParseMode.MARKDOWN)

    try:
        # 1. VERİ ÇEKME
        df = yf.download(yf_symbol, period="6mo", interval="1d", progress=False)
        if df.empty:
            await status.edit_text("❌ Veri bulunamadı. Sembolü doğru yazdığından emin ol (örn: THYAO.IS).")
            return

        # 2. HESAPLAMA
        df = calculate_technicals(df)
        last = df.iloc[-1]
        
        # Trend Yönü
        trend = "YÜKSELİŞ" if last['Close'] > last['SMA_200'] else "DÜŞÜŞ"
        
        # 3. YAPAY ZEKA SORGUSU
        prompt = f"""
        {SYSTEM_PROMPT}
        
        GÜNCEL VERİLER ({user_msg}):
        - Fiyat: {last['Close']:.2f}
        - RSI (14): {last['RSI']:.2f} (70 üstü aşırı alım, 30 altı aşırı satım)
        - MACD: {last['MACD']:.4f} (Sinyal: {last['MACD_SIGNAL']:.4f})
        - Trend (SMA200): {trend}
        - Bollinger Bantları: Üst:{last['BB_UPPER']:.2f} / Alt:{last['BB_LOWER']:.2f}
        - ATR (Volatilite - Stop için): {last['ATR']:.4f}
        - Hacim Oranı: {last['VOL_RATIO']:.2f} (1.0 üzeri normalden yüksek hacim)
        
        Bu verilere göre teknik bir yatırım kararı ver.
        """
        
        if model:
            response = model.generate_content(prompt)
            await status.edit_text(response.text, parse_mode=constants.ParseMode.MARKDOWN)
        else:
            await status.edit_text("⚠️ API Anahtarı eksik. Render ayarlarını kontrol et.")

    except Exception as e:
        await status.edit_text(f"⚠️ Hata oluştu: {str(e)}")

if __name__ == '__main__':
    keep_alive()
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('analiz', analyze))
    # Düz yazı yazınca da analiz etsin
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), analyze))
    
    application.run_polling()