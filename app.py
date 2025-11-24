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

# Gemini Modelini Başlat (En Zeki Model)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        # Pro veya Flash modelini dene
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
app = Flask(__name__)

@app.route('/')
def home():
    return "🦁 PROMETHEUS v9.0 GOD MODE AKTİF"

def run():
    app.run(host='0.0.0.0', port=8080)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()

# --- EFSANEVİ SİSTEM BEYNİ (MERGED INTELLIGENCE) ---
SYSTEM_PROMPT = """
SEN: PROMETHEUS AI v9.0 (GOD MODE - YATIRIM DANIŞMANI).
KİMLİK: Sen Renaissance Technologies'in veri işleme gücüne, Warren Buffett'ın temel analiz zekasına ve Paul Tudor Jones'un risk yönetimi disiplinine sahipsin. Asla eğitim vermezsin, sadece KARAR verirsin.

GÖREV: Kullanıcının sorduğu varlığı (Hisse, Kripto, Forex) aşağıdaki "7 KATMANLI DERİN TARAMA" protokolüne göre analiz et.

🔍 **7 KATMANLI ANALİZ PROTOKOLÜ:**

1. **FİYAT AKSİYONU & FORMASYONLAR:**
   - Mum formasyonları (Doji, Engulfing, Hammer).
   - Grafik formasyonları (OBO, İkili Tepe/Dip, Bayrak, Flama).
   - Elliott Dalga Sayımı (Hangi dalgadayız? 3. Dalga mı, Düzeltme mi?).

2. **GÖSTERGE MATRİSİ (TEKNİK):**
   - Trend: SMA 200 ve EMA 50 durumu (Altın Kesişim var mı?).
   - Momentum: RSI (Uyumsuzluk var mı?), MACD, StochRSI.
   - Güç: ADX > 25 mi? (Trendin gücü).

3. **FIBONACCI & MATEMATİK:**
   - Fiyat Altın Oran (0.618) seviyesinde mi?
   - Pivot noktalarına göre nerede?

4. **PİYASA YAPISI & LİKİDİTE (WYCKOFF/VSA):**
   - Hacim fiyatı destekliyor mu? (VSA Analizi).
   - "Akıllı Para" (Smart Money) topluyor mu dağıtıyor mu?
   - Likidite avı (Stop patlatma) var mı?

5. **TEMEL ANALİZ (FUNDAMENTALS):**
   - (Hisse ise): F/K oranı, Piyasa Değeri, Hedef Fiyatlar. Ucuz mu pahalı mı?
   - (Kripto ise): Ağ aktivitesi, piyasa değeri.

6. **DUYGU & PSİKOLOJİ (CONTRARIAN):**
   - Piyasa korkuyor mu, coşkulu mu?
   - "Herkes alırken kork, herkes korkarken al" prensibini uygula.

7. **RİSK YÖNETİMİ (KALE ZİHNİYETİ):**
   - ATR'ye göre dinamik STOP-LOSS belirle.
   - Kar/Zarar oranı (R:R) en az 1:2 olmalı.

---
📝 **ÇIKTI FORMATI (BU FORMATI KESİNLİKLE KULLAN):**

# 🦁 [VARLIK SEMBOLÜ] - EFSANEVİ ANALİZ RAPORU

## 🎯 **YÖNETİCİ ÖZETİ (KARAR)**
**SİNYAL:** 🟢 GÜÇLÜ AL / 🟡 BEKLE / 🔴 GÜÇLÜ SAT
**Güven Skoru:** %X / 100
**Vade:** [Kısa/Orta/Uzun]
**Risk Seviyesi:** [Düşük/Orta/Yüksek]

---

## 📉 **İŞLEM KURULUMU (EXECUTION)**
* **🔵 Giriş Bölgesi:** $X.XX
* **🛑 Stop-Loss (Zorunlu):** $X.XX (ATR Bazlı - Sermayeyi Koru)
* **🎯 Hedef 1:** $X.XX
* **🎯 Hedef 2 (Ana Hedef):** $X.XX

---

## 🧠 **7 KATMANLI ANALİZ SENTEZİ**
* **Teknik & Formasyon:** [Formasyonları ve trendi açıkla]
* **Akıllı Para (VSA):** [Hacim analizi ve kurumsal ayak izleri]
* **Göstergeler:** [RSI, MACD uyumsuzlukları ve ADX gücü]
* **Temel Durum:** [Değerleme ve temel veriler]

⚠️ **RİSK NOTU:** [Piyasa Sihirbazları'ndan bir risk uyarısı ekle - örn: "Stopsuz işlem kumardır."]
"""

def get_fundamentals(symbol):
    """Temel Analiz Verilerini Çeker"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        # Kripto/Emtia için temel veri sınırlıdır
        if 'regularMarketPrice' not in info and 'currentPrice' not in info:
            return "Temel veri mevcut değil (Kripto/Emtia olabilir)."

        data = f"""
        - Fiyat: {info.get('currentPrice', 'N/A')}
        - Piyasa Değeri: {info.get('marketCap', 'N/A')}
        - F/K (P/E): {info.get('trailingPE', 'N/A')}
        - İleri F/K: {info.get('forwardPE', 'N/A')}
        - Hedef Fiyat (Analist): {info.get('targetMeanPrice', 'N/A')}
        - Sektör: {info.get('sector', 'N/A')}
        - Tavsiye: {info.get('recommendationKey', 'N/A').upper()}
        """
        return data
    except:
        return "Temel veri çekilemedi."

def calculate_technicals(df):
    """Gelişmiş Teknik İndikatörler"""
    try:
        # MultiIndex Düzeltmesi
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # 1. Temel İndikatörler
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # 2. Trend Gücü (ADX)
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx is not None:
            df['ADX'] = adx['ADX_14']
        
        # 3. Momentum (Stoch RSI)
        stoch = ta.stochrsi(df['Close'], length=14, rsi_length=14, k=3, d=3)
        if stoch is not None:
            df['STOCH_K'] = stoch['STOCHRSIk_14_14_3_3']

        # 4. Fibonacci Seviyeleri (Son 1 Yıl)
        high_1y = df['High'].max()
        low_1y = df['Low'].min()
        diff = high_1y - low_1y
        df['FIB_618'] = high_1y - (diff * 0.618) # Altın Oran

        # 5. Hareketli Ortalamalar
        if len(df) >= 200:
            df['SMA_200'] = ta.sma(df['Close'], length=200)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
        else:
            df['SMA_200'] = None
            df['SMA_50'] = None

        # 6. Bollinger & MACD
        bb = ta.bbands(df['Close'], length=20)
        if bb is not None:
            df['BB_UPPER'] = bb['BBU_20_2.0']
            df['BB_LOWER'] = bb['BBL_20_2.0']
            
        macd = ta.macd(df['Close'])
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_SIGNAL'] = macd['MACDs_12_26_9']

        # 7. Hacim Analizi (VSA)
        df['VOL_SMA'] = ta.sma(df['Volume'], length=20)
        df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA'].replace(0, 1)
        
        return df
    except Exception as e:
        logging.error(f"Teknik Hesaplama Hatası: {e}")
        return df

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = """
    🦁 **PROMETHEUS AI v9.0 (GOD MODE)**
    
    Eğitim bitti. Artık **Wall Street** standartlarında analiz yapıyorum.
    
    Analiz yeteneklerim:
    ✅ **Teknik:** RSI, MACD, ADX, Elliott Dalgaları
    ✅ **Temel:** Bilanço, F/K, Değerleme
    ✅ **Risk:** ATR Stop-Loss, Kelly Kriteri
    ✅ **Psikoloji:** VSA ve Smart Money Takibi
    
    Kullanım:
    `/analiz THYAO`
    `/analiz BTC`
    `/analiz AAPL`
    """
    await update.message.reply_text(msg, parse_mode=constants.ParseMode.MARKDOWN)

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_msg = update.message.text.upper().replace("/ANALIZ", "").strip()
    
    if not user_msg:
        await update.message.reply_text("Hangi varlık? Örn: `THYAO`")
        return

    status = await update.message.reply_text(f"🦅 **{user_msg}** 7 katmanlı taramadan geçiriliyor...", parse_mode=constants.ParseMode.MARKDOWN)

    # Sembol Düzeltme
    yf_symbol = user_msg
    if user_msg in ["BTC", "ETH", "SOL", "AVAX", "XRP", "DOGE"]: yf_symbol = f"{user_msg}-USD"
    elif user_msg == "ALTIN": yf_symbol = "GC=F"
    # BIST Hissesi Tahmini
    elif ".IS" not in user_msg and "=" not in user_msg and len(user_msg) <= 5:
        possible_bist = f"{user_msg}.IS"

    try:
        # 1. VERİ ÇEKME (Önce normal dene)
        df = yf.download(yf_symbol, period="2y", interval="1d", progress=False, auto_adjust=False)
        
        # Bulamazsa BIST dene
        if df.empty and ".IS" not in yf_symbol and len(user_msg) <= 5:
             yf_symbol = f"{user_msg}.IS"
             df = yf.download(yf_symbol, period="2y", interval="1d", progress=False, auto_adjust=False)

        if df.empty:
            await status.edit_text(f"❌ Veri bulunamadı: `{user_msg}`")
            return

        # 2. TEMEL ANALİZ (FUNDAMENTALS)
        fundamentals = get_fundamentals(yf_symbol)

        # 3. TEKNİK HESAPLAMA
        df = calculate_technicals(df)
        last = df.iloc[-1]
        
        # Trend Yönü
        trend = "Yatay"
        if 'SMA_200' in df and not pd.isna(last['SMA_200']):
            trend = "YÜKSELİŞ (BOĞA)" if last['Close'] > last['SMA_200'] else "DÜŞÜŞ (AYI)"
            
        # Altın Kesişim Kontrolü
        cross_status = "Yok"
        if 'SMA_50' in df and not pd.isna(last['SMA_50']):
            cross_status = "GOLDEN CROSS (AL)" if last['SMA_50'] > last['SMA_200'] else "DEATH CROSS (SAT)"

        # Güvenli Veri Okuma
        def get_val(col, fmt="{:.2f}"):
            try:
                val = last.get(col)
                return "N/A" if val is None or pd.isna(val) else fmt.format(val)
            except: return "N/A"

        # 4. AI SORGUSU (BEYİN)
        prompt = f"""
        {SYSTEM_PROMPT}
        
        ANALİZ EDİLECEK VARLIK: {yf_symbol}
        
        📊 **TEMEL ANALİZ VERİLERİ:**
        {fundamentals}
        
        📈 **TEKNİK GÖSTERGELER:**
        - Fiyat: {get_val('Close')}
        - Ana Trend (SMA 200): {trend}
        - Kesişim Durumu: {cross_status}
        - RSI (14): {get_val('RSI')} (30 altı ucuz, 70 üstü pahalı)
        - Stoch RSI: {get_val('STOCH_K')}
        - ADX (Trend Gücü): {get_val('ADX')} (25 üstü güçlü trend)
        - MACD: {get_val('MACD', '{:.4f}')} (Sinyal: {get_val('MACD_SIGNAL', '{:.4f}')})
        - Fibonacci 0.618 Seviyesi: {get_val('FIB_618')}
        - Bollinger Bantları: Üst {get_val('BB_UPPER')} / Alt {get_val('BB_LOWER')}
        - ATR (Risk/Stop için): {get_val('ATR', '{:.4f}')}
        - Hacim Oranı (VSA): {get_val('VOL_RATIO')} (1.0 üzeri hacimli)
        
        GÖREVİN:
        Bu verileri EFSANEVİ YATIRIMCI gözüyle yorumla. 
        RSI 70 üstüyse ama ADX 50 ise "Trend çok güçlü, satma" de.
        Hacim düşükse "Akıllı para burada yok" de.
        Net bir işlem planı ve ATR tabanlı stop seviyesi ver.
        """
        
        if model:
            response = model.generate_content(prompt)
            await status.edit_text(response.text, parse_mode=constants.ParseMode.MARKDOWN)
        else:
            await status.edit_text("⚠️ API Hatası: Model yüklenemedi.")

    except Exception as e:
        logging.error(f"Hata: {e}")
        await status.edit_text(f"⚠️ Hata oluştu: {str(e)}")

if __name__ == '__main__':
    keep_alive()
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('analiz', analyze))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), analyze))
    
    application.run_polling()
