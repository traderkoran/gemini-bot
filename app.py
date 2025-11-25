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
from datetime import datetime
import json

# === API AYARLARI ===
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
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
    return "🦁 PROMETHEUS AI v9.1 - ULTRA STABLE"

def run():
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run)
    t.daemon = True
    t.start()

# === AKILLI SEMBOL DÖNÜŞTÜRME ===
def smart_symbol_convert(symbol):
    """Önce doğrula, sonra dönüştür"""
    symbol = symbol.upper().strip()
    
    # Bilinen kripto/emtia haritası
    direct_map = {
        'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD',
        'AVAX': 'AVAX-USD', 'XRP': 'XRP-USD', 'DOGE': 'DOGE-USD',
        'ADA': 'ADA-USD', 'DOT': 'DOT-USD', 'LINK': 'LINK-USD',
        'MATIC': 'MATIC-USD', 'BNB': 'BNB-USD', 'UNI': 'UNI-USD',
        'ALTIN': 'GC=F', 'GOLD': 'GC=F',
        'GÜMÜŞ': 'SI=F', 'SILVER': 'SI=F',
        'PETROL': 'CL=F', 'OIL': 'CL=F',
        'BIST': 'XU100.IS', 'XU100': 'XU100.IS'
    }
    
    # Direkt eşleşme varsa kullan
    if symbol in direct_map:
        return direct_map[symbol], symbol
    
    # Orijinal sembolle dene
    try:
        test = yf.download(symbol, period='5d', progress=False, timeout=15)
        if not test.empty:
            logging.info(f"✅ Sembol doğrulandı: {symbol}")
            return symbol, symbol
    except Exception as e:
        logging.debug(f"Orijinal sembol test başarısız: {e}")
    
    # Türk hissesi olabilir mi?
    if len(symbol) <= 6 and not any(c in symbol for c in ['.', '=', '-', '/']):
        turkish_symbol = f"{symbol}.IS"
        try:
            test = yf.download(turkish_symbol, period='5d', progress=False, timeout=15)
            if not test.empty:
                logging.info(f"✅ Türk hissesi bulundu: {turkish_symbol}")
                return turkish_symbol, symbol
        except Exception as e:
            logging.debug(f"Türk hissesi test başarısız: {e}")
    
    # Hiçbiri çalışmadı
    logging.warning(f"⚠️ Sembol doğrulanamadı: {symbol}")
    return symbol, symbol

# === GÜVENLİ TEKNİK GÖSTERGE HESAPLAMA ===
def safe_calculate_indicators(df):
    """Tüm göstergeleri güvenli şekilde hesapla"""
    try:
        if df.empty or len(df) < 20:
            return df
        
        # === TEMEL GÖSTERGELER ===
        # RSI
        try:
            df['RSI'] = ta.rsi(df['Close'], length=14)
        except:
            pass
        
        # SMA'lar
        try:
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['SMA_200'] = df['Close'].rolling(200).mean()
        except:
            pass
        
        # EMA'lar
        try:
            df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
            df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        except:
            pass
        
        # MACD - Dinamik kolon bulma
        try:
            macd = ta.macd(df['Close'])
            if macd is not None and not macd.empty:
                cols = macd.columns.tolist()
                # MACD line (h ve s olmayan)
                macd_col = next((c for c in cols if 'MACD' in c and 'h' not in c.lower() and 's' not in c.lower()), None)
                # Signal line
                signal_col = next((c for c in cols if 's' in c.lower() and 'MACD' in c), None)
                
                if macd_col:
                    df['MACD'] = macd[macd_col]
                if signal_col:
                    df['MACD_SIGNAL'] = macd[signal_col]
        except Exception as e:
            logging.debug(f"MACD hesaplama hatası: {e}")
        
        # Bollinger Bands - Tamamen güvenli
        try:
            bb = ta.bbands(df['Close'], length=20, std=2)
            if bb is not None and not bb.empty:
                cols = bb.columns.tolist()
                # Upper band
                upper = next((c for c in cols if 'BBU' in c or 'upper' in c.lower()), None)
                # Lower band
                lower = next((c for c in cols if 'BBL' in c or 'lower' in c.lower()), None)
                # Middle band
                middle = next((c for c in cols if 'BBM' in c or 'mid' in c.lower() or 'basis' in c.lower()), None)
                
                if upper:
                    df['BB_UPPER'] = bb[upper]
                if lower:
                    df['BB_LOWER'] = bb[lower]
                if middle:
                    df['BB_MID'] = bb[middle]
                elif upper and lower:
                    df['BB_MID'] = (df['BB_UPPER'] + df['BB_LOWER']) / 2
        except Exception as e:
            logging.debug(f"Bollinger Bands hatası: {e}")
        
        # ATR - Manuel fallback ile
        try:
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        except:
            try:
                # Manuel ATR hesaplama
                high_low = df['High'] - df['Low']
                high_close = abs(df['High'] - df['Close'].shift())
                low_close = abs(df['Low'] - df['Close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                df['ATR'] = true_range.rolling(14).mean()
            except:
                pass
        
        # Volume
        try:
            df['VOL_SMA'] = df['Volume'].rolling(20).mean()
            df['VOL_RATIO'] = df['Volume'] / df['VOL_SMA'].replace(0, 1)
        except:
            pass
        
        # Stochastic
        try:
            stoch = ta.stoch(df['High'], df['Low'], df['Close'])
            if stoch is not None and not stoch.empty:
                cols = stoch.columns.tolist()
                k_col = next((c for c in cols if 'k' in c.lower()), None)
                if k_col:
                    df['STOCH_K'] = stoch[k_col]
        except Exception as e:
            logging.debug(f"Stochastic hatası: {e}")
        
        # ADX
        try:
            adx = ta.adx(df['High'], df['Low'], df['Close'])
            if adx is not None and not adx.empty:
                cols = adx.columns.tolist()
                adx_col = next((c for c in cols if c.startswith('ADX')), None)
                if adx_col:
                    df['ADX'] = adx[adx_col]
        except Exception as e:
            logging.debug(f"ADX hatası: {e}")
        
        return df
        
    except Exception as e:
        logging.error(f"Gösterge hesaplama genel hatası: {e}")
        return df

# === MUM KALIPLARI ===
def detect_candlestick_patterns(df):
    """Temel mum kalıplarını tespit et"""
    patterns = []
    
    try:
        if len(df) < 3:
            return patterns
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        body = abs(last['Close'] - last['Open'])
        full_range = last['High'] - last['Low']
        
        if full_range > 0:
            body_ratio = body / full_range
            
            # Doji
            if body_ratio < 0.1:
                patterns.append("🕯️ DOJI - Kararsızlık")
            
            # Hammer / Shooting Star
            lower_shadow = min(last['Open'], last['Close']) - last['Low']
            upper_shadow = last['High'] - max(last['Open'], last['Close'])
            
            if lower_shadow > 2 * body and upper_shadow < body:
                patterns.append("🔨 HAMMER - Dip sinyali")
            elif upper_shadow > 2 * body and lower_shadow < body:
                patterns.append("⭐ SHOOTING STAR - Tepe sinyali")
        
        # Bullish Engulfing
        if (prev['Close'] < prev['Open'] and last['Close'] > last['Open'] and
            last['Open'] <= prev['Close'] and last['Close'] >= prev['Open']):
            patterns.append("🟢 BULLISH ENGULFING - Güçlü alım")
        
        # Bearish Engulfing
        if (prev['Close'] > prev['Open'] and last['Close'] < last['Open'] and
            last['Open'] >= prev['Close'] and last['Close'] <= prev['Open']):
            patterns.append("🔴 BEARISH ENGULFING - Güçlü satım")
        
    except Exception as e:
        logging.debug(f"Pattern tespit hatası: {e}")
    
    return patterns

# === SİNYAL OLUŞTURMA ===
def generate_signal(df):
    """Akıllı sinyal oluştur"""
    try:
        if df.empty or len(df) < 20:
            return "BEKLE", 50, "YÜKSEK"
        
        last = df.iloc[-1]
        price = last['Close']
        
        score = 0
        factors = 0
        
        # RSI
        rsi = last.get('RSI')
        if pd.notna(rsi):
            if rsi < 30:
                score += 2
            elif rsi < 40:
                score += 1
            elif rsi > 70:
                score -= 2
            elif rsi > 60:
                score -= 1
            factors += 1
        
        # Trend (SMA 50)
        sma_50 = last.get('SMA_50')
        if pd.notna(sma_50):
            if price > sma_50:
                score += 1
            else:
                score -= 1
            factors += 1
        
        # MACD
        macd = last.get('MACD')
        macd_signal = last.get('MACD_SIGNAL')
        if pd.notna(macd) and pd.notna(macd_signal):
            if macd > macd_signal:
                score += 1
            else:
                score -= 1
            factors += 1
        
        # Volume
        vol_ratio = last.get('VOL_RATIO', 1)
        if vol_ratio > 1.5:
            score += 1
            factors += 1
        
        # Normalize score
        if factors > 0:
            normalized_score = (score / factors) * 50 + 50
        else:
            normalized_score = 50
        
        # Karar
        if normalized_score >= 75:
            return "GÜÇLÜ AL", int(normalized_score), "DÜŞÜK"
        elif normalized_score >= 60:
            return "AL", int(normalized_score), "ORTA"
        elif normalized_score <= 25:
            return "GÜÇLÜ SAT", int(100 - normalized_score), "ORTA"
        elif normalized_score <= 40:
            return "SAT", int(100 - normalized_score), "YÜKSEK"
        else:
            return "BEKLE", 50, "ORTA"
        
    except Exception as e:
        logging.error(f"Sinyal oluşturma hatası: {e}")
        return "BEKLE", 50, "YÜKSEK"

# === TELEGRAM KOMUTLARI ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("⚡ Hızlı Analiz", callback_data="quick_help")],
        [InlineKeyboardButton("📚 Komutlar", callback_data="help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    msg = """
🦁 **PROMETHEUS AI v9.1**
_Ultra Stable Edition_

**KULLANIM:**
• Direkt sembol yaz: `BTC`, `THYAO`, `AAPL`
• `/hizli BTC` - Hızlı analiz
• `/analiz ETH` - Detaylı analiz  
• `/risk THYAO` - Risk analizi

**DESTEKLENEN VARLIKLAR:**
• Kripto: BTC, ETH, SOL, AVAX...
• Hisse: AAPL, TSLA, THYAO, GARAN...
• Emtia: ALTIN, PETROL, GÜMÜŞ

Butona bas veya sembol yaz! 👇
    """
    
    await update.message.reply_text(msg, reply_markup=reply_markup, parse_mode=constants.ParseMode.MARKDOWN)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data == "help":
        help_text = """
📚 **KOMUTLAR:**

`/start` - Botu başlat
`/hizli BTC` - Hızlı analiz (10sn)
`/analiz ETH` - Detaylı analiz (20sn)
`/risk THYAO` - Risk analizi

**Direkt Kullanım:**
Sadece sembol yaz: `BTC`, `AAPL`, `THYAO`

**Örnekler:**
• `BTC` → Bitcoin analizi
• `THYAO` → Türk Hava Yolları
• `AAPL` → Apple hissesi
• `ALTIN` → Altın fiyatı
        """
        await query.edit_message_text(help_text, parse_mode=constants.ParseMode.MARKDOWN)

async def quick_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Hızlı analiz"""
    user_input = update.message.text.upper().replace("/HIZLI", "").replace("/ANALIZ", "").strip()
    
    if not user_input:
        await update.message.reply_text("📌 Kullanım: `/hizli BTC` veya sadece `BTC` yaz")
        return
    
    status_msg = await update.message.reply_text(f"⚡ **{user_input}** analiz ediliyor...")
    
    try:
        # Sembol dönüştür
        yf_symbol, display_symbol = smart_symbol_convert(user_input)
        
        # Veri çek
        df = yf.download(yf_symbol, period="2mo", interval="1d", progress=False, timeout=30)
        
        if df.empty:
            await status_msg.edit_text(
                f"❌ **{user_input}** için veri bulunamadı!\n\n"
                f"Sebep olabilir:\n"
                f"• Sembol Yahoo Finance'de yok\n"
                f"• Hisse borsadan çıkmış (delisted)\n"
                f"• Yanlış yazım\n\n"
                f"Alternatif dene: `/hizli BTC` veya `/hizli AAPL`",
                parse_mode=constants.ParseMode.MARKDOWN
            )
            return
        
        # Göstergeleri hesapla
        df = safe_calculate_indicators(df)
        
        # Pattern'leri tespit et
        patterns = detect_candlestick_patterns(df)
        
        # Sinyal oluştur
        signal, confidence, risk = generate_signal(df)
        
        last = df.iloc[-1]
        price = last['Close']
        rsi = last.get('RSI', 50)
        atr = last.get('ATR', price * 0.02)
        
        stop_loss = price - (2 * atr)
        target = price + (3 * atr)
        
        # Analiz mesajı
        analysis = f"""
🦁 **PROMETHEUS AI - {display_symbol}**

🎯 **SİNYAL:** {signal}
📊 **Güven:** %{confidence}
⚠️ **Risk:** {risk}

💰 **Fiyat:** ${price:.2f}
📈 **RSI:** {rsi:.1f}
📊 **ATR:** {atr:.2f}

🕯️ **Pattern'ler:**
{chr(10).join(patterns) if patterns else '• Normal'}

💡 **İşlem Planı:**
• Giriş: ${price:.2f}
• Stop: ${stop_loss:.2f}
• Hedef: ${target:.2f}

📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}
        """
        
        await status_msg.edit_text(analysis, parse_mode=constants.ParseMode.MARKDOWN)
        
    except Exception as e:
        logging.error(f"Analiz hatası: {e}")
        await status_msg.edit_text(
            f"⚠️ **Analiz sırasında hata oluştu**\n\n"
            f"Hata: `{str(e)[:100]}`\n\n"
            f"Lütfen başka bir sembol dene veya tekrar dene.",
            parse_mode=constants.ParseMode.MARKDOWN
        )

async def handle_direct_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Direkt mesaj - otomatik hızlı analiz"""
    await quick_analysis(update, context)

# === BOT BAŞLATMA ===
def start_bot():
    if not TELEGRAM_TOKEN:
        logging.error("❌ TELEGRAM_TOKEN bulunamadı!")
        return
    
    try:
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        
        # Komut handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("hizli", quick_analysis))
        application.add_handler(CommandHandler("analiz", quick_analysis))
        application.add_handler(CommandHandler("risk", quick_analysis))
        
        # Callback handler
        application.add_handler(CallbackQueryHandler(button_handler))
        
        # Direkt mesaj handler
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_direct_message
        ))
        
        logging.info("🚀 PROMETHEUS AI v9.1 başlatılıyor...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        logging.error(f"Bot başlatma hatası: {e}")
        raise

if __name__ == '__main__':
    keep_alive()
    try:
        start_bot()
    except KeyboardInterrupt:
        logging.info("Bot kapatılıyor...")
    except Exception as e:
        logging.error(f"Kritik hata: {e}")
        raise