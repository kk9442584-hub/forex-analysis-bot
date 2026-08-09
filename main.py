import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ==== المفاتيح (من GitHub Secrets) ====
TWELVE_DATA_KEY = os.environ["TWELVE_DATA_KEY"]
GEMINI_KEY = os.environ["GEMINI_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
ALPHA_VANTAGE_KEY = os.environ["ALPHA_VANTAGE_KEY"]

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD",
         "NZD/USD", "GBP/JPY", "EUR/JPY", "EUR/GBP"]

def get_forex_data(pair):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": pair,
        "interval": "4h",
        "outputsize": 210,
        "apikey": TWELVE_DATA_KEY
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    if "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df = df.iloc[::-1].reset_index(drop=True)
    return df

def compute_indicators(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema50 = close.ewm(span=50).mean().iloc[-1]
    ema200 = close.ewm(span=200).mean().iloc[-1]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / tr.rolling(14).mean())
    minus_di = 100 * (minus_dm.rolling(14).mean() / tr.rolling(14).mean())
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(14).mean().iloc[-1]

    support = low.tail(20).min()
    resistance = high.tail(20).max()
    price = close.iloc[-1]

    return {
        "price": round(price, 5),
        "ema50": round(ema50, 5),
        "ema200": round(ema200, 5),
        "rsi": round(rsi, 2),
        "adx": round(adx, 2),
        "atr": round(atr, 5),
        "support": round(support, 5),
        "resistance": round(resistance, 5)
    }

def get_forex_news():
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "topics": "economy_macro,finance",
        "limit": 15,
        "apikey": ALPHA_VANTAGE_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        articles = data.get("feed", [])
        headlines = []
        for a in articles[:10]:
            title = a.get("title", "")
            summary = a.get("summary", "")[:150]
            headlines.append(f"- {title}: {summary}")
        return "\n".join(headlines) if headlines else "لا توجد أخبار متاحة حالياً"
    except Exception:
        return "لا توجد أخبار متاحة حالياً"

def ask_gemini(all_data, news_text):
    prompt = f"""أنت محلل فني وإخباري للفوركس. لديك بيانات {len(all_data)} أزواج عملات على فريم 4 ساعات، بالإضافة لأهم الأخبار الاقتصادية الحالية.

قواعد الإشارة القوية (الشروط الفنية الثلاثة لازم تتوافق معاً، والأخبار تُستخدم كتأكيد أو تحذير إضافي):
1. الاتجاه: EMA50 فوق EMA200 = صاعد قوي، أو EMA50 تحت EMA200 = هابط قوي
2. الزخم: RSI بين 40-60 يدعم استمرار الاتجاه (مش متطرف)
3. قوة الترند: ADX أعلى من 25
4. الأخبار: إذا وجدت أخبار مهمة تدعم أو تتعارض بشكل واضح مع اتجاه زوج معين، اعتبر ذلك عامل تأكيد إضافي. إذا كانت الأخبار تحذر من تقلب عالٍ متوقع قريباً لعملة معينة، لا تصدر إشارة على الأزواج المرتبطة بها حتى لو الشروط الفنية متوفرة.

البيانات الفنية:
{all_data}

أهم الأخبار الاقتصادية الحالية:
{news_text}

أعطني فقط الأزواج التي تحقق الشروط الفنية الثلاثة معاً بدقة، والأخبار لا تتعارض معها بشكل خطير. اكتب الرد بهذا الشكل بالضبط لكل زوج مستوفي (وإن لم يوجد أي زوج مستوفٍ، اكتب فقط: NONE):

PAIR: [اسم الزوج]
DIRECTION: [صاعد/هابط]
"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(url, json=body, timeout=60)
    result = r.json()
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return "NONE"

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})

def main():
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # الاثنين=0 ... الجمعة=4, السبت=5, الأحد=6
    hour = now.hour

    # تخطي الفحص وقت إغلاق السوق (جمعة 21:00 UTC - أحد 22:00 UTC)
    market_closed = (
        (weekday == 4 and hour >= 21) or
        (weekday == 5) or
        (weekday == 6 and hour < 22)
    )
    if market_closed:
        print("السوق مقفول - تخطي الفحص")
        return

    all_data = {}
    indicators_map = {}

    # رسالة تأكيد يومية الساعة 10 صباحاً بتوقيت الأردن/فلسطين (UTC+3) = 7 صباحاً UTC
    if hour == 22:
        send_telegram("✅ البوت شغال - بدأ فحص جديد لليوم")

    for pair in PAIRS:
        df = get_forex_data(pair)
        if df is None or len(df) < 200:
            continue
        ind = compute_indicators(df)
        indicators_map[pair] = ind
        all_data[pair] = ind

    if not all_data:
        return

    news_text = get_forex_news()
    analysis = ask_gemini(all_data, news_text)

    if "NONE" in analysis or "PAIR:" not in analysis:
        print("لا توجد إشارات قوية اليوم")
        return

    blocks = analysis.split("PAIR:")
    for block in blocks[1:]:
        lines = block.strip().split("\n")
        pair_name = lines[0].strip()
        direction = lines[1].replace("DIRECTION:", "").strip() if len(lines) > 1 else ""

        if pair_name in indicators_map:
            ind = indicators_map[pair_name]
            emoji = "🟢" if "صاعد" in direction else "🔴"
            msg = f"""{emoji} إشارة قوية - {pair_name} ({direction})

السعر وقت التنبيه: {ind['price']}
ATR: {ind['atr']}

أقرب دعم: {ind['support']}
أقرب مقاومة: {ind['resistance']}"""
            send_telegram(msg)
            print(f"تم إرسال إشارة: {pair_name}")

if __name__ == "__main__":
    main()
         
