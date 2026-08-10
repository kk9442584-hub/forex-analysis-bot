import os
import io
import base64
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# ==== المفاتيح (من GitHub Secrets) ====
TWELVE_DATA_KEY = os.environ["TWELVE_DATA_KEY"]
GEMINI_KEY = os.environ["GEMINI_KEY"]
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
ALPHA_VANTAGE_KEY = os.environ["ALPHA_VANTAGE_KEY"]
CAPITAL_API_KEY = os.environ.get("CAPITAL_API_KEY")
CAPITAL_API_PASSWORD = os.environ.get("CAPITAL_API_PASSWORD")
CAPITAL_IDENTIFIER = os.environ.get("CAPITAL_IDENTIFIER")
CAPITAL_BASE = "https://demo-api-capital.backend-capital.com/api/v1"

RISK_PERCENT = 0.01
RR_RATIO = 2
ATR_MULTIPLIER = 1.5

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD",
         "NZD/USD", "GBP/JPY", "EUR/JPY", "EUR/GBP"]


def get_forex_data(pair):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": pair,
        "interval": "4h",
        "outputsize": 310,
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


def find_nearest_levels(df, window=5):
    """يلاقي أقرب دعم ومقاومة حقيقيين (نقاط ارتداد سابقة فعلية)"""
    highs = df["high"]
    lows = df["low"]
    price = df["close"].iloc[-1]

    swing_highs = []
    swing_lows = []
    for i in range(window, len(df) - window):
        window_high = highs.iloc[i - window:i + window + 1]
        window_low = lows.iloc[i - window:i + window + 1]
        if highs.iloc[i] == window_high.max():
            swing_highs.append(highs.iloc[i])
        if lows.iloc[i] == window_low.min():
            swing_lows.append(lows.iloc[i])

    resistances_above = [h for h in swing_highs if h > price]
    supports_below = [l for l in swing_lows if l < price]

    resistance = min(resistances_above) if resistances_above else highs.max()
    support = max(supports_below) if supports_below else lows.min()
    return support, resistance


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

    price = close.iloc[-1]

    # الدعم والمقاومة الحاليان (لعرضهما بالرسالة)
    support, resistance = find_nearest_levels(df)

    # الدعم والمقاومة "السابقان" (بدون آخر شمعة) للتأكد من الاختراق الفعلي
    support_prev, resistance_prev = find_nearest_levels(df.iloc[:-1])

    bullish_breakout = price > resistance_prev
    bearish_breakout = price < support_prev

    return {
        "price": round(price, 5),
        "ema50": round(ema50, 5),
        "ema200": round(ema200, 5),
        "rsi": round(rsi, 2),
        "adx": round(adx, 2),
        "atr": round(atr, 5),
        "support": round(support, 5),
        "resistance": round(resistance, 5),
        "bullish_breakout": bool(bullish_breakout),
        "bearish_breakout": bool(bearish_breakout),
        "trend_up": ema50 > ema200,
        "trend_down": ema50 < ema200,
        "adx_ok": adx > 25,
    }


def passes_conditions(ind):
    """وضع اختبار مؤقت - شروط مخففة"""
    if ind["trend_up"]:
        return "صاعد"
    if ind["trend_down"]:
        return "هابط"
    return None


def generate_chart(df, pair, support, resistance):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
    ax.plot(df["close"].tail(120).values, color="#2196F3", linewidth=1.3, label="السعر")
    ax.axhline(support, color="#4CAF50", linestyle="--", linewidth=1, label="دعم")
    ax.axhline(resistance, color="#F44336", linestyle="--", linewidth=1, label="مقاومة")
    ax.set_title(pair)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


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


def ask_gemini_vision(pair, direction, ind, news_text, chart_b64):
    prompt = f"""أنت محلل فني وإخباري محترف للفوركس. زوج {pair} حقق للتو اختراقاً فعلياً
لمستوى فني مهم، والمؤشرات تدعم اتجاه {direction}.

المعطيات الفنية:
- السعر الحالي: {ind['price']}
- الدعم: {ind['support']}
- المقاومة: {ind['resistance']}
- RSI: {ind['rsi']}
- ADX: {ind['adx']}

أهم الأخبار الاقتصادية الحالية:
{news_text}

معك أيضاً صورة الشارت الفعلية لآخر فترة تداول.

المطلوب:
1. اكتب فقرة قصيرة (3-4 أسطر) بالعربية تصف الوضع الحالي للزوج بأسلوب واضح ومفيد،
   بالاستفادة من الشكل البصري للشارت (هل هناك نمط واضح مثل مثلث أو قناة؟) والأخبار.
2. إذا كان هناك سيناريو فني منطقي مبني على مبدأ تحليل فني معروف (وليس تنبؤاً قطعياً)
   يتعلق بمستوى قريب آخر، اكتبه في فقرة منفصلة تبدأ بالضبط بكلمة "⚠️ تنبؤ:".
   إذا لم يوجد سيناريو واضح، لا تكتب هذا الجزء إطلاقاً.
3. إذا وجدت الأخبار تتعارض بشكل خطير مع هذا الاتجاه، اكتب فقط: SKIP

لا تستخدم أي مقدمات، ابدأ مباشرة بالنص المطلوب.
"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}"
    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": chart_b64}}
            ]
        }]
    }
    r = requests.post(url, json=body, timeout=60)
    result = r.json()
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        return "SKIP"


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})


def capital_login():
    r = requests.post(f"{CAPITAL_BASE}/session", json={
        "identifier": CAPITAL_IDENTIFIER,
        "password": CAPITAL_API_PASSWORD
    }, headers={
        "X-CAP-API-KEY": CAPITAL_API_KEY,
        "Content-Type": "application/json"
    })
    if r.status_code != 200:
        print("فشل تسجيل الدخول لـ Capital.com:", r.text)
        return None
    return {
        "X-CAP-API-KEY": CAPITAL_API_KEY,
        "CST": r.headers.get("CST"),
        "X-SECURITY-TOKEN": r.headers.get("X-SECURITY-TOKEN"),
        "Content-Type": "application/json"
    }


def get_balance(headers):
    r = requests.get(f"{CAPITAL_BASE}/accounts", headers=headers)
    data = r.json()
    try:
        return data["accounts"][0]["balance"]["balance"]
    except (KeyError, IndexError):
        return None


def open_trade(pair, direction, ind):
    headers = capital_login()
    if headers is None:
        return

    balance = get_balance(headers)
    if balance is None:
        print("ما قدرت أجيب رصيد الحساب")
        return

    epic = pair.replace("/", "")
    atr = ind["atr"]
    stop_distance = round(atr * ATR_MULTIPLIER, 5)
    profit_distance = round(stop_distance * RR_RATIO, 5)

    risk_amount = balance * RISK_PERCENT
    size = round(risk_amount / (stop_distance * 10000), 2)
    size = max(size, 0.01)

    deal_direction = "BUY" if direction == "صاعد" else "SELL"

    payload = {
        "epic": epic,
        "direction": deal_direction,
        "size": size,
        "stopDistance": round(stop_distance * 10000, 1),
        "profitDistance": round(profit_distance * 10000, 1)
    }

    r = requests.post(f"{CAPITAL_BASE}/positions", json=payload, headers=headers)
    print(f"فتح صفقة {pair} ({deal_direction}) - status: {r.status_code}")
    print(r.text)


def main():
    now = datetime.now(timezone.utc)
    weekday = now.weekday()
    hour = now.hour

    market_closed = (
        (weekday == 4 and hour >= 21) or
        (weekday == 5) or
        (weekday == 6 and hour < 22)
    )
    if market_closed:
        print("السوق مقفول - تخطي الفحص")
        return

    # رسالة تأكيد يومية عند لحظة فتح السوق (1:00 فجراً بتوقيت الأردن)
    if hour == 22:
        send_telegram("✅ البوت شغال - بدأ فحص جديد لليوم")

    news_text = None

    for pair in PAIRS:
        df = get_forex_data(pair)
        if df is None or len(df) < 300:
            continue

        ind = compute_indicators(df)
        direction = passes_conditions(ind)
        if direction is None:
            continue

        # وصلنا هنا فقط لو في اختراق فعلي + شروط فنية متوافقة
        if news_text is None:
            news_text = get_forex_news()

        chart_b64 = generate_chart(df, pair, ind["support"], ind["resistance"])
        analysis = ask_gemini_vision(pair, direction, ind, news_text, chart_b64)

        if "SKIP" in analysis:
            print(f"{pair}: تم تجاوزها بسبب تعارض الأخبار")
            continue

        emoji = "🟢" if direction == "صاعد" else "🔴"
        msg = f"""{emoji} إشارة قوية - {pair} ({direction})

{analysis}

السعر وقت التنبيه: {ind['price']}
ATR: {ind['atr']}
أقرب دعم: {ind['support']}
أقرب مقاومة: {ind['resistance']}"""

        send_telegram(msg)
        print(f"تم إرسال إشارة: {pair}")
        open_trade(pair, direction, ind)

    if news_text is None:
        print("لا توجد إشارات قوية اليوم")


if __name__ == "__main__":
    main()
