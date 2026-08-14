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

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD",
         "USD/CAD", "GBP/JPY", "EUR/JPY"]

def get_forex_data_batch(pairs):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": ",".join(pairs),
        "interval": "15min",
        "outputsize": 310,
        "apikey": TWELVE_DATA_KEY
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    result = {}
    for pair in pairs:
        pair_data = data.get(pair)
        if not pair_data or "values" not in pair_data:
            result[pair] = None
            continue
        df = pd.DataFrame(pair_data["values"])
        df["close"] = df["close"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df = df.iloc[::-1].reset_index(drop=True)
        result[pair] = df
    return result


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
    """فحص أولي حتمي (بدون AI): هل توفرت الشروط الفنية + اختراق فعلي؟"""
    if not ind["adx_ok"]:
        return None
    if ind["trend_up"] and ind["bullish_breakout"] and ind["rsi"] > 55:
        return "صاعد"
    if ind["trend_down"] and ind["bearish_breakout"] and ind["rsi"] < 45:
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
    prompt = f"""أنت محلل فني وإخباري محترف للفوركس. زوج {pair} حقق اختراقاً فعلياً {direction}.
المستوى فني مهم, والمؤشرات تدعم اتجاه {direction}.

المعطيات الفنية:
- السعر الحالي: {ind['price']}
- الدعم: {ind['support']}
- المقاومة: {ind['resistance']}
- RSI: {ind['rsi']}
- ADX: {ind['adx']}

أهم الأخبار الاقتصادية الحالية:
{news_text}

معك أيضاً صورة الشارت الفعلية لآخر فترة تداول.

المطلوب: رد بصيغة JSON فقط، بدون أي نص خارج الأقواس، بهذا الشكل بالضبط:

{{
  "decision": "GO" أو "SKIP",
  "summary": "فقرة قصيرة (3-4 أسطر) بالعربية تصف الوضع الحالي بأسلوب واضح ومفيد، بالاستفادة من الشكل البصري للشارت (هل هناك نمط واضح مثل مثلث أو قناة؟) والأخبار",
  "next_target": "توقع لمستوى قريب ثاني إذا كان فيه سيناريو منطقي واضح، أو فارغ لو لا يوجد",
  "skip_reason": "إذا كان decision هو SKIP، اكتب هنا سبب واضح ومختصر (سطر أو سطرين): هل الأخبار تتعارض بشكل خطير مع الاتجاه؟ أو الشارت غير مقنع بصرياً؟ إذا كان decision هو GO، اترك هذا الحقل فارغاً"
}}

لا تكتب أي مقدمات أو نص خارج الـ JSON. لا تستخدم علامات ```json أو أي تنسيق آخر، فقط الكائن JSON مباشرة."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_KEY}"
    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": chart_b64}}
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json",
            "thinking_config": {"thinking_budget": 0}
        }
    }
    try:
        r = requests.post(url, json=body, timeout=120)
        result = r.json()
        raw_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        print(f"--- رد Gemini الخام لـ {pair} ---")
        print(raw_text)
        print("--- نهاية الرد ---")

        import json
        parsed = json.loads(raw_text)

        decision = parsed.get("decision", "SKIP").strip().upper()
        summary = parsed.get("summary", "").strip()
        next_target = parsed.get("next_target", "").strip()
        skip_reason = parsed.get("skip_reason", "").strip()

        if decision == "GO":
            final_text = summary
            if next_target:
                final_text += f"\n\nالهدف القادم المحتمل: {next_target}"
            return final_text
        else:
            reason = skip_reason if skip_reason else "لم يتم تحديد سبب من Gemini"
            return f"SKIP\n{reason}"

    except Exception as e:
        print(f"⚠️ خطأ في معالجة رد Gemini لـ {pair}: {e}")
        try:
            print(f"محتوى الرد الخام: {result}")
        except Exception:
            print(f"لم يصل رد صالح من الخادم. status_code: {r.status_code if 'r' in dir() else 'غير معروف'}, نص الرد: {r.text if 'r' in dir() else 'غير متوفر'}")
        return "SKIP\nخطأ تقني في معالجة رد Gemini (راجع اللوق)"
             
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


def has_open_position(headers):
    r = requests.get(f"{CAPITAL_BASE}/positions", headers=headers)
    data = r.json()
    positions = data.get("positions", [])
    return len(positions) > 0

def open_trade(pair, direction, ind):
    headers = capital_login()
    if headers is None:
        return

    if has_open_position(headers):
        print(f"فيه صفقة مفتوحة بالفعل - تم تخطي {pair}")
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

    all_pairs_data = get_forex_data_batch(PAIRS)

    for pair in PAIRS:
        df = all_pairs_data.get(pair)
        if df is None or len(df) < 300:
            continue

        ind = compute_indicators(df)
        direction = passes_conditions(ind)
        print(f"{pair} | RSI: {ind['rsi']} | ADX: {ind['adx']} | trend_up: {ind['trend_up']} | trend_down: {ind['trend_down']} | bullish_breakout: {ind['bullish_breakout']} | bearish_breakout: {ind['bearish_breakout']} | النتيجة: {direction}")
        if direction is None:
            continue

        # وصلنا هنا فقط لو في اختراق فعلي + شروط فنية متوافقة
        if news_text is None:
            news_text = get_forex_news()

        chart_b64 = generate_chart(df, pair, ind["support"], ind["resistance"])
        analysis = ask_gemini_vision(pair, direction, ind, news_text, chart_b64)

        if "SKIP" in analysis:
            print(f"{pair}: تم تجاوزها بسبب تعارض الأخبار")
            print(f"--- رد Gemini الكامل لـ {pair} ---")
            print(analysis)
            print("--- نهاية الرد ---")
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
