"""
FILE → Place at: backend/utils/sentiment.py

Financial Sentiment Analysis Engine
─────────────────────────────────────
Two engines, used in order:
  1. FinSentiment — built-in financial keyword scorer (no install needed, instant)
  2. Claude AI    — deep analysis via Anthropic API (optional, for rich summaries)

Usage:
    from backend.utils.sentiment import analyze, batch_analyze, sentiment_summary

    result = analyze("Tesla beats earnings expectations by 20%")
    # → {
    #     "label":  "positive",
    #     "score":  0.78,          # -1.0 to +1.0
    #     "pos":    0.78,
    #     "neu":    0.15,
    #     "neg":    0.07,
    #     "emoji":  "🟢",
    #     "badge":  "Bullish",
    #     "impact": "+0.78",
    #     "engine": "finsent"
    #   }
"""

import re
import math
import os

# ══════════════════════════════════════════════════════════════════════════════
#  FINANCIAL KEYWORD LEXICON
#  Score range: +1.0 (very bullish) to -1.0 (very bearish)
# ══════════════════════════════════════════════════════════════════════════════

_BULLISH = {
    # Earnings / growth
    "beat":0.7, "beats":0.7, "exceeded":0.75, "surpass":0.75, "surpassed":0.75,
    "record":0.65, "record-high":0.8, "all-time high":0.85,
    "profit":0.6, "profitable":0.65, "earnings":0.45, "revenue growth":0.7,
    "revenue beat":0.75, "eps beat":0.75, "strong earnings":0.8,
    "outperform":0.7, "outperformed":0.7, "upgrade":0.65, "upgraded":0.65,
    "buy rating":0.75, "strong buy":0.85, "overweight":0.6,
    # Market / stock action
    "rally":0.65, "rallied":0.65, "surge":0.75, "surged":0.75, "soar":0.8,
    "soared":0.8, "jump":0.6, "jumped":0.6, "rise":0.55, "rose":0.55,
    "gain":0.55, "gains":0.55, "breakout":0.7, "bull":0.6, "bullish":0.75,
    "upside":0.6, "uptrend":0.65, "momentum":0.5, "52-week high":0.8,
    "new high":0.8, "higher":0.45,
    # Business
    "partnership":0.55, "deal":0.5, "contract":0.55, "acquisition":0.5,
    "merger":0.45, "expansion":0.6, "launch":0.5, "launched":0.5,
    "innovation":0.55, "breakthrough":0.75, "approval":0.65, "approved":0.65,
    "fda approved":0.85, "fda approves":0.85, "approves":0.6, "approved":0.65, "dividend":0.5, "buyback":0.6, "stock buyback":0.65,
    "share buyback":0.65, "guidance raised":0.8, "raised guidance":0.8,
    "beat expectations":0.85, "top estimates":0.75, "above estimates":0.75,
    "strong demand":0.7, "robust":0.6, "optimistic":0.6, "positive":0.55,
    "confident":0.55, "opportunity":0.5, "growth":0.55,
    # Macro
    "rate cut":0.65, "stimulus":0.6, "easing":0.55, "dovish":0.6,
    "soft landing":0.65,
}

_BEARISH = {
    # Earnings / decline
    "miss":0.7, "missed":0.7, "miss expectations":0.85, "below estimates":0.8,
    "disappointed":0.65, "disappointing":0.65, "weak":0.55, "weakness":0.6,
    "loss":0.65, "losses":0.65, "decline":0.6, "declined":0.6,
    "fell":0.6, "fall":0.6, "drop":0.65, "dropped":0.65, "plunge":0.8,
    "plunged":0.8, "crash":0.85, "crashed":0.85, "tank":0.75, "tanked":0.75,
    "slump":0.7, "slumped":0.7, "sink":0.65, "sank":0.65,
    "downgrade":0.7, "downgraded":0.7, "sell rating":0.75, "underperform":0.65,
    "underweight":0.6, "cut":0.5, "guidance cut":0.8, "lowered guidance":0.8,
    "lowered":0.5,
    # Business trouble
    "recall":0.65, "recalled":0.65, "lawsuit":0.65, "sued":0.7, "fine":0.6,
    "fined":0.65, "penalty":0.6, "fraud":0.85, "investigation":0.65,
    "probe":0.6, "scandal":0.8, "default":0.85, "bankruptcy":0.95,
    "bankrupt":0.95, "layoffs":0.65, "layoff":0.65, "job cuts":0.7,
    "restructuring":0.5, "write-off":0.7, "writedown":0.7,
    "debt":0.45, "deficit":0.55, "shortfall":0.6, "loss":0.65,
    "negative":0.5, "risk":0.4, "concern":0.45, "warning":0.6,
    "safety issue":0.7, "safety recall":0.8, "regulatory":0.4,
    "rejected":0.65, "denied":0.55, "blocked":0.55,
    # Market / macro
    "bear":0.6, "bearish":0.75, "sell-off":0.75, "selloff":0.75,
    "downturn":0.7, "correction":0.55, "recession":0.75, "inflation":0.5,
    "rate hike":0.6, "hawkish":0.55, "uncertainty":0.5, "volatile":0.45,
    "volatility":0.45, "turbulence":0.5, "headwinds":0.55,
    "52-week low":0.8, "new low":0.8, "lower":0.45,
}

# Intensifiers / negators
_INTENSIFY = {"very":1.3,"extremely":1.5,"significantly":1.3,"sharply":1.3,
              "massively":1.4,"heavily":1.3,"substantially":1.2,"major":1.2,
              "huge":1.3,"big":1.1,"strong":1.2,"strongly":1.3,"record":1.3}
_NEUTRAL_WORDS = {"unchanged","flat","stable","steady","sideways","mixed","hold","inline",
                  "as expected","in line","meets","meet","consensus"}
_NEGATE    = {"not","no","never","neither","nor","without","lack","lacks",
              "lacking","fail","fails","failed","failure","unable","cannot",
              "can't","won't","wouldn't","didn't","doesn't","don't"}


# ══════════════════════════════════════════════════════════════════════════════
#  CORE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\-']+", text.lower())


def _score_text(text: str) -> dict:
    """Score a single text string. Returns raw pos/neg/neu floats."""
    # Neutralise if headline is explicitly "unchanged/flat/stable/as expected"
    raw_lower = text.lower()
    neutral_override = any(w in raw_lower for w in _NEUTRAL_WORDS)

    tokens   = _tokenize(text)
    raw_text = text.lower()

    pos_hits, neg_hits = [], []

    # Multi-word phrase match first (up to 4 words)
    for phrase_len in (4, 3, 2, 1):
        for i in range(len(tokens) - phrase_len + 1):
            phrase = " ".join(tokens[i : i + phrase_len])

            # Check for negation in window before phrase
            window_start = max(0, i - 3)
            window       = set(tokens[window_start:i])
            negated      = bool(window & _NEGATE)

            # Intensifier
            intensifier = 1.0
            if i > 0 and tokens[i-1] in _INTENSIFY:
                intensifier = _INTENSIFY[tokens[i-1]]

            if phrase in _BULLISH:
                base = _BULLISH[phrase] * intensifier
                if negated:
                    neg_hits.append(base * 0.6)   # negated bullish → mildly bearish
                else:
                    pos_hits.append(min(base, 1.0))

            elif phrase in _BEARISH:
                base = _BEARISH[phrase] * intensifier
                if negated:
                    pos_hits.append(base * 0.4)   # negated bearish → mildly bullish
                else:
                    neg_hits.append(min(base, 1.0))

    # Aggregate — use mean of top-3 hits to avoid length bias
    def top_mean(hits, n=3):
        if not hits:
            return 0.0
        top = sorted(hits, reverse=True)[:n]
        return sum(top) / len(top)

    pos = top_mean(pos_hits)
    neg = top_mean(neg_hits)

    # Net compound score (-1 to +1), softened by a tanh-like curve
    raw    = pos - neg
    denom  = 1 + abs(raw)
    compound = raw / denom if denom else 0.0
    compound = max(-1.0, min(1.0, compound * 1.6))  # slight amplification

    # Normalize to pos/neu/neg distribution (sum = 1)
    total = pos + neg
    if total == 0:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

    share_pos = pos / total
    share_neg = neg / total
    strength  = min(total / 1.5, 1.0)   # how confident we are

    out_pos = round(share_pos * strength, 3)
    out_neg = round(share_neg * strength, 3)
    out_neu = round(1.0 - out_pos - out_neg, 3)

    if neutral_override:
        compound *= 0.3   # dampen to near-neutral

    return {
        "compound": round(compound, 3),
        "pos":      out_pos,
        "neg":      out_neg,
        "neu":      max(out_neu, 0.0),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def analyze(text: str, use_claude: bool = False, api_key: str = "") -> dict:
    """
    Analyze a single headline/text.

    Parameters
    ----------
    text       : news headline or summary
    use_claude : if True and api_key provided, calls Claude for deeper analysis
    api_key    : Anthropic API key (reads ANTHROPIC_API_KEY env var if empty)

    Returns
    -------
    dict with keys: label, score, pos, neg, neu, emoji, badge, impact, engine
    """
    raw = _score_text(text)

    compound = raw["compound"]
    pos      = raw["pos"]
    neg      = raw["neg"]
    neu      = raw["neu"]

    # Label + badge
    if compound >= 0.15:
        label = "positive"
        emoji = "🟢"
        badge = "Bullish"
    elif compound <= -0.15:
        label = "negative"
        emoji = "🔴"
        badge = "Bearish"
    else:
        label = "neutral"
        emoji = "🟡"
        badge = "Neutral"

    result = {
        "label":   label,
        "score":   compound,
        "pos":     pos,
        "neg":     neg,
        "neu":     neu,
        "emoji":   emoji,
        "badge":   badge,
        "impact":  f"{compound:+.2f}",
        "engine":  "finsent",
        "summary": "",
    }

    # ── Optional: Claude deep analysis ───────────────────────────────────────
    if use_claude:
        try:
            key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
            if key:
                result = _claude_analyze(text, result, key)
        except Exception as e:
            print(f"Claude sentiment error: {e}")

    return result


def _claude_analyze(text: str, base: dict, api_key: str) -> dict:
    """Call Claude API for enriched sentiment + one-line market impact summary."""
    import urllib.request, json as _json

    prompt = f"""Analyze this financial news headline for market sentiment.
Headline: "{text}"

Respond in JSON only (no markdown, no explanation):
{{
  "label": "positive" | "neutral" | "negative",
  "score": <float -1.0 to 1.0>,
  "pos": <float 0.0 to 1.0>,
  "neu": <float 0.0 to 1.0>,
  "neg": <float 0.0 to 1.0>,
  "summary": "<one sentence: what this means for investors>"
}}
Scores must sum to 1.0. Be precise about financial impact."""

    payload = _json.dumps({
        "model":      "claude-haiku-4-5-20251001",
        "max_tokens": 256,
        "messages":   [{"role": "user", "content": prompt}]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data    = payload,
        headers = {
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        method = "POST"
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = _json.loads(resp.read())

    text_block = data["content"][0]["text"].strip()
    # Strip markdown fences if present
    text_block = re.sub(r"```[a-z]*\n?", "", text_block).strip("`").strip()
    parsed     = _json.loads(text_block)

    compound = float(parsed.get("score", base["score"]))
    label    = parsed.get("label",   base["label"])
    pos      = float(parsed.get("pos", base["pos"]))
    neg      = float(parsed.get("neg", base["neg"]))
    neu      = float(parsed.get("neu", base["neu"]))
    summary  = parsed.get("summary", "")

    emoji = "🟢" if label == "positive" else "🔴" if label == "negative" else "🟡"
    badge = "Bullish" if label == "positive" else "Bearish" if label == "negative" else "Neutral"

    return {
        "label":   label,
        "score":   round(compound, 3),
        "pos":     round(pos, 3),
        "neg":     round(neg, 3),
        "neu":     round(neu, 3),
        "emoji":   emoji,
        "badge":   badge,
        "impact":  f"{compound:+.2f}",
        "engine":  "claude",
        "summary": summary,
    }


def batch_analyze(texts: list[str], use_claude: bool = False, api_key: str = "") -> list[dict]:
    """Analyze a list of headlines. Returns list of result dicts."""
    return [analyze(t, use_claude=use_claude, api_key=api_key) for t in texts]


def sentiment_summary(results: list[dict]) -> dict:
    """
    Aggregate a list of analyze() results into a stock-level sentiment summary.

    Returns
    -------
    {
        "overall_label":  "positive" | "neutral" | "negative",
        "overall_score":  float,           # -1.0 to +1.0
        "overall_emoji":  "🟢" | "🟡" | "🔴",
        "overall_badge":  "Bullish" | ...,
        "pos_pct":        int,             # 0-100
        "neu_pct":        int,
        "neg_pct":        int,
        "total":          int,
        "buy_signal":     bool,
        "signal_strength": "Strong" | "Moderate" | "Weak",
        "breakdown": [{"label","score","emoji","badge","impact"}, ...]
    }
    """
    if not results:
        return {
            "overall_label": "neutral", "overall_score": 0.0,
            "overall_emoji": "🟡",      "overall_badge": "Neutral",
            "pos_pct": 0, "neu_pct": 100, "neg_pct": 0,
            "total": 0, "buy_signal": False,
            "signal_strength": "Weak", "breakdown": []
        }

    pos_count = sum(1 for r in results if r["label"] == "positive")
    neg_count = sum(1 for r in results if r["label"] == "negative")
    neu_count = sum(1 for r in results if r["label"] == "neutral")
    total     = len(results)

    avg_score = round(sum(r["score"] for r in results) / total, 3)

    pos_pct = round(pos_count / total * 100)
    neg_pct = round(neg_count / total * 100)
    neu_pct = 100 - pos_pct - neg_pct

    if avg_score >= 0.15:
        label, emoji, badge = "positive", "🟢", "Bullish"
    elif avg_score <= -0.15:
        label, emoji, badge = "negative", "🔴", "Bearish"
    else:
        label, emoji, badge = "neutral",  "🟡", "Neutral"

    abs_score = abs(avg_score)
    if abs_score >= 0.5:
        strength = "Strong"
    elif abs_score >= 0.25:
        strength = "Moderate"
    else:
        strength = "Weak"

    return {
        "overall_label":   label,
        "overall_score":   avg_score,
        "overall_emoji":   emoji,
        "overall_badge":   badge,
        "pos_pct":         pos_pct,
        "neu_pct":         neu_pct,
        "neg_pct":         neg_pct,
        "total":           total,
        "buy_signal":      label == "positive" and abs_score >= 0.25,
        "signal_strength": strength,
        "breakdown":       [
            {k: r[k] for k in ("label","score","emoji","badge","impact","summary","engine")}
            for r in results
        ]
    }