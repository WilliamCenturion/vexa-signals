import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# VEXA SIGNALS 2.1 | Streamlit
# Public market data only. No order execution.
# ============================================================

st.set_page_config(
    page_title="VEXA Signals 2.1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background:#090909; color:#F5F5F5; }
[data-testid="stHeader"] { background:rgba(9,9,9,0); }
[data-testid="stSidebar"] { background:#141414; }
h1,h2,h3,p,label,span { color:#F5F5F5; }
h1,h2 { color:#F5C518 !important; }
div[data-testid="stMetric"] {
  background:#171717; border:1px solid #3A3214;
  padding:14px; border-radius:10px;
}
div[data-testid="stMetricLabel"] { color:#F5C518 !important; }
div[data-testid="stMetricValue"] { color:#FFFFFF !important; }
.stButton>button { background:#F5C518; color:#090909; border:0; font-weight:700; }
.stButton>button:hover { background:#FFD83D; color:#090909; }
[data-testid="stDataFrame"] { border:1px solid #3A3214; }
small { color:#B7A96B !important; }
</style>
""", unsafe_allow_html=True)

BASE_HL = "https://api.hyperliquid.xyz/info"
BASE_BG = "https://api.bitget.com"
HEADERS = {"User-Agent": "VEXA-Signals/2.1", "Accept": "application/json"}
TIMEFRAME = "15m"
MAX_MARKETS = 81
MIN_VOLUME_USD = 10_000_000
CANDLE_LIMIT = 500
SL_ATR = 1.5
TP_ATR = 3.0

st.title("VEXA SIGNALS 2.1")
st.caption("MULTI-EXCHANGE MARKET INTELLIGENCE | HYPERLIQUID + BITGET | 15M")
st.warning("Analytical signals only. No orders are placed. Validate every setup before trading.")

@st.cache_data(ttl=60, show_spinner=False)
def get_market_universe():
    rows = []

    # Hyperliquid perpetual markets
    r = requests.post(BASE_HL, json={"type": "metaAndAssetCtxs"},
                      headers=HEADERS, timeout=20)
    r.raise_for_status()
    payload = r.json()
    meta, contexts = payload[0], payload[1]
    for asset, ctx in zip(meta.get("universe", []), contexts):
        try:
            price = float(ctx.get("markPx") or 0)
            volume = float(ctx.get("dayNtlVlm") or 0)
            if price > 0 and volume >= MIN_VOLUME_USD:
                rows.append({
                    "exchange": "Hyperliquid",
                    "symbol": asset["name"],
                    "price": price,
                    "volume_24h_usd": volume,
                    "funding_rate": float(ctx.get("funding") or 0),
                })
        except (ValueError, TypeError, KeyError):
            continue

    # Bitget USDT perpetual markets
    r = requests.get(
        f"{BASE_BG}/api/v2/mix/market/tickers",
        params={"productType": "USDT-FUTURES"},
        headers=HEADERS, timeout=20
    )
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "00000":
        raise RuntimeError(f"Bitget API: {data.get('msg', 'resposta inesperada')}")
    for item in data.get("data", []):
        try:
            symbol = item["symbol"]
            price = float(item.get("lastPr") or 0)
            volume = float(item.get("quoteVolume") or 0)
            if symbol.endswith("USDT") and price > 0 and volume >= MIN_VOLUME_USD:
                rows.append({
                    "exchange": "Bitget",
                    "symbol": symbol,
                    "price": price,
                    "volume_24h_usd": volume,
                    "funding_rate": np.nan,
                })
        except (ValueError, TypeError, KeyError):
            continue

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("volume_24h_usd", ascending=False).head(MAX_MARKETS).reset_index(drop=True)

def get_candles(market):
    now = int(time.time() * 1000)
    if market["exchange"] == "Hyperliquid":
        start = now - CANDLE_LIMIT * 15 * 60 * 1000
        r = requests.post(
            BASE_HL,
            json={"type": "candleSnapshot",
                  "req": {"coin": market["symbol"], "interval": TIMEFRAME,
                          "startTime": start, "endTime": now}},
            headers=HEADERS, timeout=20
        )
        r.raise_for_status()
        raw = r.json()
        df = pd.DataFrame([{
            "timestamp": int(x["t"]),
            "open": float(x["o"]), "high": float(x["h"]),
            "low": float(x["l"]), "close": float(x["c"]),
            "volume": float(x["v"])
        } for x in raw])
    else:
        r = requests.get(
            f"{BASE_BG}/api/v2/mix/market/candles",
            params={"symbol": market["symbol"], "productType": "USDT-FUTURES",
                    "granularity": "15m", "limit": "200"},
            headers=HEADERS, timeout=20
        )
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "00000":
            raise RuntimeError(data.get("msg", "Bitget candles error"))
        raw = data.get("data", [])
        # Bitget v2 candle: [ts, open, high, low, close, baseVolume, quoteVolume]
        df = pd.DataFrame([{
            "timestamp": int(x[0]),
            "open": float(x[1]), "high": float(x[2]),
            "low": float(x[3]), "close": float(x[4]),
            "volume": float(x[6] if len(x) > 6 else x[5])
        } for x in raw])

    if df.empty:
        raise ValueError("Sem candles retornados")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    # Exclude the currently forming 15-minute candle.
    current_open = (now // (15 * 60 * 1000)) * (15 * 60 * 1000)
    df = df[df["timestamp"] < current_open].copy()
    if len(df) < 200:
        raise ValueError(f"Histórico insuficiente: {len(df)} candles fechados; mínimo 200")
    return df

def add_indicators(df):
    d = df.copy()
    close, high, low = d["close"], d["high"], d["low"]
    for period in (9, 21, 50, 200):
        d[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(),
                    (low-prev_close).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()

    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_safe = d["atr"].replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_safe
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    d["adx"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0)
    d["rvol"] = d["volume"] / d["volume"].shift(1).rolling(20).mean()
    return d

def analyze_market(market):
    candles = add_indicators(get_candles(market))
    last = candles.iloc[-1]
    prev = candles.iloc[-2]
    price = float(last["close"])
    atr = float(last["atr"])
    rsi = float(last["rsi"])
    adx = float(last["adx"])
    rvol = float(last["rvol"]) if pd.notna(last["rvol"]) else 0.0

    bull = (last.ema_9 > last.ema_21 > last.ema_50 > last.ema_200)
    bear = (last.ema_9 < last.ema_21 < last.ema_50 < last.ema_200)
    long_momentum = 52 <= rsi <= 70
    short_momentum = 30 <= rsi <= 48
    trend_strength = adx >= 20
    volume_ok = rvol >= 1.2

    score_long = (35 if bull else 0) + (20 if long_momentum else 0) + (25 if trend_strength else 0) + (20 if volume_ok else 0)
    score_short = (35 if bear else 0) + (20 if short_momentum else 0) + (25 if trend_strength else 0) + (20 if volume_ok else 0)

    signal = "AGUARDAR"
    direction_score = max(score_long, score_short)
    if bull and long_momentum and trend_strength and volume_ok:
        signal, direction_score = "LONG", score_long
    elif bear and short_momentum and trend_strength and volume_ok:
        signal, direction_score = "SHORT", score_short

    opportunity = "SEM CONFLUÊNCIA"
    if signal in ("LONG", "SHORT"):
        opportunity = "CONFIRMADO"
    elif direction_score >= 75:
        opportunity = "PRÉ-OPORTUNIDADE"

    entry = sl = tp = np.nan
    if signal == "LONG" and atr > 0:
        entry, sl, tp = price, price - SL_ATR*atr, price + TP_ATR*atr
    elif signal == "SHORT" and atr > 0:
        entry, sl, tp = price, price + SL_ATR*atr, price - TP_ATR*atr

    return {
        **market,
        "signal": signal,
        "opportunity_status": opportunity,
        "signal_score": int(direction_score),
        "score_long": int(score_long),
        "score_short": int(score_short),
        "entry": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "atr": atr,
        "risk_reward": "1:2" if signal in ("LONG", "SHORT") else "—",
        "rsi": rsi, "adx": adx, "rvol": rvol,
        "candle_timestamp": int(last["timestamp"]),
    }

with st.sidebar:
    st.header("Filtros")
    exchange_filter = st.multiselect(
        "Corretora", ["Bitget", "Hyperliquid"],
        default=["Bitget", "Hyperliquid"]
    )
    direction_filter = st.multiselect(
        "Direção", ["LONG", "SHORT", "AGUARDAR"],
        default=["LONG", "SHORT", "AGUARDAR"]
    )
    min_score = st.slider("Pontuação mínima", 0, 100, 0, 5)
    st.caption("Dados públicos | Sem execução automática")

if st.button("🔄 Atualizar scanner", type="primary"):
    st.cache_data.clear()

try:
    with st.spinner("Consultando mercados públicos das corretoras..."):
        markets = get_market_universe()
    if markets.empty:
        st.error("Nenhum mercado encontrado. Verifique as APIs.")
        st.stop()

    st.caption(f"Mercados na fila: {len(markets)} | Última consulta: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC')}")
    results, errors = [], []
    progress = st.progress(0, text="Analisando mercados...")
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(analyze_market, row.to_dict()): row.to_dict()
                   for _, row in markets.iterrows()}
        total = len(futures)
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                results.append(future.result())
            except Exception as exc:
                m = futures[future]
                errors.append({"exchange": m["exchange"], "symbol": m["symbol"], "error": str(exc)})
            progress.progress(i / total, text=f"Analisados {i}/{total} | Falhas: {len(errors)}")

    df = pd.DataFrame(results)
    if df.empty:
        st.error("Nenhum mercado pôde ser analisado.")
        st.stop()

    df = df.sort_values(["opportunity_status", "signal_score"],
                        ascending=[True, False])
    # Explicit priority order for stable ranking.
    order = {"CONFIRMADO": 0, "PRÉ-OPORTUNIDADE": 1, "SEM CONFLUÊNCIA": 2}
    df["_priority"] = df["opportunity_status"].map(order).fillna(3)
    df = df.sort_values(["_priority", "signal_score"], ascending=[True, False]).drop(columns="_priority")

    confirmed = df[df["signal"].isin(["LONG", "SHORT"])]
    longs = int((df["signal"] == "LONG").sum())
    shorts = int((df["signal"] == "SHORT").sum())
    pre = int((df["opportunity_status"] == "PRÉ-OPORTUNIDADE").sum())

    st.markdown("## VEXA — MARKET OVERVIEW")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("MERCADOS ANALISADOS", len(df))
    c2.metric("LONG", longs)
    c3.metric("SHORT", shorts)
    c4.metric("PRÉ-OPORTUNIDADES", pre)
    c5.metric("FALHAS", len(errors))

    view = df[
        df["exchange"].isin(exchange_filter)
        & df["signal"].isin(direction_filter)
        & (df["signal_score"] >= min_score)
    ].copy()

    display_cols = [
        "exchange","symbol","signal","opportunity_status","signal_score",
        "price","volume_24h_usd","funding_rate","rsi","adx","rvol",
        "entry","stop_loss","take_profit","risk_reward"
    ]
    st.markdown("## VEXA — MARKET SCANNER")
    st.dataframe(view[display_cols], use_container_width=True, hide_index=True)

    st.markdown("## OPORTUNIDADES CONFIRMADAS")
    if confirmed.empty:
        st.info("Nenhuma oportunidade confirmada neste ciclo.")
    else:
        st.dataframe(confirmed[[
            "exchange","symbol","signal","signal_score","price",
            "entry","stop_loss","take_profit","atr","risk_reward",
            "rsi","adx","rvol"
        ]], use_container_width=True, hide_index=True)

    st.markdown("## PRÉ-OPORTUNIDADES")
    pre_df = df[df["opportunity_status"] == "PRÉ-OPORTUNIDADE"]
    if pre_df.empty:
        st.info("Nenhuma pré-oportunidade neste ciclo.")
    else:
        st.dataframe(pre_df[[
            "exchange","symbol","score_long","score_short","signal_score",
            "rsi","adx","rvol"
        ]], use_container_width=True, hide_index=True)

    if errors:
        with st.expander(f"Detalhes das {len(errors)} falhas"):
            st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)

    st.caption("Entrada é preço de referência do último candle fechado, não preço garantido de execução. TP/SL são níveis teóricos por ATR (1,5x / 3x). Sinais não são recomendação financeira.")
except Exception as exc:
    st.error(f"Falha ao carregar o VEXA: {exc}")
    st.info("Tente atualizar a aplicação. Se o erro persistir, envie a mensagem completa.")
