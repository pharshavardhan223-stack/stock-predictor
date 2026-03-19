# ==============================================================
# backend/routes/portfolio_routes.py
# AI Portfolio Advisor — Markowitz Optimization + Sharpe Ratio
# ==============================================================

import json
import traceback
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, session

portfolio_bp = Blueprint("portfolio", __name__)


# ── Page route ────────────────────────────────────────────────
@portfolio_bp.route("/portfolio-advisor")
def portfolio_advisor():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("portfolio_advisor.html")


# ── API route ─────────────────────────────────────────────────
@portfolio_bp.route("/api/portfolio-advisor", methods=["POST"])
def api_portfolio_advisor():
    """
    Real portfolio optimization:
      1. Fetch 2-year daily price history via yfinance
      2. Compute annualized returns, volatility, covariance matrix
      3. Markowitz Mean-Variance optimization via scipy (30 random starts)
      4. Maximize Sharpe Ratio subject to weight bounds per risk level
      5. Return per-stock allocation + portfolio-level metrics
    """
    try:
        import yfinance as yf
        from scipy.optimize import minimize
    except ImportError:
        return json.dumps({
            "error": "Missing packages. Run:  pip install yfinance scipy"
        }), 500

    try:
        data        = request.get_json(force=True)
        amount      = float(data.get("amount", 100000))
        risk_level  = data.get("risk", "medium").lower()   # low / medium / high
        horizon_yrs = float(data.get("horizon", 1))
        market      = data.get("market", "IN")             # IN  / US
        custom_syms = [s.strip().upper() for s in data.get("symbols", []) if s.strip()]

        # ── Stock universes ───────────────────────────────────
        UNIVERSE = {
            "IN": [
                "RELIANCE.NS", "INFY.NS",     "TCS.NS",       "HDFCBANK.NS",
                "ICICIBANK.NS","ITC.NS",       "WIPRO.NS",     "LT.NS",
                "SBIN.NS",     "AXISBANK.NS",  "BAJFINANCE.NS","HINDUNILVR.NS",
                "TATAMOTORS.NS","SUNPHARMA.NS","ONGC.NS",
            ],
            "US": [
                "AAPL", "MSFT",  "GOOGL", "AMZN", "NVDA",
                "JPM",  "JNJ",   "V",     "PG",   "UNH",
                "META", "TSLA",  "BRK-B", "HD",   "XOM",
            ],
        }
        symbols = custom_syms if custom_syms else UNIVERSE.get(market, UNIVERSE["IN"])

        # ── Risk parameters ───────────────────────────────────
        RISK_CFG = {
            "low":    {"max_alloc": 0.25, "min_stocks": 6,  "max_vol": 0.18},
            "medium": {"max_alloc": 0.35, "min_stocks": 5,  "max_vol": 0.28},
            "high":   {"max_alloc": 0.45, "min_stocks": 4,  "max_vol": 0.99},
        }
        rp      = RISK_CFG.get(risk_level, RISK_CFG["medium"])
        rf_rate = 0.065 if market == "IN" else 0.045   # risk-free rate

        # ── Fetch price data ──────────────────────────────────
        period = "2y" if horizon_yrs >= 1 else "1y"
        raw    = yf.download(symbols, period=period, auto_adjust=True, progress=False)["Close"]

        # Drop stocks with too many missing values, then drop remaining NaN rows
        raw = raw.dropna(axis=1, thresh=int(len(raw) * 0.85)).dropna()

        if raw.shape[1] < 3:
            return json.dumps({
                "error": "Not enough valid stock data returned. Try different symbols or check your internet connection."
            }), 400

        available = list(raw.columns)

        # ── Returns & covariance ──────────────────────────────
        returns     = raw.pct_change().dropna()
        ann_returns = returns.mean() * 252
        ann_vol     = returns.std()  * np.sqrt(252)
        cov_matrix  = returns.cov() * 252

        # Filter by max volatility for low/medium risk
        if risk_level != "high":
            eligible = [s for s in available if ann_vol[s] <= rp["max_vol"]]
            if len(eligible) < 3:
                eligible = available   # fallback — keep all
        else:
            eligible = available

        ret_e  = ann_returns[eligible].values
        cov_e  = cov_matrix.loc[eligible, eligible].values
        n      = len(eligible)

        # ── Markowitz optimization (maximize Sharpe) ──────────
        def neg_sharpe(w):
            port_ret = float(np.dot(w, ret_e))
            port_vol = float(np.sqrt(w @ cov_e @ w))
            return -(port_ret - rf_rate) / (port_vol + 1e-9)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds      = [(0.03, rp["max_alloc"])] * n
        best        = None

        for _ in range(40):                                   # 40 random starts
            w0 = np.random.dirichlet(np.ones(n))
            r  = minimize(neg_sharpe, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 600, "ftol": 1e-12})
            if r.success and (best is None or r.fun < best.fun):
                best = r

        # ── Build weight dict ─────────────────────────────────
        if best is None:
            # Fallback: top-5 by individual Sharpe ratio
            sharpes = (ann_returns[eligible] - rf_rate) / (ann_vol[eligible] + 1e-9)
            top5    = sharpes.nlargest(5).index.tolist()
            weights = {s: round(1.0 / len(top5), 4) for s in top5}
        else:
            raw_w = best.x.copy()
            raw_w[raw_w < 0.02] = 0.0          # zero tiny allocations
            raw_w /= raw_w.sum()

            # Keep top N meaningful allocations
            idx_sorted = np.argsort(raw_w)[::-1]
            keep       = [i for i in idx_sorted if raw_w[i] >= 0.04][:10]
            if len(keep) < rp["min_stocks"]:
                keep = list(idx_sorted[:rp["min_stocks"]])

            final_w = np.zeros(n)
            for i in keep:
                final_w[i] = raw_w[i]
            final_w /= final_w.sum()

            weights = {
                eligible[i]: round(float(final_w[i]), 4)
                for i in range(n) if final_w[i] > 0.001
            }

        # ── Portfolio-level metrics ───────────────────────────
        sel      = list(weights.keys())
        w_arr    = np.array([weights[s] for s in sel])
        ret_sel  = ann_returns[sel].values
        cov_sel  = cov_matrix.loc[sel, sel].values

        port_ret     = float(np.dot(w_arr, ret_sel))
        port_vol     = float(np.sqrt(w_arr @ cov_sel @ w_arr))
        sharpe_ratio = round((port_ret - rf_rate) / (port_vol + 1e-9), 2)
        port_ret_pct = round(port_ret * 100, 2)
        port_vol_pct = round(port_vol * 100, 2)

        # ── Per-stock details ─────────────────────────────────
        # Market benchmark returns (equal-weight) for beta calc
        mkt_ret = returns[available].mean(axis=1)
        mkt_var = float(mkt_ret.var()) + 1e-9

        allocations = []
        for sym in sorted(weights, key=lambda s: -weights[s]):
            w          = weights[sym]
            alloc_amt  = round(amount * w, 2)
            r_pct      = round(float(ann_returns[sym]) * 100, 2)
            v_pct      = round(float(ann_vol[sym])     * 100, 2)
            sh         = round((float(ann_returns[sym]) - rf_rate) /
                               (float(ann_vol[sym]) + 1e-9), 2)
            last_price = float(raw[sym].iloc[-1])
            shares     = int(alloc_amt / last_price) if last_price > 0 else 0
            beta       = round(float(returns[sym].cov(mkt_ret)) / mkt_var, 2) \
                         if sym in returns.columns else 1.0

            allocations.append({
                "symbol":      sym.replace(".NS", "").replace(".BO", ""),
                "full_symbol": sym,
                "weight_pct":  round(w * 100, 1),
                "amount":      alloc_amt,
                "exp_return":  r_pct,
                "volatility":  v_pct,
                "sharpe":      sh,
                "last_price":  round(last_price, 2),
                "shares":      shares,
                "beta":        beta,
            })

        # ── Sector map ────────────────────────────────────────
        SECTOR = {
            "RELIANCE.NS":  "Energy",   "INFY.NS":       "IT",
            "TCS.NS":       "IT",       "HDFCBANK.NS":   "Finance",
            "ICICIBANK.NS": "Finance",  "ITC.NS":        "FMCG",
            "WIPRO.NS":     "IT",       "LT.NS":         "Infra",
            "SBIN.NS":      "Finance",  "AXISBANK.NS":   "Finance",
            "BAJFINANCE.NS":"Finance",  "HINDUNILVR.NS": "FMCG",
            "TATAMOTORS.NS":"Auto",     "SUNPHARMA.NS":  "Pharma",
            "ONGC.NS":      "Energy",   "AAPL":          "Tech",
            "MSFT":         "Tech",     "GOOGL":         "Tech",
            "AMZN":         "Tech",     "NVDA":          "Tech",
            "JPM":          "Finance",  "JNJ":           "Healthcare",
            "V":            "Finance",  "PG":            "Consumer",
            "UNH":          "Healthcare","META":         "Tech",
            "TSLA":         "Auto",     "BRK-B":        "Finance",
            "HD":           "Consumer", "XOM":          "Energy",
        }
        sector_alloc = {}
        for a in allocations:
            sec = SECTOR.get(a["full_symbol"], "Other")
            sector_alloc[sec] = round(sector_alloc.get(sec, 0) + a["weight_pct"], 1)

        # ── Projection ────────────────────────────────────────
        projected_value = round(amount * ((1 + port_ret) ** horizon_yrs), 2)
        projected_gain  = round(projected_value - amount, 2)
        gain_pct        = round((projected_gain / amount) * 100, 1)

        # ── Risk label ────────────────────────────────────────
        actual_risk = (
            "Low"    if port_vol_pct < 15 else
            "Medium" if port_vol_pct < 25 else
            "High"
        )

        # ── AI reasoning text ─────────────────────────────────
        top3    = [a["symbol"] for a in allocations[:3]]
        n_sec   = len(sector_alloc)
        sec_str = ", ".join(
            f"{k} {v}%" for k, v in
            sorted(sector_alloc.items(), key=lambda x: -x[1])[:4]
        )
        rebal   = "quarterly" if horizon_yrs <= 1 else "every 6 months"
        reasoning = (
            f"Portfolio built using Markowitz Mean-Variance optimization across "
            f"{len(available)} real stocks ({period} of market data). "
            f"Ran 40 random optimization starts to find the global Sharpe maximum. "
            f"Top holdings — {', '.join(top3)} — selected for highest risk-adjusted returns. "
            f"Diversified across {n_sec} sector(s): {sec_str}. "
            f"Risk-free rate: {rf_rate*100:.1f}% ({"RBI" if market=="IN" else "Fed"} benchmark). "
            f"Portfolio rebalancing recommended {rebal}."
        )

        return json.dumps({
            "success":          True,
            "allocations":      allocations,
            "portfolio_return": port_ret_pct,
            "portfolio_vol":    port_vol_pct,
            "sharpe_ratio":     sharpe_ratio,
            "projected_value":  projected_value,
            "projected_gain":   projected_gain,
            "gain_pct":         gain_pct,
            "actual_risk":      actual_risk,
            "sector_alloc":     sector_alloc,
            "reasoning":        reasoning,
            "stocks_analyzed":  len(available),
            "market":           market,
            "amount":           amount,
            "horizon":          horizon_yrs,
        })

    except Exception as e:
        return json.dumps({
            "error":   str(e),
            "details": traceback.format_exc()
        }), 500