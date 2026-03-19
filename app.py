"""
=============================================================================
  DUAL MOMENTUM ETF — WEB SERVICE  (Render FREE tier)
=============================================================================
  Adattamenti rispetto alla versione Starter:
  1. Gunicorn con --timeout 0  → nessun kill sul worker dopo 30s
  2. Thread daemon per il backtest → risposta HTTP immediata, polling dal browser
  3. Self-ping ogni 10 minuti  → evita lo spin-down dopo 15 min di inattività
  4. Zero scritture su disco    → tutto in RAM (filesystem effimero sul free tier)
  5. Cold-start banner          → avvisa l'utente del risveglio ~60s
=============================================================================
  Endpoints:
    GET  /          → dashboard con form parametri + cold-start warning
    POST /run       → avvia thread backtest, risponde subito 202
    GET  /status    → polling JSON {running, progress, done, error}
    GET  /results   → pagina risultati completa (grafici base64 inline)
    GET  /api/results → JSON puro per integrazioni esterne
    GET  /health    → health check + self-ping trigger
=============================================================================
"""

import os, io, base64, warnings, threading, time
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # OBBLIGATORIO: server headless, niente display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import yfinance as yf
from flask import Flask, request, jsonify, render_template_string, redirect

app = Flask(__name__)

# =============================================================================
#  STATO IN MEMORIA  (niente disco: il free tier lo cancella ad ogni riavvio)
# =============================================================================

_lock    = threading.Lock()
_status  = {"running": False, "progress": "", "done": False, "error": ""}
_results = {}          # dizionario con stats + grafici base64
_boot_at = datetime.now()

# =============================================================================
#  SELF-PING  (evita lo spin-down dopo 15 min di inattività)
#  Parte in background all'avvio del processo, pinga /health ogni 10 minuti.
# =============================================================================

def _self_ping():
    """Daemon thread: pinga il proprio /health ogni 10 minuti."""
    time.sleep(30)   # aspetta che Gunicorn sia pronto
    port = os.environ.get("PORT", "10000")
    url  = f"http://0.0.0.0:{port}/health"
    while True:
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=10)
        except Exception:
            pass
        time.sleep(600)   # 10 minuti

threading.Thread(target=_self_ping, daemon=True, name="self-ping").start()

# =============================================================================
#  UNIVERSO ETF
# =============================================================================

ETF_UNIVERSE = {
    "EXV1.DE": {"name": "Banks EU 600",      "sector": "Bancario",    "color": "#185FA5"},
    "EXH7.DE": {"name": "Oil & Gas EU 600",  "sector": "Energia",     "color": "#BA7517"},
    "EXV4.DE": {"name": "Healthcare EU 600", "sector": "Healthcare",  "color": "#1D9E75"},
    "EXH4.DE": {"name": "Industrial EU 600", "sector": "Industriali", "color": "#534AB7"},
    "EXV5.DE": {"name": "Technology EU 600", "sector": "Tecnologia",  "color": "#D85A30"},
    "EXH9.DE": {"name": "Utilities EU 600",  "sector": "Utilities",   "color": "#639922"},
    "EXH5.DE": {"name": "Insurance EU 600",  "sector": "Assicur.",    "color": "#993556"},
    "EXH2.DE": {"name": "Basic Resources",   "sector": "Risorse",     "color": "#5F5E5A"},
}
CASH_ETF  = {"ticker": "XEON.DE", "name": "EUR Overnight"}
BENCHMARK = {"ticker": "EXSA.DE", "name": "STOXX Europe 600", "color": "#888780"}

# =============================================================================
#  LOGICA BACKTEST
# =============================================================================

def _download(tickers, start, end, prog_cb=None):
    out = {}
    for t in tickers:
        try:
            d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if d.empty:
                continue
            close = d["Close"].squeeze() if isinstance(d.columns, pd.MultiIndex) else d["Close"]
            out[t] = close
            if prog_cb:
                prog_cb(f"Scaricato {t}  ({len(close)} barre)")
        except Exception:
            pass
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).ffill().dropna(how="all")


def _momentum(prices, lb):
    m = prices.resample("ME").last()
    return m, m.pct_change(periods=lb)


def _backtest(prices, monthly_p, monthly_m, etf_tickers, cash_ticker, cfg):
    lb      = cfg["lookback_months"]
    top_n   = cfg["top_n"]
    tc      = cfg["transaction_cost"]
    thresh  = cfg.get("cash_threshold", 0.0)
    capital = float(cfg["initial_capital"])

    rdates   = monthly_p.index[lb:]
    holdings = {}
    port_d   = {}
    trades   = []
    allocs   = []
    prev_d   = prices.index[0]
    port_d[prev_d] = capital

    for i, rd in enumerate(rdates):
        mom      = monthly_m.loc[rd, etf_tickers].dropna()
        positive = mom[mom > thresh].sort_values(ascending=False)
        selected = list(positive.index[:top_n])
        in_cash  = len(selected) == 0
        if in_cash and cash_ticker and cash_ticker in prices.columns:
            selected = [cash_ticker]

        old, new = set(holdings), set(selected)
        n_trades = len(old.symmetric_difference(new))
        capital -= capital * tc * n_trades / max(len(selected), 1) if n_trades else 0

        if old != new:
            for t in old - new:
                trades.append({"data": str(rd.date()), "azione": "SELL",
                               "ticker": t, "nome": ETF_UNIVERSE.get(t, {}).get("name", t)})
            for t in new - old:
                name = CASH_ETF["name"] if t == cash_ticker else ETF_UNIVERSE.get(t, {}).get("name", t)
                trades.append({"data": str(rd.date()), "azione": "BUY",
                               "ticker": t, "nome": name})

        w = 1.0 / len(selected) if selected else 0
        holdings = {t: w for t in selected}
        row = {"date": str(rd.date()), "in_cash": in_cash}
        for t in etf_tickers:
            row[t] = round(holdings.get(t, 0), 4)
        allocs.append(row)

        next_rd   = rdates[i+1] if i+1 < len(rdates) else prices.index[-1]
        day_range = prices.loc[rd:next_rd].index
        prev_val  = capital

        for d in day_range[1:]:
            dr = 0.0
            for t, ww in holdings.items():
                if t in prices.columns:
                    pp = prices.loc[prev_d, t] if prev_d in prices.index else np.nan
                    cp = prices.loc[d, t]
                    if pd.notna(pp) and pd.notna(cp) and pp > 0:
                        dr += ww * (cp / pp - 1)
            prev_val *= (1 + dr)
            port_d[d] = prev_val
            prev_d = d

        capital = prev_val

    portfolio = pd.Series(port_d).sort_index()
    portfolio = portfolio[~portfolio.index.duplicated(keep="last")]
    monthly_r = portfolio.resample("ME").last().pct_change().dropna()
    return {"portfolio": portfolio, "monthly_returns": monthly_r,
            "trades": trades, "allocations": allocs}


def _stats(eq, rf=0.03):
    r  = eq.pct_change().dropna()
    mo = eq.resample("ME").last().pct_change().dropna()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    tot  = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1
    vol  = r.std() * np.sqrt(252)
    rfd  = (1+rf)**(1/252) - 1
    exc  = r - rfd
    sh   = exc.mean()/exc.std()*np.sqrt(252) if exc.std() > 0 else 0
    neg  = r[r < rfd]
    so   = exc.mean()/neg.std()*np.sqrt(252) if len(neg) > 0 and neg.std() > 0 else 0
    pk   = eq.expanding().max()
    dd   = (eq - pk) / pk
    mdd  = dd.min()
    cal  = cagr / abs(mdd) if mdd else 0
    return {
        "rendimento_totale": round(tot*100, 2),
        "cagr":              round(cagr*100, 2),
        "volatilita":        round(vol*100, 2),
        "sharpe":            round(sh, 2),
        "sortino":           round(so, 2),
        "calmar":            round(cal, 2),
        "max_drawdown":      round(mdd*100, 2),
        "win_rate":          round(float((mo > 0).mean())*100, 1),
        "miglior_mese":      round(float(mo.max())*100, 2),
        "peggior_mese":      round(float(mo.min())*100, 2),
    }

# =============================================================================
#  GRAFICI → base64 PNG  (tutto in RAM, niente file su disco)
# =============================================================================

def _fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

_STYLE = {
    "figure.facecolor": "#FAFAF8", "axes.facecolor": "#FAFAF8",
    "axes.grid": True, "grid.alpha": 0.25, "grid.color": "#CCCCCC",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "font.family": "sans-serif", "font.size": 10,
}

def _chart_equity(strat, bench):
    plt.rcParams.update(_STYLE)
    fig, ax = plt.subplots(figsize=(12, 3.8), facecolor="#FAFAF8")
    ax.set_facecolor("#FAFAF8")
    s = strat / strat.iloc[0] * 100
    b = bench / bench.iloc[0] * 100
    ax.fill_between(s.index, s, 100, alpha=0.1, color="#185FA5")
    ax.plot(s.index, s, color="#185FA5", lw=2,   label="Dual Momentum")
    ax.plot(b.index, b, color="#888780", lw=1.5, ls="--", label=BENCHMARK["name"])
    ax.axhline(100, color="#CCCCCC", lw=0.8)
    ax.legend(fontsize=9, framealpha=0)
    ax.set_title("Equity curve (base 100)", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}"))
    fig.tight_layout()
    return _fig_b64(fig)

def _chart_dd(strat, bench):
    plt.rcParams.update(_STYLE)
    fig, ax = plt.subplots(figsize=(12, 3), facecolor="#FAFAF8")
    ax.set_facecolor("#FAFAF8")
    def dd(x): pk=x.expanding().max(); return (x-pk)/pk*100
    ax.fill_between(strat.index, dd(strat), 0, alpha=0.25, color="#E24B4A")
    ax.plot(strat.index, dd(strat), color="#E24B4A", lw=1.5, label="Strategia")
    ax.plot(bench.index, dd(bench), color="#888780", lw=1,   ls="--", label="Benchmark")
    ax.legend(fontsize=9, framealpha=0)
    ax.set_title("Drawdown %", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    fig.tight_layout()
    return _fig_b64(fig)

def _chart_heatmap(monthly_rets):
    plt.rcParams.update(_STYLE)
    mr = monthly_rets.copy()
    mr.index = pd.to_datetime(mr.index)
    df = pd.DataFrame({"year": mr.index.year, "month": mr.index.month,
                        "ret": mr.values * 100})
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Gen","Feb","Mar","Apr","Mag","Giu",
                     "Lug","Ago","Set","Ott","Nov","Dic"]
    h = max(3, len(pivot) * 0.42)
    fig, ax = plt.subplots(figsize=(12, h), facecolor="#FAFAF8")
    ax.set_facecolor("#FAFAF8")
    cmap = LinearSegmentedColormap.from_list("rg",["#C0392B","#FAFAF8","#1D9E75"],N=256)
    sns.heatmap(pivot, ax=ax, cmap=cmap, center=0, annot=True, fmt=".1f",
                annot_kws={"size":7}, linewidths=0.4, linecolor="#E0E0E0",
                cbar_kws={"shrink": 0.5, "label": "Rendimento %"}, vmin=-10, vmax=10)
    ax.set_title("Rendimenti mensili %", fontsize=11, fontweight="bold")
    ax.set_ylabel("Anno"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return _fig_b64(fig)

# =============================================================================
#  JOB THREAD  (lavoro pesante → non blocca Gunicorn)
# =============================================================================

def _run_job(cfg):
    global _status, _results

    def prog(msg):
        with _lock:
            _status["progress"] = msg

    with _lock:
        _status = {"running": True, "progress": "Avvio…", "done": False, "error": ""}

    try:
        etf_t  = list(ETF_UNIVERSE.keys())
        cash_t = CASH_ETF["ticker"]
        bm_t   = BENCHMARK["ticker"]

        prog("Download prezzi ETF…")
        prices = _download(etf_t + [cash_t, bm_t],
                           cfg["start_date"], cfg["end_date"], prog)

        if prices.empty:
            raise ValueError("Nessun prezzo scaricato — Yahoo Finance non raggiungibile?")

        avail   = [t for t in etf_t if t in prices.columns]
        if not avail:
            raise ValueError("Nessun ETF dell'universo disponibile.")

        bm_p    = prices.get(bm_t)
        cash_av = cash_t if cash_t in prices.columns else None
        use_col = [t for t in avail + ([cash_t] if cash_av else []) if t in prices.columns]
        etf_px  = prices[use_col]

        prog("Calcolo momentum…")
        mp, mm = _momentum(etf_px[avail], cfg["lookback_months"])

        prog("Backtest in corso…")
        res = _backtest(etf_px, mp, mm, avail, cash_av, cfg)

        strat = res["portfolio"]
        if bm_p is not None:
            ba = bm_p.reindex(strat.index).ffill()
            bench_eq = ba / ba.iloc[0] * cfg["initial_capital"]
        else:
            bench_eq = pd.Series([cfg["initial_capital"]]*len(strat), index=strat.index)

        prog("Calcolo statistiche…")
        ss = _stats(strat)
        bs = _stats(bench_eq)

        prog("Generazione grafici…")
        eq_b64 = _chart_equity(strat, bench_eq)
        dd_b64 = _chart_dd(strat, bench_eq)
        hm_b64 = _chart_heatmap(res["monthly_returns"])

        last_a   = res["allocations"][-1] if res["allocations"] else {}
        in_cash  = last_a.get("in_cash", True)
        active   = [ETF_UNIVERSE[t]["name"] for t in avail if last_a.get(t, 0) > 0]
        if not active:
            active = ["Monetario (XEON)"]
        cash_pct = round(sum(1 for a in res["allocations"] if a.get("in_cash"))
                         / max(len(res["allocations"]), 1) * 100, 1)

        with _lock:
            _results = {
                "config":         cfg,
                "stats_strat":    ss,
                "stats_bench":    bs,
                "trades":         res["trades"][-20:],
                "allocations":    res["allocations"][-12:],
                "active":         active,
                "in_cash":        in_cash,
                "cash_pct":       cash_pct,
                "chart_equity":   eq_b64,
                "chart_dd":       dd_b64,
                "chart_heatmap":  hm_b64,
                "run_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            _status = {"running": False, "progress": "Completato.", "done": True, "error": ""}

    except Exception as e:
        with _lock:
            _status = {"running": False, "progress": "", "done": False, "error": str(e)}

# =============================================================================
#  HTML TEMPLATES
# =============================================================================

_INDEX = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dual Momentum ETF</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f3;color:#1a1a18;min-height:100vh}
.topbar{background:#fff;border-bottom:1px solid #e8e8e4;padding:.85rem 2rem;display:flex;align-items:center;gap:1rem}
.topbar h1{font-size:1.1rem;font-weight:600}
.topbar .pill{font-size:.75rem;color:#888;background:#f0f0ec;padding:3px 10px;border-radius:20px}
.topbar a{margin-left:auto;font-size:.82rem;color:#185FA5;text-decoration:none}
.wrap{max-width:860px;margin:2rem auto;padding:0 1.5rem}
.card{background:#fff;border:1px solid #e8e8e4;border-radius:12px;padding:1.4rem;margin-bottom:1.2rem}
.card h2{font-size:.82rem;font-weight:600;color:#555;text-transform:uppercase;letter-spacing:.05em;margin-bottom:1rem}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
label{display:block;font-size:.78rem;color:#777;margin-bottom:4px}
input,select{width:100%;padding:7px 9px;border:1px solid #ddd;border-radius:7px;font-size:.88rem;background:#fafaf8;color:#1a1a18;outline:none}
input:focus,select:focus{border-color:#185FA5}
.btn{width:100%;padding:10px;background:#185FA5;color:#fff;border:none;border-radius:8px;font-size:.95rem;cursor:pointer;font-weight:500;margin-top:.6rem;transition:background .15s}
.btn:hover{background:#0c447c}
.btn:disabled{background:#b0b0b0;cursor:not-allowed}
.bar-wrap{background:#f0f0ec;border-radius:4px;height:6px;margin:.6rem 0 .3rem;overflow:hidden}
.bar{height:6px;background:#185FA5;border-radius:4px;width:0%;transition:width .4s}
.status{padding:.65rem 1rem;border-radius:8px;font-size:.84rem;margin-top:.8rem;display:none}
.running{background:#EAF3DE;color:#3B6D11;display:block}
.done{background:#E6F1FB;color:#185FA5;display:block}
.errs{background:#FCEBEB;color:#A32D2D;display:block}
.cold{background:#FAEEDA;color:#854F0B;border:1px solid #F9CB42;border-radius:8px;padding:.7rem 1rem;font-size:.84rem;margin-bottom:1rem}
.etfgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:8px;margin-top:.5rem}
.etfbadge{padding:6px 9px;border-radius:8px;font-size:.76rem;border:1px solid #e8e8e4;background:#fafaf8}
.etfbadge strong{display:block;font-size:.8rem}
footer{text-align:center;font-size:.72rem;color:#bbb;padding:2rem 0}
@media(max-width:600px){.g3{grid-template-columns:1fr 1fr}.g2{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="topbar">
  <div>
    <h1>Dual Momentum ETF Rotation</h1>
  </div>
  <span class="pill">Borsa Italiana · Free tier</span>
  {% if has_results %}<a href="/results">→ Ultimi risultati</a>{% endif %}
</div>

<div class="wrap">

  {% if cold_start %}
  <div class="cold">
    ⏱ Il servizio era in sleep (piano Free di Render). Si è risvegliato — puoi procedere normalmente.
  </div>
  {% endif %}

  <div class="card">
    <h2>Parametri backtest</h2>
    <form id="f">
      <div class="g3">
        <div><label>Data inizio</label><input type="date" name="start_date" value="2009-01-01"></div>
        <div><label>Lookback momentum</label>
          <select name="lookback_months">
            <option value="3">3 mesi</option>
            <option value="6" selected>6 mesi</option>
            <option value="9">9 mesi</option>
            <option value="12">12 mesi</option>
          </select>
        </div>
        <div><label>Top N ETF</label>
          <select name="top_n">
            <option value="1">1 ETF</option>
            <option value="2" selected>2 ETF</option>
            <option value="3">3 ETF</option>
            <option value="4">4 ETF</option>
          </select>
        </div>
        <div><label>Capitale iniziale (€)</label><input type="number" name="initial_capital" value="10000" min="1000" step="1000"></div>
        <div><label>Costo per trade</label>
          <select name="transaction_cost">
            <option value="0.0005">0.05%</option>
            <option value="0.001" selected>0.10%</option>
            <option value="0.002">0.20%</option>
          </select>
        </div>
        <div><label>Soglia cash</label><input type="number" name="cash_threshold" value="0" step="0.01" min="-0.2" max="0.1"></div>
      </div>
      <button class="btn" type="submit" id="runBtn">▶ Esegui Backtest</button>
    </form>
    <div class="bar-wrap" id="barWrap" style="display:none"><div class="bar" id="bar"></div></div>
    <div class="status" id="statusBox"></div>
  </div>

  <div class="card">
    <h2>Universo ETF (8 settori · iShares STOXX Europe 600 · Borsa Italiana)</h2>
    <div class="etfgrid">
      {% for ticker, info in etfs.items() %}
      <div class="etfbadge" style="border-left:3px solid {{info.color}}">
        <strong>{{info.name}}</strong>
        <span style="color:#999">{{ticker}} · {{info.sector}}</span>
      </div>
      {% endfor %}
    </div>
    <p style="font-size:.76rem;color:#999;margin-top:.75rem">
      Safe haven → <strong>XEON.MI</strong> (EUR Overnight) · Benchmark → <strong>EXSA.MI</strong>
    </p>
  </div>

</div>
<footer>Dati via Yahoo Finance · Solo scopo didattico · Non è consulenza finanziaria</footer>

<script>
const form   = document.getElementById('f');
const btn    = document.getElementById('runBtn');
const box    = document.getElementById('statusBox');
const barW   = document.getElementById('barWrap');
const bar    = document.getElementById('bar');
let barPct   = 0, barTimer = null;

function animBar() {
  barTimer = setInterval(() => {
    barPct = Math.min(barPct + (100 - barPct) * 0.04, 92);
    bar.style.width = barPct + '%';
  }, 800);
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  btn.disabled = true;
  barW.style.display = 'block';
  barPct = 5; bar.style.width = '5%';
  animBar();
  box.className = 'status running';
  box.textContent = 'Avvio backtest…';

  const data = Object.fromEntries(new FormData(form));
  await fetch('/run', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});

  const poll = setInterval(async () => {
    try {
      const r = await fetch('/status');
      const s = await r.json();
      if (s.running) {
        box.textContent = s.progress || 'In elaborazione…';
      } else if (s.done) {
        clearInterval(poll); clearInterval(barTimer);
        bar.style.width = '100%';
        box.className = 'status done';
        box.innerHTML = '✓ Completato! <a href="/results">→ Vedi risultati</a>';
        btn.disabled = false;
      } else if (s.error) {
        clearInterval(poll); clearInterval(barTimer);
        bar.style.width = '0%'; barW.style.display = 'none';
        box.className = 'status errs';
        box.textContent = 'Errore: ' + s.error;
        btn.disabled = false;
      }
    } catch(_) {}
  }, 2000);
});
</script>
</body></html>
"""

_RESULTS = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Risultati — Dual Momentum ETF</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f3;color:#1a1a18}
.topbar{background:#fff;border-bottom:1px solid #e8e8e4;padding:.85rem 2rem;display:flex;align-items:center;gap:.8rem}
.topbar h1{font-size:1.05rem;font-weight:600}
.meta{font-size:.75rem;color:#999;margin-top:2px}
.topbar a{margin-left:auto;font-size:.82rem;color:#185FA5;text-decoration:none}
.wrap{max-width:1060px;margin:2rem auto;padding:0 1.5rem}
.card{background:#fff;border:1px solid #e8e8e4;border-radius:12px;padding:1.4rem;margin-bottom:1.2rem}
.card h2{font-size:.8rem;font-weight:600;color:#666;text-transform:uppercase;letter-spacing:.05em;margin-bottom:1rem}
.kpi{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.kpi-box{background:#f5f5f3;border-radius:8px;padding:.8rem 1rem}
.kpi-box .lbl{font-size:.7rem;color:#999;margin-bottom:3px}
.kpi-box .val{font-size:1.3rem;font-weight:600}
.kpi-box .sub{font-size:.7rem;color:#bbb;margin-top:2px}
.pos{color:#1D9E75}.neg{color:#E24B4A}
.chart-img{width:100%;border-radius:6px;display:block;margin-top:.4rem}
table{width:100%;border-collapse:collapse;font-size:.83rem}
th{text-align:left;padding:6px 10px;background:#f5f5f3;font-weight:500;color:#777;font-size:.76rem}
td{padding:6px 10px;border-bottom:1px solid #f0f0ec}
td:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:.72rem;font-weight:500}
.cash{background:#FAEEDA;color:#854F0B}.inv{background:#EAF3DE;color:#3B6D11}
.trow{display:grid;grid-template-columns:100px 55px 100px 1fr;gap:6px;padding:5px 0;
      border-bottom:1px solid #f5f5f3;font-size:.78rem;align-items:center}
.buy{color:#1D9E75;font-weight:600}.sell{color:#E24B4A;font-weight:600}
footer{text-align:center;font-size:.72rem;color:#bbb;padding:2rem 0}
@media(max-width:700px){.kpi{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<div class="topbar">
  <div>
    <h1>Dual Momentum ETF — Risultati</h1>
    <div class="meta">Lookback {{cfg.lookback_months}}m · Top-{{cfg.top_n}} · {{cfg.start_date}} → {{run_at}}</div>
  </div>
  <a href="/">← Nuovo backtest</a>
</div>

<div class="wrap">

  <div class="card">
    <h2>Performance vs {{bm_name}}</h2>
    <div class="kpi">
      <div class="kpi-box">
        <div class="lbl">CAGR annualizzato</div>
        <div class="val pos">{{ss.cagr}}%</div>
        <div class="sub">Benchmark: {{bs.cagr}}%</div>
      </div>
      <div class="kpi-box">
        <div class="lbl">Max drawdown</div>
        <div class="val neg">{{ss.max_drawdown}}%</div>
        <div class="sub">Benchmark: {{bs.max_drawdown}}%</div>
      </div>
      <div class="kpi-box">
        <div class="lbl">Sharpe ratio</div>
        <div class="val">{{ss.sharpe}}</div>
        <div class="sub">Sortino: {{ss.sortino}}</div>
      </div>
      <div class="kpi-box">
        <div class="lbl">Win rate mensile</div>
        <div class="val">{{ss.win_rate}}%</div>
        <div class="sub">Cash: {{cash_pct}}% dei mesi</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Equity curve</h2>
    <img class="chart-img" src="data:image/png;base64,{{chart_equity}}" alt="equity">
  </div>

  <div class="card">
    <h2>Drawdown</h2>
    <img class="chart-img" src="data:image/png;base64,{{chart_dd}}" alt="drawdown">
  </div>

  <div class="card">
    <h2>Rendimenti mensili</h2>
    <img class="chart-img" src="data:image/png;base64,{{chart_heatmap}}" alt="heatmap">
  </div>

  <div class="card">
    <h2>Statistiche complete</h2>
    <table>
      <tr><th>Metrica</th><th>Dual Momentum</th><th>{{bm_name}}</th></tr>
      <tr><td>Rendimento totale</td><td class="pos">{{ss.rendimento_totale}}%</td><td>{{bs.rendimento_totale}}%</td></tr>
      <tr><td>CAGR</td><td class="pos">{{ss.cagr}}%</td><td>{{bs.cagr}}%</td></tr>
      <tr><td>Volatilità annua</td><td>{{ss.volatilita}}%</td><td>{{bs.volatilita}}%</td></tr>
      <tr><td>Sharpe ratio</td><td class="pos">{{ss.sharpe}}</td><td>{{bs.sharpe}}</td></tr>
      <tr><td>Sortino ratio</td><td class="pos">{{ss.sortino}}</td><td>{{bs.sortino}}</td></tr>
      <tr><td>Calmar ratio</td><td class="pos">{{ss.calmar}}</td><td>{{bs.calmar}}</td></tr>
      <tr><td>Max drawdown</td><td class="neg">{{ss.max_drawdown}}%</td><td class="neg">{{bs.max_drawdown}}%</td></tr>
      <tr><td>Miglior mese</td><td class="pos">{{ss.miglior_mese}}%</td><td>{{bs.miglior_mese}}%</td></tr>
      <tr><td>Peggior mese</td><td class="neg">{{ss.peggior_mese}}%</td><td class="neg">{{bs.peggior_mese}}%</td></tr>
      <tr><td>Win rate mensile</td><td>{{ss.win_rate}}%</td><td>{{bs.win_rate}}%</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Allocazione attuale
      {% if in_cash %}<span class="badge cash">CASH</span>
      {% else %}<span class="badge inv">INVESTITO</span>{% endif %}
    </h2>
    <p style="font-size:.88rem;margin:.4rem 0">
      {% for a in active %}<strong>{{a}}</strong>{% if not loop.last %}, {% endif %}{% endfor %}
    </p>
    <p style="font-size:.75rem;color:#aaa">Mesi in cash storicamente: {{cash_pct}}%</p>
  </div>

  <div class="card">
    <h2>Ultime rotazioni</h2>
    {% for t in trades %}
    <div class="trow">
      <span style="color:#aaa">{{t.data}}</span>
      <span class="{{'buy' if t.azione=='BUY' else 'sell'}}">{{t.azione}}</span>
      <span style="font-size:.72rem;color:#888">{{t.ticker}}</span>
      <span>{{t.nome}}</span>
    </div>
    {% endfor %}
  </div>

</div>
<footer>Dati via Yahoo Finance · Solo scopo didattico · Non è consulenza finanziaria</footer>
</body></html>
"""

# =============================================================================
#  ROUTES
# =============================================================================

@app.route("/")
def index():
    uptime_s = (datetime.now() - _boot_at).total_seconds()
    cold_start = uptime_s < 90    # mostra banner se il processo è appena partito
    return render_template_string(
        _INDEX,
        etfs=ETF_UNIVERSE,
        has_results=bool(_results),
        cold_start=cold_start,
    )


@app.route("/run", methods=["POST"])
def run():
    with _lock:
        if _status.get("running"):
            return jsonify({"error": "Job già in esecuzione"}), 409
    data = request.get_json() or {}
    cfg = {
        "start_date":       data.get("start_date", "2009-01-01"),
        "end_date":         datetime.today().strftime("%Y-%m-%d"),
        "lookback_months":  int(data.get("lookback_months", 6)),
        "top_n":            int(data.get("top_n", 2)),
        "initial_capital":  float(data.get("initial_capital", 10000)),
        "transaction_cost": float(data.get("transaction_cost", 0.001)),
        "cash_threshold":   float(data.get("cash_threshold", 0.0)),
    }
    threading.Thread(target=_run_job, args=(cfg,), daemon=True, name="backtest").start()
    return jsonify({"status": "started"}), 202   # risposta immediata, non blocca


@app.route("/status")
def status():
    with _lock:
        return jsonify(dict(_status))


@app.route("/results")
def results():
    if not _results:
        return redirect("/")
    r = _results
    return render_template_string(
        _RESULTS,
        cfg=type("C", (), r["config"])(),
        ss=type("S", (), r["stats_strat"])(),
        bs=type("S", (), r["stats_bench"])(),
        trades=r["trades"],
        active=r["active"],
        in_cash=r["in_cash"],
        cash_pct=r["cash_pct"],
        chart_equity=r["chart_equity"],
        chart_dd=r["chart_dd"],
        chart_heatmap=r["chart_heatmap"],
        run_at=r["run_at"],
        bm_name=BENCHMARK["name"],
    )


@app.route("/api/results")
def api_results():
    if not _results:
        return jsonify({"error": "Nessun risultato — esegui prima il backtest"}), 404
    r = _results
    return jsonify({
        "stats_strategia":   r["stats_strat"],
        "stats_benchmark":   r["stats_bench"],
        "allocazione_attuale": r["active"],
        "in_cash":           r["in_cash"],
        "cash_pct":          r["cash_pct"],
        "run_at":            r["run_at"],
        "config":            r["config"],
    })


@app.route("/health")
def health():
    """Health check per Render + trigger del self-ping."""
    return jsonify({"status": "ok", "uptime_s": int((datetime.now()-_boot_at).total_seconds())})


# =============================================================================
#  AVVIO LOCALE
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
