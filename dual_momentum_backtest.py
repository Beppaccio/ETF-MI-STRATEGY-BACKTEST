"""
=============================================================================
  DUAL MOMENTUM ETF ROTATION — versione Render.com
  Il backtest gira come Cron Job mensile su Render.
  Output: stampa su stdout (visibile nei log Render) +
          salva PNG e CSV in /tmp/backtest_output/
          (opzionale: invia report via email o Telegram)
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, json
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # OBBLIGATORIO su server headless (niente display)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

try:
    import yfinance as yf
except ImportError:
    print("ERRORE: yfinance non installato.")
    sys.exit(1)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# =============================================================================
#  CONFIGURAZIONE — legge da variabili d'ambiente se presenti (Render dashboard)
# =============================================================================

def env(key, default):
    return os.environ.get(key, default)

CONFIG = {
    "start_date":        env("START_DATE", "2009-01-01"),
    "end_date":          datetime.today().strftime("%Y-%m-%d"),
    "lookback_months":   int(env("LOOKBACK_MONTHS", "6")),
    "top_n":             int(env("TOP_N", "2")),
    "transaction_cost":  float(env("TRANSACTION_COST", "0.001")),
    "initial_capital":   float(env("INITIAL_CAPITAL", "10000")),
    "cash_threshold":    float(env("CASH_THRESHOLD", "0.0")),
    "output_dir":        env("OUTPUT_DIR", "/tmp/backtest_output"),
    # Notifiche (opzionale)
    "telegram_token":    env("TELEGRAM_TOKEN", ""),
    "telegram_chat_id":  env("TELEGRAM_CHAT_ID", ""),
    "smtp_host":         env("SMTP_HOST", ""),
    "smtp_user":         env("SMTP_USER", ""),
    "smtp_pass":         env("SMTP_PASS", ""),
    "smtp_to":           env("SMTP_TO", ""),
}

ETF_UNIVERSE = {
    "EXV1.MI": {"name": "Banks EU 600",      "sector": "Bancario",    "color": "#185FA5"},
    "EXH7.MI": {"name": "Oil & Gas EU 600",  "sector": "Energia",     "color": "#BA7517"},
    "EXV4.MI": {"name": "Healthcare EU 600", "sector": "Healthcare",  "color": "#1D9E75"},
    "EXH4.MI": {"name": "Industrial EU 600", "sector": "Industriali", "color": "#534AB7"},
    "EXV5.MI": {"name": "Technology EU 600", "sector": "Tecnologia",  "color": "#D85A30"},
    "EXH9.MI": {"name": "Utilities EU 600",  "sector": "Utilities",   "color": "#639922"},
    "EXH5.MI": {"name": "Insurance EU 600",  "sector": "Assicur.",    "color": "#993556"},
    "EXH2.MI": {"name": "Basic Resources",   "sector": "Risorse",     "color": "#5F5E5A"},
}

CASH_ETF  = {"ticker": "XEON.MI", "name": "EUR Overnight"}
BENCHMARK = {"ticker": "EXSA.MI", "name": "STOXX Europe 600"}


# =============================================================================
#  NOTIFICHE OPZIONALI
# =============================================================================

def send_telegram(message: str, image_path: Optional[str] = None):
    """Invia messaggio (e opzionalmente immagine) via bot Telegram."""
    token = CONFIG["telegram_token"]
    chat  = CONFIG["telegram_chat_id"]
    if not token or not chat:
        return
    try:
        import requests as req
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                req.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={"chat_id": chat, "caption": message[:1024]},
                    files={"photo": f}, timeout=30
                )
        else:
            req.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": message, "parse_mode": "HTML"},
                timeout=30
            )
        print("  [OK] Notifica Telegram inviata.")
    except Exception as e:
        print(f"  [!] Telegram fallito: {e}")


def send_email(subject: str, body: str, attachment: Optional[str] = None):
    """Invia report via SMTP (es. Gmail con App Password)."""
    host = CONFIG["smtp_host"]
    user = CONFIG["smtp_user"]
    pwd  = CONFIG["smtp_pass"]
    to   = CONFIG["smtp_to"]
    if not all([host, user, pwd, to]):
        return
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders

        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = user, to, subject
        msg.attach(MIMEText(body, "plain"))

        if attachment and os.path.exists(attachment):
            with open(attachment, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition",
                            f'attachment; filename="{os.path.basename(attachment)}"')
            msg.attach(part)

        with smtplib.SMTP_SSL(host, 465) as s:
            s.login(user, pwd)
            s.sendmail(user, to, msg.as_string())
        print("  [OK] Email inviata.")
    except Exception as e:
        print(f"  [!] Email fallita: {e}")


# =============================================================================
#  DOWNLOAD + BACKTEST (stessa logica del file principale, adattata per server)
# =============================================================================

def download_prices(tickers, start, end):
    print(f"\n[{datetime.now():%H:%M:%S}] Download prezzi...")
    all_prices = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end,
                               auto_adjust=True, progress=False)
            if data.empty:
                print(f"  [!] {ticker} — nessun dato")
                continue
            close = data["Close"].squeeze() if isinstance(data.columns, pd.MultiIndex) else data["Close"]
            all_prices[ticker] = close
            print(f"  [OK] {ticker} — {len(close)} barre")
        except Exception as e:
            print(f"  [ERR] {ticker} — {e}")
    prices = pd.DataFrame(all_prices).ffill().dropna(how="all")
    return prices


def compute_momentum(prices, lookback_months):
    monthly = prices.resample("ME").last()
    momentum = monthly.pct_change(periods=lookback_months)
    return monthly, momentum


def run_backtest(prices, monthly_prices, monthly_momentum,
                 etf_tickers, cash_ticker, config):
    lookback = config["lookback_months"]
    top_n    = config["top_n"]
    tc       = config["transaction_cost"]
    thresh   = config["cash_threshold"]
    capital  = float(config["initial_capital"])

    rebalance_dates = monthly_prices.index[lookback:]
    current_holdings = {}
    portfolio_daily  = {}
    trades_log       = []
    monthly_alloc    = []

    prev_date = prices.index[0]
    portfolio_daily[prev_date] = capital

    for i, rdate in enumerate(rebalance_dates):
        mom = monthly_momentum.loc[rdate, etf_tickers].dropna()
        positive_mom = mom[mom > thresh].sort_values(ascending=False)
        selected = list(positive_mom.index[:top_n])
        in_cash  = len(selected) == 0

        if in_cash and cash_ticker and cash_ticker in prices.columns:
            selected = [cash_ticker]

        old_set = set(current_holdings.keys())
        new_set = set(selected)
        n_trades = len(old_set.symmetric_difference(new_set))
        cost = capital * tc * n_trades / max(len(selected), 1) if n_trades > 0 else 0
        capital -= cost

        if old_set != new_set:
            for t in old_set - new_set:
                trades_log.append({"data": rdate.date(), "azione": "SELL",
                                   "ticker": t,
                                   "nome": ETF_UNIVERSE.get(t, {}).get("name", t)})
            for t in new_set - old_set:
                name = CASH_ETF["name"] if t == cash_ticker else ETF_UNIVERSE.get(t, {}).get("name", t)
                trades_log.append({"data": rdate.date(), "azione": "BUY",
                                   "ticker": t, "nome": name})

        weight = 1.0 / len(selected) if selected else 0
        current_holdings = {t: weight for t in selected}

        alloc_row = {"date": rdate, "in_cash": in_cash}
        for t in etf_tickers:
            alloc_row[t] = current_holdings.get(t, 0)
        monthly_alloc.append(alloc_row)

        next_rdate = (rebalance_dates[i+1] if i+1 < len(rebalance_dates)
                      else prices.index[-1])
        day_range = prices.loc[rdate:next_rdate].index
        prev_val  = capital

        for d in day_range[1:]:
            day_ret = 0.0
            for t, w in current_holdings.items():
                if t in prices.columns:
                    prev_p = prices.loc[prev_date, t] if prev_date in prices.index else np.nan
                    curr_p = prices.loc[d, t]
                    if pd.notna(prev_p) and pd.notna(curr_p) and prev_p > 0:
                        day_ret += w * (curr_p / prev_p - 1)
            prev_val *= (1 + day_ret)
            portfolio_daily[d] = prev_val
            prev_date = d

        capital = prev_val

    portfolio = pd.Series(portfolio_daily).sort_index()
    portfolio = portfolio[~portfolio.index.duplicated(keep="last")]
    monthly_portfolio = portfolio.resample("ME").last()
    monthly_returns   = monthly_portfolio.pct_change().dropna()
    alloc_df = pd.DataFrame(monthly_alloc).set_index("date") if monthly_alloc else pd.DataFrame()

    return {"portfolio": portfolio, "monthly_returns": monthly_returns,
            "trades_log": trades_log, "allocations": alloc_df}


def compute_stats(equity, rf_annual=0.03):
    returns = equity.pct_change().dropna()
    monthly = equity.resample("ME").last().pct_change().dropna()
    years   = (equity.index[-1] - equity.index[0]).days / 365.25

    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    cagr      = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1
    vol_ann   = returns.std() * np.sqrt(252)
    rf_daily  = (1 + rf_annual) ** (1/252) - 1
    excess    = returns - rf_daily
    sharpe    = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    neg       = returns[returns < rf_daily]
    sortino   = (excess.mean() / neg.std() * np.sqrt(252)) if len(neg) > 0 and neg.std() > 0 else 0
    pk        = equity.expanding().max()
    dd        = (equity - pk) / pk
    max_dd    = dd.min()
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0
    win_rate  = (monthly > 0).mean()

    return {
        "CAGR":             cagr,
        "Volatilità ann.":  vol_ann,
        "Sharpe":           sharpe,
        "Sortino":          sortino,
        "Calmar":           calmar,
        "Max drawdown":     max_dd,
        "Win rate":         win_rate,
        "Rendimento tot.":  total_ret,
    }


# =============================================================================
#  GRAFICI (headless matplotlib)
# =============================================================================

def generate_charts(strat, bench, monthly_rets, alloc, etf_tickers, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "#FAFAF8", "axes.facecolor": "#FAFAF8",
        "axes.grid": True, "axes.grid.alpha": 0.3, "grid.color": "#CCCCCC",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "font.family": "sans-serif", "font.size": 10,
    })

    fig = plt.figure(figsize=(16, 18), constrained_layout=True)
    fig.suptitle(
        f"Dual Momentum ETF Rotation — Borsa Italiana\n"
        f"Lookback: {CONFIG['lookback_months']}m · Top-{CONFIG['top_n']} ETF · "
        f"Run: {datetime.today():%Y-%m-%d}",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    s = strat / strat.iloc[0] * 100
    b = bench / bench.iloc[0] * 100
    ax1.fill_between(s.index, s, 100, alpha=0.08, color="#185FA5")
    ax1.plot(s.index, s, color="#185FA5", lw=2, label="Dual Momentum")
    ax1.plot(b.index, b, color="#888780", lw=1.5, ls="--", label=BENCHMARK["name"])
    ax1.axhline(100, color="#CCCCCC", lw=0.8)
    ax1.set_title("Equity curve (base 100)")
    ax1.legend(fontsize=9, framealpha=0)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    def dd_s(x): pk=x.expanding().max(); return (x-pk)/pk*100
    ax2.fill_between(strat.index, dd_s(strat), 0, alpha=0.25, color="#E24B4A")
    ax2.plot(strat.index, dd_s(strat), color="#E24B4A", lw=1.5, label="Strategia")
    ax2.fill_between(bench.index, dd_s(bench), 0, alpha=0.10, color="#888780")
    ax2.plot(bench.index, dd_s(bench), color="#888780", lw=1, ls="--", label="Benchmark")
    ax2.set_title("Drawdown %")
    ax2.legend(fontsize=9, framealpha=0)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # 3. Heatmap mensile
    ax3 = fig.add_subplot(gs[2, :])
    mr = monthly_rets.copy()
    mr.index = pd.to_datetime(mr.index)
    df = pd.DataFrame({"year": mr.index.year, "month": mr.index.month, "ret": mr.values*100})
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"]
    cmap = LinearSegmentedColormap.from_list("rg", ["#C0392B","#FAFAF8","#1D9E75"], N=256)
    sns.heatmap(pivot, ax=ax3, cmap=cmap, center=0, annot=True, fmt=".1f",
                annot_kws={"size":7}, linewidths=0.4, linecolor="#E0E0E0",
                cbar_kws={"shrink":0.5, "label":"Rendimento %"}, vmin=-10, vmax=10)
    ax3.set_title("Rendimenti mensili %")
    ax3.set_ylabel("Anno")
    ax3.tick_params(axis="x", rotation=0)

    path = os.path.join(output_dir, "dual_momentum_report.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Grafico salvato: {path}")
    return path


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print(f"\n{'='*60}")
    print(f"  DUAL MOMENTUM BACKTEST — Render.com Cron Job")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S} UTC")
    print(f"{'='*60}")

    etf_tickers  = list(ETF_UNIVERSE.keys())
    cash_ticker  = CASH_ETF["ticker"]
    bench_ticker = BENCHMARK["ticker"]
    all_tickers  = etf_tickers + [cash_ticker, bench_ticker]

    prices = download_prices(all_tickers, CONFIG["start_date"], CONFIG["end_date"])

    bench_prices   = prices.get(bench_ticker)
    available_etf  = [t for t in etf_tickers if t in prices.columns]

    if not available_etf:
        print("ERRORE CRITICO: nessun ETF disponibile.")
        sys.exit(1)

    cash_avail  = cash_ticker if cash_ticker in prices.columns else None
    etf_prices  = prices[[t for t in available_etf + ([cash_ticker] if cash_avail else []) if t in prices.columns]]
    monthly_p, monthly_m = compute_momentum(etf_prices[available_etf], CONFIG["lookback_months"])

    print(f"\n[{datetime.now():%H:%M:%S}] Esecuzione backtest...")
    result = run_backtest(etf_prices, monthly_p, monthly_m,
                          available_etf, cash_avail, CONFIG)

    strat          = result["portfolio"]
    monthly_rets   = result["monthly_returns"]
    trades_log     = result["trades_log"]
    allocations    = result["allocations"]

    if bench_prices is not None:
        bench_aligned = bench_prices.reindex(strat.index).ffill()
        bench_equity  = bench_aligned / bench_aligned.iloc[0] * CONFIG["initial_capital"]
    else:
        bench_equity = pd.Series([CONFIG["initial_capital"]]*len(strat), index=strat.index)

    # Stats
    ss = compute_stats(strat)
    bs = compute_stats(bench_equity)

    print(f"\n{'='*60}")
    print("  RISULTATI")
    print(f"{'='*60}")
    rows = [(k, f"{ss[k]:.2%}" if isinstance(ss[k], float) else ss[k],
                f"{bs[k]:.2%}" if isinstance(bs[k], float) else bs[k])
            for k in ss]
    if HAS_TABULATE:
        print(tabulate(rows, headers=["Metrica","Dual Momentum","Benchmark"],
                       tablefmt="rounded_outline"))
    else:
        for k, sv, bv in rows:
            print(f"  {k:<22} {sv:>14}  {bv:>14}")

    # Allocazione corrente
    if not allocations.empty:
        last = allocations.iloc[-1]
        in_cash = bool(last.get("in_cash", False))
        active  = [t for t in available_etf if last.get(t, 0) > 0]
        print(f"\n  Allocazione attuale: {'CASH' if in_cash else ', '.join(active)}")

    # Export CSV
    out = CONFIG["output_dir"]
    os.makedirs(out, exist_ok=True)
    pd.DataFrame({"strategia": strat, "benchmark": bench_equity}).to_csv(
        os.path.join(out, "equity_curve.csv"))
    if not allocations.empty:
        allocations.to_csv(os.path.join(out, "allocazioni.csv"))
    if trades_log:
        pd.DataFrame(trades_log).to_csv(os.path.join(out, "trade_log.csv"), index=False)
    print(f"  [OK] CSV salvati in {out}/")

    # Grafici
    print(f"\n[{datetime.now():%H:%M:%S}] Generazione grafici...")
    chart_path = generate_charts(strat, bench_equity, monthly_rets,
                                 allocations, available_etf, out)

    # Notifiche
    cagr_strat = ss["CAGR"]
    dd_strat   = ss["Max drawdown"]
    sharpe     = ss["Sharpe"]
    active_str = "CASH" if (not allocations.empty and bool(allocations.iloc[-1].get("in_cash", False))) else ", ".join(active if 'active' in dir() else [])

    telegram_msg = (
        f"<b>Dual Momentum — Report {datetime.today():%B %Y}</b>\n\n"
        f"CAGR: {cagr_strat:.2%} | Sharpe: {sharpe:.2f} | Max DD: {dd_strat:.2%}\n"
        f"Allocazione attuale: <b>{active_str}</b>\n"
        f"Lookback: {CONFIG['lookback_months']}m · Top-{CONFIG['top_n']}"
    )
    email_body = (
        f"Dual Momentum ETF Rotation — Report {datetime.today():%B %Y}\n\n"
        f"CAGR: {cagr_strat:.2%}\n"
        f"Sharpe: {sharpe:.2f}\n"
        f"Max Drawdown: {dd_strat:.2%}\n"
        f"Allocazione attuale: {active_str}\n\n"
        f"Vedi grafico allegato."
    )

    send_telegram(telegram_msg, chart_path)
    send_email(
        subject=f"Dual Momentum Report {datetime.today():%Y-%m}",
        body=email_body,
        attachment=chart_path,
    )

    print(f"\n{'='*60}")
    print(f"  COMPLETATO — {datetime.now():%H:%M:%S}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
