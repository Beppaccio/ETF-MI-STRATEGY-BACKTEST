"""
=============================================================================
  DUAL MOMENTUM ETF ROTATION — BORSA ITALIANA
  Backtest su dati reali via yfinance
  Autore: Claude (Anthropic) — Marzo 2026
=============================================================================

STRATEGIA:
  Universo : 8 ETF settoriali iShares STOXX Europe 600 quotati su Borsa Italiana
  Segnale  : Top-N ETF per momentum assoluto (lookback configurabile)
  Filtro   : Se momentum < 0 → posizione in ETF monetario (safe haven)
  Rebalance: Mensile (primo giorno di borsa aperta di ogni mese)
  Costi    : Spread bid-ask stimato + commissioni configurabili

OUTPUT:
  - Equity curve strategia vs benchmark (STOXX 600 / FTSEMIB)
  - Drawdown comparison
  - Statistiche complete (CAGR, Sharpe, Sortino, Calmar, Win Rate)
  - Heatmap rendimenti mensili
  - Log rotazioni
  - Tutti i grafici salvati in PNG + report CSV

DIPENDENZE:
  pip install yfinance pandas numpy matplotlib seaborn scipy tabulate
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

try:
    import yfinance as yf
except ImportError:
    print("ERRORE: yfinance non installato. Esegui: pip install yfinance")
    sys.exit(1)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# =============================================================================
#  CONFIGURAZIONE — modifica qui i parametri
# =============================================================================

CONFIG = {
    # Periodo di backtest
    "start_date": "2009-01-01",
    "end_date":   datetime.today().strftime("%Y-%m-%d"),

    # Parametri strategia
    "lookback_months": 6,        # Finestra momentum (mesi)
    "top_n": 2,                  # Numero ETF da tenere contemporaneamente
    "rebalance_day": "MS",       # Primo giorno del mese (Month Start)

    # Costi di transazione (per trade, andata+ritorno)
    "transaction_cost": 0.001,   # 0.10% per operazione (spread + commissioni)

    # Capitale iniziale (€)
    "initial_capital": 10_000,

    # Soglia cash: momentum minimo per non andare in monetario (default 0)
    "cash_threshold": 0.0,

    # Cartella output
    "output_dir": "backtest_output",
}

# =============================================================================
#  UNIVERSO ETF
#  Ticker Yahoo Finance (suffisso .MI = Borsa Italiana)
# =============================================================================

ETF_UNIVERSE = {
    "EXV1.MI": {"name": "Banks EU 600",      "sector": "Bancario",       "color": "#185FA5"},
    "EXH7.MI": {"name": "Oil & Gas EU 600",  "sector": "Energia",        "color": "#BA7517"},
    "EXV4.MI": {"name": "Healthcare EU 600", "sector": "Healthcare",     "color": "#1D9E75"},
    "EXH4.MI": {"name": "Industrial EU 600", "sector": "Industriali",    "color": "#534AB7"},
    "EXV5.MI": {"name": "Technology EU 600", "sector": "Tecnologia",     "color": "#D85A30"},
    "EXH9.MI": {"name": "Utilities EU 600",  "sector": "Utilities",      "color": "#639922"},
    "EXH5.MI": {"name": "Insurance EU 600",  "sector": "Assicurazioni",  "color": "#993556"},
    "EXH2.MI": {"name": "Basic Resources",   "sector": "Risorse",        "color": "#5F5E5A"},
}

# ETF monetario (safe haven quando momentum negativo)
CASH_ETF = {
    "ticker": "XEON.MI",   # Xtrackers EUR Overnight Rate Swap
    "name":   "EUR Overnight (monetario)",
    "color":  "#EF9F27",
}

# Benchmark
BENCHMARK = {
    "ticker": "EXSA.MI",   # iShares STOXX Europe 600 (o usa "^STOXX6E")
    "name":   "STOXX Europe 600",
    "color":  "#888780",
}


# =============================================================================
#  DOWNLOAD DATI
# =============================================================================

def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Scarica prezzi di chiusura aggiustati per tutti i ticker."""
    print(f"\n{'='*60}")
    print("  DOWNLOAD PREZZI DA YAHOO FINANCE")
    print(f"{'='*60}")
    print(f"  Periodo : {start} → {end}")
    print(f"  Ticker  : {len(tickers)} strumenti\n")

    all_prices = {}
    failed = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end,
                               auto_adjust=True, progress=False)
            if data.empty:
                print(f"  [!] {ticker:<15} — nessun dato trovato")
                failed.append(ticker)
                continue
            # Gestisce MultiIndex da yfinance >=0.2
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].squeeze()
            else:
                close = data["Close"]
            all_prices[ticker] = close
            n = len(close)
            print(f"  [OK] {ticker:<15} — {n} barre "
                  f"({close.index[0].date()} → {close.index[-1].date()})")
        except Exception as e:
            print(f"  [ERR] {ticker:<15} — {e}")
            failed.append(ticker)

    if not all_prices:
        print("\nERRORE: nessun prezzo scaricato. Controlla la connessione.")
        sys.exit(1)

    prices = pd.DataFrame(all_prices)
    prices = prices.ffill().dropna(how="all")

    if failed:
        print(f"\n  Ticker non disponibili: {failed}")
        print("  La strategia userà solo i ticker con dati validi.\n")

    return prices


# =============================================================================
#  SEGNALI DI MOMENTUM
# =============================================================================

def compute_momentum(prices: pd.DataFrame, lookback_months: int) -> pd.DataFrame:
    """
    Calcola il momentum come rendimento degli ultimi N mesi.
    Usa rendimento totale: (P_t / P_{t-N}) - 1
    """
    # Resample su fine mese
    monthly = prices.resample("ME").last()

    # Rendimento su lookback mesi
    momentum = monthly.pct_change(periods=lookback_months)
    return monthly, momentum


# =============================================================================
#  BACKTEST ENGINE
# =============================================================================

def run_backtest(
    prices: pd.DataFrame,
    monthly_prices: pd.DataFrame,
    monthly_momentum: pd.DataFrame,
    etf_tickers: list[str],
    cash_ticker: Optional[str],
    config: dict,
) -> dict:
    """
    Esegue il backtest mensile della strategia Dual Momentum.

    Returns
    -------
    dict con:
        portfolio_value : pd.Series   equity curve giornaliera
        allocations     : pd.DataFrame  allocazione mensile (0/1)
        trades_log      : list[dict]   log rotazioni
        monthly_returns : pd.Series   rendimenti mensili
    """
    lookback = config["lookback_months"]
    top_n    = config["top_n"]
    tc       = config["transaction_cost"]
    thresh   = config["cash_threshold"]
    capital  = float(config["initial_capital"])

    # Mesi di backtest (escludi i primi lookback per warmup)
    rebalance_dates = monthly_prices.index[lookback:]

    # Stato
    current_holdings: dict[str, float] = {}  # ticker → peso
    portfolio_daily = {}
    trades_log = []
    monthly_alloc = []

    # Indice giornaliero
    daily_prices = prices.copy()

    # Valore iniziale
    prev_date = daily_prices.index[0]
    portfolio_daily[prev_date] = capital

    for i, rdate in enumerate(rebalance_dates):
        # --- Calcola momentum al mese precedente ---
        mom = monthly_momentum.loc[rdate, etf_tickers].dropna()

        # Seleziona top-N con momentum positivo
        positive_mom = mom[mom > thresh].sort_values(ascending=False)
        selected = list(positive_mom.index[:top_n])

        # Regime: cash se nessun ETF con momentum positivo
        in_cash = len(selected) == 0
        if in_cash and cash_ticker and cash_ticker in daily_prices.columns:
            selected = [cash_ticker]

        # --- Gestione rotazione ---
        old_set = set(current_holdings.keys())
        new_set = set(selected)
        n_trades = len(old_set.symmetric_difference(new_set))
        cost = capital * tc * n_trades / max(len(selected), 1) if n_trades > 0 else 0
        capital -= cost

        # Log trade
        if old_set != new_set:
            sold = old_set - new_set
            bought = new_set - old_set
            for t in sold:
                name = ETF_UNIVERSE.get(t, {}).get("name", t)
                trades_log.append({"data": rdate.date(), "azione": "SELL",
                                   "ticker": t, "nome": name,
                                   "regime": "cash" if in_cash else "invested"})
            for t in bought:
                name = ETF_UNIVERSE.get(t, {}).get("name", t)
                if t == cash_ticker:
                    name = CASH_ETF["name"]
                trades_log.append({"data": rdate.date(), "azione": "BUY",
                                   "ticker": t, "nome": name,
                                   "regime": "cash" if in_cash else "invested"})

        # Pesi equi
        weight = 1.0 / len(selected) if selected else 0
        current_holdings = {t: weight for t in selected}

        # Log allocazione
        alloc_row = {"date": rdate, "in_cash": in_cash, "n_positions": len(selected)}
        for t in etf_tickers:
            alloc_row[t] = current_holdings.get(t, 0)
        monthly_alloc.append(alloc_row)

        # --- Evoluzione giornaliera fino al prossimo rebalance ---
        next_rdate = (rebalance_dates[i+1] if i+1 < len(rebalance_dates)
                      else daily_prices.index[-1])

        day_range = daily_prices.loc[rdate:next_rdate].index
        prev_val = capital

        for d in day_range[1:]:
            day_ret = 0.0
            for t, w in current_holdings.items():
                if t in daily_prices.columns:
                    prev_p = daily_prices.loc[prev_date, t] if prev_date in daily_prices.index else np.nan
                    curr_p = daily_prices.loc[d, t]
                    if pd.notna(prev_p) and pd.notna(curr_p) and prev_p > 0:
                        day_ret += w * (curr_p / prev_p - 1)
            prev_val *= (1 + day_ret)
            portfolio_daily[d] = prev_val
            prev_date = d

        capital = prev_val

    portfolio = pd.Series(portfolio_daily).sort_index()
    portfolio = portfolio[~portfolio.index.duplicated(keep="last")]

    # Rendimenti mensili
    monthly_portfolio = portfolio.resample("ME").last()
    monthly_returns = monthly_portfolio.pct_change().dropna()

    alloc_df = pd.DataFrame(monthly_alloc).set_index("date") if monthly_alloc else pd.DataFrame()

    return {
        "portfolio": portfolio,
        "monthly_returns": monthly_returns,
        "trades_log": trades_log,
        "allocations": alloc_df,
    }


# =============================================================================
#  STATISTICHE
# =============================================================================

def compute_stats(equity: pd.Series, rf_annual: float = 0.03) -> dict:
    """Calcola le principali statistiche di performance."""
    returns = equity.pct_change().dropna()
    monthly = equity.resample("ME").last().pct_change().dropna()

    # Anni
    years = (equity.index[-1] - equity.index[0]).days / 365.25

    # Rendimenti
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1

    # Rischio
    vol_daily = returns.std()
    vol_annual = vol_daily * np.sqrt(252)

    # Sharpe (rf giornaliero)
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    excess = returns - rf_daily
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    # Sortino (solo drawdown)
    neg = returns[returns < rf_daily]
    sortino = (excess.mean() / neg.std() * np.sqrt(252)) if len(neg) > 0 and neg.std() > 0 else 0

    # Drawdown
    rolling_max = equity.expanding().max()
    dd = (equity - rolling_max) / rolling_max
    max_dd = dd.min()

    # Durata max drawdown
    in_dd = (dd < 0)
    dd_start = None
    dd_lengths = []
    for date, val in in_dd.items():
        if val and dd_start is None:
            dd_start = date
        elif not val and dd_start is not None:
            dd_lengths.append((date - dd_start).days)
            dd_start = None
    max_dd_duration = max(dd_lengths) if dd_lengths else 0

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate mensile
    win_rate = (monthly > 0).mean()

    # Best / Worst month
    best_month = monthly.max()
    worst_month = monthly.min()

    # Mesi positivi/negativi
    pos_months = (monthly > 0).sum()
    neg_months = (monthly <= 0).sum()

    return {
        "Rendimento totale":      f"{total_ret:.1%}",
        "CAGR":                   f"{cagr:.2%}",
        "Volatilità annua":       f"{vol_annual:.2%}",
        "Sharpe ratio":           f"{sharpe:.2f}",
        "Sortino ratio":          f"{sortino:.2f}",
        "Calmar ratio":           f"{calmar:.2f}",
        "Max drawdown":           f"{max_dd:.1%}",
        "Max DD duration (gg)":   f"{max_dd_duration}",
        "Win rate (mesi)":        f"{win_rate:.1%}",
        "Miglior mese":           f"{best_month:.2%}",
        "Peggior mese":           f"{worst_month:.2%}",
        "Mesi positivi":          f"{pos_months}",
        "Mesi negativi":          f"{neg_months}",
        "Anni backtest":          f"{years:.1f}",
    }


# =============================================================================
#  PLOTTING
# =============================================================================

def setup_style():
    plt.rcParams.update({
        "figure.facecolor":   "#FAFAF8",
        "axes.facecolor":     "#FAFAF8",
        "axes.grid":          True,
        "axes.grid.alpha":    0.3,
        "grid.color":         "#CCCCCC",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   False,
        "axes.spines.bottom": False,
        "font.family":        "sans-serif",
        "font.size":          10,
        "axes.labelsize":     10,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
    })


def plot_equity_curve(ax, strat: pd.Series, bench: pd.Series, config: dict):
    """Equity curve normalizzata a 100."""
    s = strat / strat.iloc[0] * 100
    b = bench / bench.iloc[0] * 100

    ax.fill_between(s.index, s, 100, alpha=0.08, color="#185FA5")
    ax.plot(s.index, s, color="#185FA5", lw=2, label="Dual Momentum")
    ax.plot(b.index, b, color="#888780", lw=1.5, ls="--", label=BENCHMARK["name"])
    ax.axhline(100, color="#CCCCCC", lw=0.8)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.set_title("Equity curve (capitale iniziale = 100)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0)
    ax.set_ylabel("Valore (base 100)")


def plot_drawdown(ax, strat: pd.Series, bench: pd.Series):
    """Drawdown comparison."""
    def dd_series(s):
        pk = s.expanding().max()
        return (s - pk) / pk * 100

    ds = dd_series(strat)
    db = dd_series(bench)

    ax.fill_between(ds.index, ds, 0, alpha=0.25, color="#E24B4A")
    ax.plot(ds.index, ds, color="#E24B4A", lw=1.5, label="Dual Momentum DD")
    ax.fill_between(db.index, db, 0, alpha=0.10, color="#888780")
    ax.plot(db.index, db, color="#888780", lw=1, ls="--", label=f"{BENCHMARK['name']} DD")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_title("Drawdown")
    ax.legend(loc="lower left", fontsize=9, framealpha=0)
    ax.set_ylabel("Drawdown %")


def plot_monthly_heatmap(ax, monthly_rets: pd.Series):
    """Heatmap rendimenti mensili per anno × mese."""
    mr = monthly_rets.copy()
    mr.index = pd.to_datetime(mr.index)
    df = pd.DataFrame({
        "year": mr.index.year,
        "month": mr.index.month,
        "ret": mr.values * 100
    })
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Gen","Feb","Mar","Apr","Mag","Giu",
                     "Lug","Ago","Set","Ott","Nov","Dic"]

    cmap = LinearSegmentedColormap.from_list(
        "rdgn", ["#C0392B", "#FAFAF8", "#1D9E75"], N=256
    )
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, center=0,
        annot=True, fmt=".1f", annot_kws={"size": 7},
        linewidths=0.4, linecolor="#E0E0E0",
        cbar_kws={"shrink": 0.6, "label": "Rendimento %"},
        vmin=-10, vmax=10,
    )
    ax.set_title("Rendimenti mensili (%)")
    ax.set_ylabel("Anno")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)


def plot_sector_presence(ax, alloc_df: pd.DataFrame, etf_tickers: list[str]):
    """Istogramma presenze storiche per settore."""
    presence = {}
    invested_months = alloc_df[~alloc_df["in_cash"]]
    if len(invested_months) == 0:
        ax.text(0.5, 0.5, "Nessun dato", ha="center", va="center")
        return

    for t in etf_tickers:
        if t in alloc_df.columns:
            name = ETF_UNIVERSE.get(t, {}).get("name", t)
            color = ETF_UNIVERSE.get(t, {}).get("color", "#888780")
            pct = (invested_months[t] > 0).mean() * 100
            presence[name] = (pct, color)

    names = list(presence.keys())
    pcts  = [presence[n][0] for n in names]
    cols  = [presence[n][1] for n in names]

    bars = ax.barh(names, pcts, color=cols, alpha=0.85, height=0.6)
    ax.set_xlim(0, 105)
    ax.set_xlabel("% mesi in portafoglio (quando investito)")
    ax.set_title("Presenza storica per settore")
    for bar, pct in zip(bars, pcts):
        ax.text(pct + 1, bar.get_y() + bar.get_height()/2,
                f"{pct:.0f}%", va="center", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))


def plot_rolling_stats(ax, strat: pd.Series):
    """Rolling Sharpe a 12 mesi."""
    daily_ret = strat.pct_change().dropna()
    rf = (1.03 ** (1/252)) - 1
    rolling_sharpe = (
        (daily_ret - rf).rolling(252).mean() /
        (daily_ret - rf).rolling(252).std() * np.sqrt(252)
    )
    ax.plot(rolling_sharpe.index, rolling_sharpe,
            color="#534AB7", lw=1.5, label="Rolling Sharpe (12m)")
    ax.axhline(0, color="#CCCCCC", lw=0.8)
    ax.axhline(1, color="#1D9E75", lw=0.8, ls="--", alpha=0.5, label="Sharpe = 1")
    ax.set_title("Rolling Sharpe ratio (12 mesi)")
    ax.set_ylabel("Sharpe")
    ax.legend(fontsize=8, framealpha=0)
    ax.set_ylim(-3, 5)


def generate_report(
    strat_equity: pd.Series,
    bench_equity: pd.Series,
    monthly_returns: pd.Series,
    allocations: pd.DataFrame,
    trades_log: list,
    etf_tickers: list,
    config: dict,
    output_dir: str,
):
    setup_style()

    fig = plt.figure(figsize=(18, 22), constrained_layout=True)
    fig.suptitle(
        f"Dual Momentum ETF Rotation — Borsa Italiana\n"
        f"Lookback: {config['lookback_months']}m · Top-{config['top_n']} ETF · "
        f"Costi: {config['transaction_cost']*100:.2f}% · "
        f"Periodo: {config['start_date']} → {config['end_date']}",
        fontsize=13, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    ax_eq   = fig.add_subplot(gs[0, :])
    ax_dd   = fig.add_subplot(gs[1, :])
    ax_heat = fig.add_subplot(gs[2, :])
    ax_sec  = fig.add_subplot(gs[3, 0])
    ax_roll = fig.add_subplot(gs[3, 1])

    plot_equity_curve(ax_eq, strat_equity, bench_equity, config)
    plot_drawdown(ax_dd, strat_equity, bench_equity)
    plot_monthly_heatmap(ax_heat, monthly_returns)
    plot_sector_presence(ax_sec, allocations, etf_tickers)
    plot_rolling_stats(ax_roll, strat_equity)

    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, "dual_momentum_report.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] Grafico salvato: {chart_path}")
    return chart_path


# =============================================================================
#  EXPORT CSV
# =============================================================================

def export_csv(strat: pd.Series, bench: pd.Series,
               alloc: pd.DataFrame, trades: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Equity
    eq = pd.DataFrame({"strategia": strat, "benchmark": bench})
    eq.to_csv(os.path.join(output_dir, "equity_curve.csv"))

    # Allocazioni mensili
    if not alloc.empty:
        alloc.to_csv(os.path.join(output_dir, "allocazioni_mensili.csv"))

    # Trade log
    if trades:
        pd.DataFrame(trades).to_csv(
            os.path.join(output_dir, "trade_log.csv"), index=False)

    print(f"  [OK] CSV esportati in: {output_dir}/")


# =============================================================================
#  STAMPA STATISTICHE
# =============================================================================

def print_stats(strat_eq: pd.Series, bench_eq: pd.Series,
                trades: list, alloc: pd.DataFrame):
    strat_stats = compute_stats(strat_eq)
    bench_stats = compute_stats(bench_eq)

    print(f"\n{'='*64}")
    print("  STATISTICHE DI PERFORMANCE")
    print(f"{'='*64}")

    rows = [(k, strat_stats[k], bench_stats[k]) for k in strat_stats]

    if HAS_TABULATE:
        print(tabulate(rows,
                       headers=["Metrica", "Dual Momentum", BENCHMARK["name"]],
                       tablefmt="rounded_outline"))
    else:
        print(f"{'Metrica':<28} {'Dual Momentum':>16} {BENCHMARK['name']:>20}")
        print("-" * 68)
        for k, s, b in rows:
            print(f"  {k:<26} {s:>16} {b:>20}")

    # Regime
    if not alloc.empty and "in_cash" in alloc.columns:
        cash_pct = alloc["in_cash"].mean() * 100
        print(f"\n  Mesi in cash (monetario) : {cash_pct:.1f}%")
        print(f"  Mesi investito           : {100-cash_pct:.1f}%")

    # Ultime 10 rotazioni
    print(f"\n{'='*64}")
    print("  ULTIME ROTAZIONI")
    print(f"{'='*64}")
    if trades:
        last = trades[-10:]
        if HAS_TABULATE:
            print(tabulate(
                [(t["data"], t["azione"], t["ticker"], t["nome"]) for t in last],
                headers=["Data", "Azione", "Ticker", "Nome ETF"],
                tablefmt="rounded_outline"
            ))
        else:
            for t in last:
                print(f"  {t['data']}  {t['azione']:<5}  {t['ticker']:<12}  {t['nome']}")
    else:
        print("  Nessuna rotazione registrata.")


# =============================================================================
#  SENSITIVITY ANALYSIS
# =============================================================================

def sensitivity_analysis(prices: pd.DataFrame, etf_tickers: list,
                          cash_ticker: Optional[str], config: dict) -> pd.DataFrame:
    """
    Testa combinazioni di lookback × top_n e restituisce tabella Sharpe/CAGR.
    """
    print(f"\n{'='*60}")
    print("  SENSITIVITY ANALYSIS (lookback × top_N)")
    print(f"{'='*60}")

    lookbacks = [3, 6, 9, 12]
    topns     = [1, 2, 3, 4]
    results   = []

    for lb in lookbacks:
        for tn in topns:
            cfg = {**config, "lookback_months": lb, "top_n": tn}
            monthly_p, monthly_m = compute_momentum(prices[etf_tickers], lb)
            all_tickers = etf_tickers + ([cash_ticker] if cash_ticker else [])
            all_prices  = prices[[t for t in all_tickers if t in prices.columns]]
            try:
                res = run_backtest(all_prices, monthly_p, monthly_m,
                                   etf_tickers, cash_ticker, cfg)
                stats = compute_stats(res["portfolio"])
                results.append({
                    "Lookback (mesi)": lb,
                    "Top N":           tn,
                    "CAGR":            stats["CAGR"],
                    "Sharpe":          stats["Sharpe ratio"],
                    "Max DD":          stats["Max drawdown"],
                    "Win Rate":        stats["Win rate (mesi)"],
                })
            except Exception as e:
                results.append({
                    "Lookback (mesi)": lb, "Top N": tn,
                    "CAGR": "ERR", "Sharpe": "ERR", "Max DD": "ERR", "Win Rate": "ERR"
                })

    df = pd.DataFrame(results)
    if HAS_TABULATE:
        print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))
    else:
        print(df.to_string(index=False))
    return df


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  DUAL MOMENTUM ETF ROTATION — BACKTEST ENGINE")
    print("  Borsa Italiana · Dati reali via yfinance")
    print("="*60)

    # Tutti i ticker da scaricare
    etf_tickers  = list(ETF_UNIVERSE.keys())
    cash_ticker  = CASH_ETF["ticker"]
    bench_ticker = BENCHMARK["ticker"]

    all_tickers = etf_tickers + [cash_ticker, bench_ticker]

    # Download
    prices = download_prices(all_tickers, CONFIG["start_date"], CONFIG["end_date"])

    # Separa benchmark e cash
    bench_prices = prices[bench_ticker] if bench_ticker in prices.columns else None

    # ETF disponibili (alcuni potrebbero mancare su Yahoo)
    available_etf = [t for t in etf_tickers if t in prices.columns]
    if not available_etf:
        print("ERRORE: nessun ETF dell'universo disponibile.")
        sys.exit(1)

    print(f"\n  ETF disponibili per backtest: {len(available_etf)}/{len(etf_tickers)}")

    # Prezzi mensili e momentum
    etf_prices_daily = prices[[t for t in available_etf +
                                ([cash_ticker] if cash_ticker in prices.columns else [])
                                if t in prices.columns]]

    monthly_prices, monthly_momentum = compute_momentum(
        etf_prices_daily[available_etf], CONFIG["lookback_months"]
    )

    # --- BACKTEST STRATEGIA ---
    print(f"\n  Eseguo backtest principale...")
    print(f"  Lookback: {CONFIG['lookback_months']}m · Top-N: {CONFIG['top_n']} · "
          f"Costi: {CONFIG['transaction_cost']*100:.2f}%")

    result = run_backtest(
        etf_prices_daily, monthly_prices, monthly_momentum,
        available_etf,
        cash_ticker if cash_ticker in prices.columns else None,
        CONFIG,
    )

    strat_equity   = result["portfolio"]
    monthly_returns = result["monthly_returns"]
    trades_log     = result["trades_log"]
    allocations    = result["allocations"]

    # Allinea benchmark alla strategia
    if bench_prices is not None:
        bench_aligned = bench_prices.reindex(strat_equity.index).ffill()
        bench_equity  = bench_aligned / bench_aligned.iloc[0] * CONFIG["initial_capital"]
    else:
        print("  [!] Benchmark non disponibile, uso proxy uniforme.")
        bench_equity = pd.Series(
            [CONFIG["initial_capital"]] * len(strat_equity), index=strat_equity.index
        )

    # --- STATISTICHE ---
    print_stats(strat_equity, bench_equity, trades_log, allocations)

    # --- GRAFICI ---
    print(f"\n  Genero report grafico...")
    generate_report(
        strat_equity, bench_equity, monthly_returns,
        allocations, trades_log, available_etf,
        CONFIG, CONFIG["output_dir"]
    )

    # --- EXPORT CSV ---
    export_csv(strat_equity, bench_equity, allocations,
               trades_log, CONFIG["output_dir"])

    # --- SENSITIVITY ---
    print(f"\n  Eseguo sensitivity analysis...")
    sens_df = sensitivity_analysis(
        etf_prices_daily, available_etf,
        cash_ticker if cash_ticker in prices.columns else None,
        CONFIG
    )
    sens_df.to_csv(
        os.path.join(CONFIG["output_dir"], "sensitivity.csv"), index=False
    )

    print(f"\n{'='*60}")
    print(f"  BACKTEST COMPLETATO")
    print(f"  Output in: ./{CONFIG['output_dir']}/")
    print(f"    - dual_momentum_report.png")
    print(f"    - equity_curve.csv")
    print(f"    - allocazioni_mensili.csv")
    print(f"    - trade_log.csv")
    print(f"    - sensitivity.csv")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
