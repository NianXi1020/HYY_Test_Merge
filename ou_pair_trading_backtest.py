from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class Config:
    etf_path: Path = Path("merged_outputs/511090_20250701_20251231_merged.csv")
    futures_path: Path = Path("国债期货合约数据/快照/30年TL合约/TL_main_snapshot.csv")
    start: str = "2025-07-01"
    end: str = "2025-12-31"
    resample_rule: str = "1min"
    initial_cash: float = 10_000_000
    update_every_bars: int = 180
    window_size: int = 180
    entry_z: float = 2.0
    exit_z: float = 0.5
    max_futures_contracts: int = 8
    commission: float = 0.0001


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def estimate_ou_params(spread: pd.Series, dt: float = 1.0) -> Dict[str, float]:
    s = spread.dropna().values
    if len(s) < 3:
        return {"theta": np.nan, "mu": np.nan, "sigma": np.nan, "alpha": np.nan, "beta": np.nan}

    x = s[:-1]
    y = s[1:]
    beta, alpha = np.polyfit(x, y, 1)
    beta = min(max(beta, 1e-6), 0.999999)
    theta = -np.log(beta) / dt
    mu = alpha / (1 - beta)
    resid = y - (alpha + beta * x)
    sigma = np.std(resid, ddof=1)
    return {"theta": theta, "mu": mu, "sigma": sigma, "alpha": alpha, "beta": beta}


def load_and_align_data(cfg: Config) -> pd.DataFrame:
    etf = pd.read_csv(cfg.etf_path)
    fut = pd.read_csv(cfg.futures_path, usecols=["trade_time", "last", "code"], low_memory=False)

    etf_time_col = "trade_time" if "trade_time" in etf.columns else "datetime"
    etf_price_col = "close" if "close" in etf.columns else "last"

    etf = etf[[etf_time_col, etf_price_col, *(["volume"] if "volume" in etf.columns else [])]].copy()
    etf = etf.rename(columns={etf_time_col: "datetime", etf_price_col: "etf_price"})
    fut = fut.rename(columns={"trade_time": "datetime", "last": "futures_price"})

    for df in (etf, fut):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df.dropna(subset=["datetime"], inplace=True)
        df.sort_values("datetime", inplace=True)
        df.drop_duplicates(subset=["datetime"], keep="last", inplace=True)

    start_dt = pd.Timestamp(cfg.start)
    end_dt = pd.Timestamp(cfg.end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    etf = etf[(etf["datetime"] >= start_dt) & (etf["datetime"] <= end_dt)]
    fut = fut[(fut["datetime"] >= start_dt) & (fut["datetime"] <= end_dt)]

    etf = etf.set_index("datetime").resample(cfg.resample_rule).last().ffill().reset_index()
    fut = fut.set_index("datetime").resample(cfg.resample_rule).last().ffill().reset_index()

    df = pd.merge(etf, fut[["datetime", "futures_price"]], on="datetime", how="inner")
    if "ETF_Futures_Spread" in etf.columns:
        df["spread"] = etf["ETF_Futures_Spread"]
    else:
        df["spread"] = df["etf_price"] - df["futures_price"]

    return df.dropna(subset=["spread", "etf_price", "futures_price"]).reset_index(drop=True)


class PairData(bt.feeds.PandasData):
    lines = ("spread", "zscore", "mu", "sigma")
    params = (("datetime", None), ("open", -1), ("high", -1), ("low", -1), ("close", "etf_price"),
              ("volume", -1), ("openinterest", -1), ("spread", "spread"), ("zscore", "zscore"),
              ("mu", "mu"), ("sigma", "sigma"))


class OUPairTradingStrategy(bt.Strategy):
    params = dict(entry_z=2.0, exit_z=0.5, max_fut=8)

    def __init__(self):
        self.logs: List[Dict] = []
        self.current_dir = 0

    def next(self):
        z = float(self.datas[0].zscore[0])
        if np.isnan(z):
            return

        dt = self.datas[0].datetime.datetime(0)
        etf_price = float(self.datas[0].close[0])
        fut_price = float(self.datas[1].close[0])

        target = min(int(abs(z) / 4 * self.p.max_fut), self.p.max_fut)
        desired_dir = -1 if z > self.p.entry_z else (1 if z < -self.p.entry_z else 0)

        if self.current_dir != 0 and abs(z) < self.p.exit_z:
            self.close(self.datas[0])
            self.close(self.datas[1])
            self.current_dir = 0
            signal = "close"
        elif desired_dir != 0 and desired_dir != self.current_dir:
            self.close(self.datas[0]); self.close(self.datas[1])
            etf_size = target * 100
            fut_size = target
            if desired_dir == -1:
                self.sell(self.datas[0], size=etf_size)
                self.buy(self.datas[1], size=fut_size)
                signal = "short_spread"
            else:
                self.buy(self.datas[0], size=etf_size)
                self.sell(self.datas[1], size=fut_size)
                signal = "long_spread"
            self.current_dir = desired_dir
        else:
            signal = "hold"

        self.logs.append({
            "timestamp": dt,
            "zscore": z,
            "spread": float(self.datas[0].spread[0]),
            "signal": signal,
            "etf_price": etf_price,
            "futures_price": fut_price,
            "value": self.broker.getvalue(),
            "cash": self.broker.getcash(),
        })


def prepare_model_features(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_size = int(n * 0.5)

    rolling_rows = []
    mu_col, sigma_col, z_col = [], [], []

    for i in range(n):
        if i < train_size:
            params = estimate_ou_params(df.loc[: train_size - 1, "spread"])
        else:
            start = max(0, i - cfg.window_size)
            params = estimate_ou_params(df.loc[start:i, "spread"])

        if i % cfg.update_every_bars == 0:
            rolling_rows.append({"datetime": df.loc[i, "datetime"], **params})

        mu = params["mu"]
        sigma = params["sigma"] if params["sigma"] and params["sigma"] > 1e-12 else np.nan
        z = (df.loc[i, "spread"] - mu) / sigma if pd.notna(sigma) else np.nan
        mu_col.append(mu)
        sigma_col.append(sigma)
        z_col.append(z)

    df2 = df.copy()
    df2["mu"] = mu_col
    df2["sigma"] = sigma_col
    df2["zscore"] = z_col
    return df2, pd.DataFrame(rolling_rows)


def run_backtest(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str, float]]:
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(cfg.initial_cash)
    cerebro.broker.setcommission(commission=cfg.commission)

    feed = PairData(dataname=df.set_index("datetime"))

    fut_df = df[["datetime", "futures_price"]].rename(columns={"futures_price": "close"}).set_index("datetime")
    fut_feed = bt.feeds.PandasData(dataname=fut_df)

    cerebro.adddata(feed, name="etf_pair")
    cerebro.adddata(fut_feed, name="futures")
    cerebro.addstrategy(OUPairTradingStrategy, entry_z=cfg.entry_z, exit_z=cfg.exit_z, max_fut=cfg.max_futures_contracts)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="ret")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    result = cerebro.run()[0]
    log_df = pd.DataFrame(result.logs)

    metrics = {
        "total_return": result.analyzers.ret.get_analysis().get("rtot", np.nan),
        "annual_return": result.analyzers.ret.get_analysis().get("rnorm", np.nan),
        "sharpe": result.analyzers.sharpe.get_analysis().get("sharperatio", np.nan),
        "max_drawdown": result.analyzers.dd.get_analysis().get("max", {}).get("drawdown", np.nan),
        "trade_count": result.analyzers.trades.get_analysis().get("total", {}).get("total", np.nan),
        "final_value": cerebro.broker.getvalue(),
    }
    return log_df, metrics


def save_outputs(df: pd.DataFrame, ou_hist: pd.DataFrame, trade_log: pd.DataFrame, metrics: Dict[str, float]) -> None:
    out_results = Path("outputs/results")
    out_fig = Path("outputs/figures")
    out_results.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    ou_hist.to_csv(out_results / "ou_params_history.csv", index=False)
    trade_log.to_csv(out_results / "trade_log.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_results / "performance_metrics.csv", index=False)

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(df["datetime"], df["spread"], label="spread")
    ax[1].plot(df["datetime"], df["zscore"], label="zscore")
    ax[1].axhline(2.0, color="r", ls="--")
    ax[1].axhline(-2.0, color="r", ls="--")
    ax[1].axhline(0.5, color="g", ls=":")
    ax[1].axhline(-0.5, color="g", ls=":")
    ax[0].legend(); ax[1].legend()
    fig.tight_layout(); fig.savefig(out_fig / "spread_zscore.png", dpi=150)

    if not trade_log.empty:
        trade_log["timestamp"] = pd.to_datetime(trade_log["timestamp"])
        eq = trade_log[["timestamp", "value"]].drop_duplicates("timestamp")
        plt.figure(figsize=(12, 4))
        plt.plot(eq["timestamp"], eq["value"])
        plt.title("Equity Curve")
        plt.tight_layout()
        plt.savefig(out_fig / "equity_curve.png", dpi=150)

        pos = trade_log[["timestamp", "signal"]].copy()
        pos["fut_pos"] = np.where(pos["signal"].eq("long_spread"), -1, np.where(pos["signal"].eq("short_spread"), 1, 0))
        pos["etf_pos"] = -pos["fut_pos"]
        plt.figure(figsize=(12, 4))
        plt.step(pos["timestamp"], pos["etf_pos"], where="post", label="ETF pos")
        plt.step(pos["timestamp"], pos["fut_pos"], where="post", label="Futures pos")
        plt.legend(); plt.tight_layout(); plt.savefig(out_fig / "positions.png", dpi=150)


def main() -> None:
    cfg = Config()
    setup_logging(Path("outputs/logs/trading_log.txt"))
    logging.info("Start OU pair trading backtest")
    df = load_and_align_data(cfg)
    logging.info("Aligned data shape=%s time=[%s,%s]", df.shape, df["datetime"].min(), df["datetime"].max())
    df, ou_hist = prepare_model_features(df, cfg)
    trade_log, metrics = run_backtest(df, cfg)
    save_outputs(df, ou_hist, trade_log, metrics)
    logging.info("Done. metrics=%s", metrics)


if __name__ == "__main__":
    main()
