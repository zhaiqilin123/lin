"""Reproduce all numbers in README.md.

Run: python analysis.py
Reads part_dispatches.csv, writes charts to ./charts/ and prints results.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COST_FAIL = 90  # work-hours lost per on-arrival shortage
COST_FLAG = 20  # work-hours spent per early shortage flag
OUT = Path("charts")
OUT.mkdir(exist_ok=True)


def load():
    df = pd.read_csv("part_dispatches.csv", encoding="utf-8")
    raw_rows, dup_rows = len(df), int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    # dispatch_time carries +08:00, stock_synced_at carries Z -> parse both as UTC
    df["dispatch_utc"] = pd.to_datetime(df.dispatch_time, utc=True)
    df["synced_utc"] = pd.to_datetime(df.stock_synced_at, utc=True)
    df["staleness_h"] = (df.dispatch_utc - df.synced_utc).dt.total_seconds() / 3600
    return df, raw_rows, dup_rows


def evaluate(flag, y) -> dict:
    flag = np.asarray(flag, dtype=bool)
    y = np.asarray(y, dtype=int)
    n = int(flag.sum())
    tp = int((flag & (y == 1)).sum())
    fp = n - tp
    fn = int((~flag & (y == 1)).sum())
    net = COST_FAIL * tp - COST_FLAG * n
    return {
        "flagged": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(tp / n, 4) if n else 0.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "flag_cost_h": COST_FLAG * n,
        "avoided_loss_h": COST_FAIL * tp,
        "net_saving_h": net,
    }


def main() -> None:
    df, raw_rows, dup_rows = load()
    d = df[df.fulfilled.notna()].copy()
    y = (d.fulfilled == 0).astype(int)
    fails = int(y.sum())
    nulls = int(df.fulfilled.isna().sum())

    # largest gap in sorted staleness = data-driven threshold
    s = np.sort(d.staleness_h.values)
    gaps = np.diff(s)
    gi = int(gaps.argmax())
    threshold = float((s[gi] + s[gi + 1]) / 2)
    print(f"largest staleness gap: [{s[gi]:.4f}, {s[gi+1]:.4f}] -> threshold {threshold:.4f}h")

    stale = d.staleness_h > threshold
    fresh_rate = float((d.loc[~stale, "fulfilled"] == 0).mean())
    stale_rate = float((d.loc[stale, "fulfilled"] == 0).mean())
    print(f"fresh fail rate {fresh_rate:.3f} (n={int((~stale).sum())}), "
          f"stale fail rate {stale_rate:.3f} (n={int(stale.sum())})")

    # ---- strategies on all evaluable rows ----
    strategies = {}
    strategies["do_nothing"] = {
        "flagged": 0, "tp": 0, "fp": 0, "fn": fails,
        "flag_cost_h": 0, "avoided_loss_h": 0, "net_saving_h": 0,
    }
    strategies["flag_all"] = evaluate(pd.Series(True, index=d.index), y)
    strategies["staleness_gate"] = evaluate(stale, y)
    strategies["stale_or_low_stock"] = evaluate(stale | (d.stock_qty <= 2), y)
    strategies["stale_and_low_stock"] = evaluate(stale & (d.stock_qty <= 2), y)

    print("\n== strategy comparison (all evaluable) ==")
    for k, v in strategies.items():
        print(k, v)

    # ---- net saving vs threshold curve ----
    curve = []
    for t in np.arange(0.0, 1.2, 0.01):
        r = evaluate(d.staleness_h > t, y)
        curve.append((round(float(t), 2), r["net_saving_h"], r["flagged"]))
    best_t, best_net, _ = max(curve, key=lambda x: x[1])
    print(f"\nbest threshold by net saving: {best_t}h -> {best_net}h")

    # ---- temporal holdout: Sep 1-5 train, Sep 6-7 test ----
    dates = d.dispatch_utc.dt.date
    train_dates = set(pd.date_range("2026-09-01", "2026-09-05").date)
    tr = dates.isin(train_dates).values
    te = ~tr
    d_tr, d_te = d[tr], d[te]
    y_tr, y_te = y[tr], y[te]

    s_tr = np.sort(d_tr.staleness_h.values)
    g_tr = np.diff(s_tr)
    i_tr = int(g_tr.argmax())
    thr_tr = float((s_tr[i_tr] + s_tr[i_tr + 1]) / 2)

    rule_te = evaluate(d_te.staleness_h > thr_tr, y_te)
    print(f"\n== temporal holdout ==")
    print(f"train rows {len(d_tr)} (fails {int(y_tr.sum())}), test rows {len(d_te)} (fails {int(y_te.sum())})")
    print(f"threshold learned on train: {thr_tr:.4f}h -> test rule: {rule_te}")

    # logistic on top of the freshness gate (numeric + categorical)
    num_cols = ["staleness_h", "stock_qty", "daily_ticket_count"]
    cat_cols = ["region_name", "device_category"]

    def design(frame):
        X = pd.get_dummies(
            frame[num_cols + cat_cols], columns=cat_cols, drop_first=True, dtype=float
        )
        return X.reindex(columns=feature_cols, fill_value=0.0)

    feature_cols = list(
        pd.get_dummies(d[num_cols + cat_cols], columns=cat_cols, drop_first=True, dtype=float).columns
    )
    scaler = StandardScaler().fit(d_tr[num_cols])
    X_tr = design(d_tr).copy()
    X_te = design(d_te).copy()
    X_tr[num_cols] = scaler.transform(d_tr[num_cols])
    X_te[num_cols] = scaler.transform(d_te[num_cols])

    model = LogisticRegression(max_iter=2000)
    model.fit(X_tr, y_tr)
    p_tr = model.predict_proba(X_tr)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    # pick probability cutoff on train by net saving, apply to test
    best = None
    for cut in np.arange(0.0, 0.96, 0.02):
        r = evaluate(pd.Series(p_tr >= cut), y_tr)
        if best is None or r["net_saving_h"] > best[1]:
            best = (float(cut), r["net_saving_h"])
    cut = best[0]
    model_te = evaluate(pd.Series(p_te >= cut), y_te)
    print(f"logistic cutoff {cut:.2f} (train net {best[1]}h) -> test: {model_te}")
    print("coefs:", dict(zip(feature_cols, np.round(model.coef_[0], 3))))

    # ---- charts ----
    # 1. failure rate vs staleness
    bins = np.arange(0, 1.21, 0.05)
    d["bin"] = pd.cut(d.staleness_h, bins)
    agg = d.groupby("bin", observed=True).agg(
        n=("fulfilled", "size"), fail=("fulfilled", lambda x: int((x == 0).sum()))
    )
    agg = agg[agg.n > 0]
    agg["rate"] = agg.fail / agg.n
    mid = [iv.mid for iv in agg.index]
    plt.figure(figsize=(9, 4.5))
    plt.plot(mid, agg.rate, "o-", color="#c0392b", lw=1.5, ms=4)
    plt.axvline(threshold, ls="--", color="#2c3e50", label=f"自然断点 {threshold:.2f}h")
    plt.xlabel("库存快照滞后（小时）")
    plt.ylabel("到场缺货率")
    plt.title("快照滞后 > 0.3h 后，缺货率从 3.6% 跳到 40%+")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "failure_vs_staleness.png", dpi=150)
    plt.close()

    # 2. net saving vs threshold
    xs = [c[0] for c in curve]
    ys = [c[1] for c in curve]
    plt.figure(figsize=(9, 4.5))
    plt.plot(xs, ys, color="#27ae60", lw=2)
    plt.axvline(threshold, ls="--", color="#2c3e50", label=f"自然断点 {threshold:.2f}h")
    plt.axhline(0, color="gray", lw=0.8)
    plt.xlabel("标记阈值：快照滞后超过 x 小时")
    plt.ylabel("净节省工时（7 天样本）")
    plt.title("阈值—净收益曲线：0.2~0.6h 区间收益最高")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "net_saving_vs_threshold.png", dpi=150)
    plt.close()

    # 3. strategy comparison
    names = ["flag_all", "stale_and_low_stock", "stale_or_low_stock", "staleness_gate"]
    labels = ["全量标记", "滞后且低库存(≤2)", "滞后或低库存(≤2)", "仅滞后门控"]
    vals = [strategies[n]["net_saving_h"] for n in names]
    plt.figure(figsize=(8, 4.2))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in vals]
    plt.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        plt.text(i, v + (80 if v >= 0 else -300), f"{v:,.0f}h", ha="center", fontsize=10)
    plt.axhline(0, color="gray", lw=0.8)
    plt.ylabel("净节省工时（7 天样本）")
    plt.title("策略对比：越“精准”越值钱，全量标记是亏的")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUT / "strategy_comparison.png", dpi=150)
    plt.close()

    summary = {
        "raw_rows": raw_rows,
        "duplicate_rows": dup_rows,
        "evaluable_rows": len(d),
        "missing_labels": nulls,
        "failures": fails,
        "baseline_loss_h": fails * COST_FAIL,
        "threshold_h": threshold,
        "fresh_fail_rate": round(fresh_rate, 4),
        "stale_fail_rate": round(stale_rate, 4),
        "strategies": strategies,
        "best_threshold_h": best_t,
        "best_net_h": best_net,
        "temporal_holdout": {"threshold": thr_tr, "rule": rule_te,
                             "logistic_cutoff": cut, "logistic": model_te,
                             "test_rows": len(d_te), "test_fails": int(y_te.sum())},
    }
    Path("results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote results.json and charts/")


if __name__ == "__main__":
    main()
