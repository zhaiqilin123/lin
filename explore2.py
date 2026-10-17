import pandas as pd
import numpy as np

df = pd.read_csv("part_dispatches.csv", encoding="utf-8").drop_duplicates()
df["dispatch_utc"] = pd.to_datetime(df.dispatch_time, utc=True)
df["synced_utc"] = pd.to_datetime(df.stock_synced_at, utc=True)
df["staleness_h"] = (df.dispatch_utc - df.synced_utc).dt.total_seconds() / 3600
d = df[df.fulfilled.notna()].copy()
print("valid:", len(d), "fails:", int((d.fulfilled == 0).sum()))

print("\n== fine staleness bins (0.05h) ==")
d["sb"] = pd.cut(d.staleness_h, np.arange(0, 1.21, 0.05))
print(d.groupby("sb", observed=True).agg(n=("fulfilled", "size"),
                                         fail=("fulfilled", lambda s: int((s == 0).sum())),
                                         rate=("fulfilled", lambda s: 1 - s.mean())).to_string())

print("\n== staleness > x cumulative ==")
for x in [0.25, 0.5, 0.75, 0.9, 1.0]:
    m = d.staleness_h > x
    n = int(m.sum()); f = int((d.loc[m, "fulfilled"] == 0).sum())
    print(f">{x:.2f}h: flagged={n}, true_fails={f}, precision={f/n:.3f}, "
          f"net_save={90*f - 20*n:.0f}h")

print("\n== fail rate staleness>1 x stock_qty ==")
d["stale"] = d.staleness_h > 1
d["qb"] = pd.cut(d.stock_qty, [-1, 2, 5, 10, 15, 20, 31])
print(d.groupby(["stale", "qb"], observed=True).agg(n=("fulfilled", "size"),
                                                    fail=("fulfilled", lambda s: int((s == 0).sum())),
                                                    rate=("fulfilled", lambda s: 1 - s.mean())).to_string())

print("\n== fail rate staleness>0.75 x ticket quintile ==")
d["tb"] = pd.qcut(d.daily_ticket_count, 4)
print(d.groupby(["stale", "tb"], observed=True).agg(n=("fulfilled", "size"),
                                                    rate=("fulfilled", lambda s: 1 - s.mean())).to_string())

print("\n== staleness>1 by region/device ==")
print(d[d.stale].groupby("region_name").agg(n=("fulfilled", "size"),
                                            rate=("fulfilled", lambda s: 1 - s.mean())))
print(d[d.stale].groupby("device_category").agg(n=("fulfilled", "size"),
                                                rate=("fulfilled", lambda s: 1 - s.mean())))

# how many daily dispatches overall / staleness over time
print("\n== staleness by date ==")
d["date"] = d.dispatch_utc.dt.date
print(d.groupby("date").agg(n=("fulfilled", "size"),
                            stale_rate=("stale", "mean"),
                            fail_rate=("fulfilled", lambda s: 1 - s.mean())))

# simple combined rule check
for thr in [0.75, 0.9, 1.0]:
    for q in [2, 5, 10, 15]:
        m = (d.staleness_h > thr) | (d.stock_qty <= q)
        n = int(m.sum()); f = int((d.loc[m, "fulfilled"] == 0).sum())
        print(f"OR rule stale>{thr} or qty<={q}: n={n}, TP={f}, prec={f/n:.3f}, net={90*f-20*n:.0f}h")
