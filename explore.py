import pandas as pd
import numpy as np

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

df = pd.read_csv("part_dispatches.csv", encoding="utf-8")
df["dispatch_utc"] = pd.to_datetime(df.dispatch_time, utc=True)
df["synced_utc"] = pd.to_datetime(df.stock_synced_at, utc=True)
df["staleness_h"] = (df.dispatch_utc - df.synced_utc).dt.total_seconds() / 3600

print("rows", len(df), "| dup full rows", df.duplicated().sum(),
      "| dup dispatch_id", df.dispatch_id.duplicated().sum())
print("negative staleness:", (df.staleness_h < 0).sum(),
      "| fulfilled null:", df.fulfilled.isna().sum())
print("\nstaleness_h describe:\n", df.staleness_h.describe())
print("\ndispatch date range:", df.dispatch_utc.min(), "->", df.dispatch_utc.max())

# duplicated dispatch_id: are rows identical ignoring id?
dup = df[df.dispatch_id.duplicated(keep=False)].sort_values("dispatch_id")
print("\ndup dispatch_id rows:", len(dup))
print(dup.head(6).to_string())

# dedupe exact duplicates, keep first
d = df.drop_duplicates().copy()
print("\nafter drop exact dups:", len(d))

# label stats
valid = d[d.fulfilled.notna()].copy()
print("\nvalid rows:", len(valid), "| fail rate:", f"{1 - valid.fulfilled.mean():.3f}",
      "| failures:", int((valid.fulfilled == 0).sum()))

# failure by staleness bins
bins = [-np.inf, 1, 3, 6, 12, 24, 48, np.inf]
valid["stale_bin"] = pd.cut(valid.staleness_h, bins)
g = valid.groupby("stale_bin", observed=True).agg(n=("fulfilled", "size"),
                                                 fail_rate=("fulfilled", lambda s: 1 - s.mean()))
print("\n== failure by staleness (hours) ==\n", g)

# failure by stock_qty bins
valid["qty_bin"] = pd.cut(valid.stock_qty, [-1, 0, 2, 5, 10, 15, 20, 31])
g2 = valid.groupby("qty_bin", observed=True).agg(n=("fulfilled", "size"),
                                                  fail_rate=("fulfilled", lambda s: 1 - s.mean()))
print("\n== failure by stock_qty ==\n", g2)

# interaction staleness x low stock
valid["low_stock"] = valid.stock_qty <= 2
print("\n== fail rate: stale x low stock ==")
print(valid.groupby(["low_stock", pd.cut(valid.staleness_h, [-np.inf, 6, 24, np.inf])],
                    observed=True).agg(n=("fulfilled", "size"),
                                       fail_rate=("fulfilled", lambda s: 1 - s.mean())))

# daily ticket count
valid["ticket_bin"] = pd.qcut(valid.daily_ticket_count, 5, duplicates="drop")
print("\n== failure by daily_ticket_count quintile ==")
print(valid.groupby("ticket_bin", observed=True).agg(n=("fulfilled", "size"),
                                                      fail_rate=("fulfilled", lambda s: 1 - s.mean())))

# region / device
print("\n== by region ==\n", valid.groupby("region_name").agg(n=("fulfilled", "size"),
                                                              fail_rate=("fulfilled", lambda s: 1 - s.mean())))
print("\n== by device (top/bottom) ==\n",
      valid.groupby("device_category").agg(n=("fulfilled", "size"),
                                           fail_rate=("fulfilled", lambda s: 1 - s.mean()))
      .sort_values("fail_rate", ascending=False))

# missingness of fulfilled
d["miss"] = d.fulfilled.isna()
print("\n== missing fulfilled by staleness bin ==")
d["stale_bin"] = pd.cut(d.staleness_h, bins)
print(d.groupby("stale_bin", observed=True).agg(n=("miss", "size"), miss_rate=("miss", "mean")))
print("\n== missing by region ==\n", d.groupby("region_name").agg(n=("miss", "size"),
                                                                  miss_rate=("miss", "mean")))

# correlation
print("\ncorr staleness/fail:", valid[["staleness_h", "fulfilled"]].corr().iloc[0, 1])
print("corr stock/fail:", valid[["stock_qty", "fulfilled"]].corr().iloc[0, 1])
print("corr tickets/fail:", valid[["daily_ticket_count", "fulfilled"]].corr().iloc[0, 1])

# dispatch hour
valid["hour"] = valid.dispatch_utc.dt.hour
print("\n== fail by dispatch hour ==\n", valid.groupby("hour").agg(n=("fulfilled", "size"),
                                                                    fail_rate=("fulfilled", lambda s: 1 - s.mean())))
