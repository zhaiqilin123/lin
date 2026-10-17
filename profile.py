"""Generic CSV profiler: run as `python profile.py data.csv`."""
import sys
from pathlib import Path

import pandas as pd


def load(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise SystemExit(f"cannot read {path}: {last_err}")


def main() -> None:
    path = Path(sys.argv[1])
    df = load(path)
    out = []
    p = out.append

    p(f"file: {path}  rows: {len(df)}  cols: {len(df.columns)}")
    p("\n== columns ==")
    for c in df.columns:
        s = df[c]
        p(
            f"- {c!r}: dtype={s.dtype}, missing={s.isna().sum()} "
            f"({s.isna().mean():.1%}), unique={s.nunique(dropna=True)}"
        )
        if pd.api.types.is_numeric_dtype(s):
            p(f"    min={s.min()}, max={s.max()}, mean={s.mean():.2f}, median={s.median()}")
        else:
            vc = s.value_counts(dropna=False).head(8)
            p(f"    top: {dict(vc)}")

    p("\n== sample ==")
    p(df.head(5).to_string())

    full = "\n".join(out)
    print(full)
    Path("profile_report.txt").write_text(full, encoding="utf-8")


if __name__ == "__main__":
    main()
