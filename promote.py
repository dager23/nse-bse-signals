"""Wave 4 stage 2: select screen survivors for full-engine promotion.

Reads results/screen/*.parquet, applies sanity filters, ranks by screen
Sharpe, and picks the top configs per family (diversity caps) into
results/promoted_specs.csv, which cat_R_promoted.py interprets.
"""
import glob
import os

import pandas as pd

import config

CAPS = {"V": 60, "T": 60, "U": 60, "W": 120, "S": 100}


def main():
    parts = glob.glob(os.path.join(config.RESULTS_DIR, "screen", "*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    total = len(df)
    ok = df[(df.exposure > 0.03) & (df.turn_yr < 150) & df.sharpe.notna()]
    picks = []
    for fam, cap in CAPS.items():
        sub = ok[ok.family == fam].sort_values("sharpe", ascending=False)
        # light dedup: for W keep at most 3 specs sharing the same first condition
        if fam == "W":
            first = sub.spec.str.split("|").str[1].str.split("&").str[0]
            sub = sub.groupby(first, sort=False).head(3)
        picks.append(sub.head(cap))
    out = pd.concat(picks, ignore_index=True)
    out.to_csv(os.path.join(config.RESULTS_DIR, "promoted_specs.csv"), index=False)
    print(f"screened total: {total}, sane: {len(ok)}, promoted: {len(out)}")
    print(out.groupby("family").sharpe.agg(["count", "max", "mean"]).round(2))
    print("\ntop 10 by screen sharpe:")
    print(out.nlargest(10, "sharpe")[["family", "spec", "sharpe", "cagr", "maxdd"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
