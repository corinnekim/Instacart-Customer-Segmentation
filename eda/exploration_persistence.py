"""고객의 초반 탐색 성향이 이후에도 이어지는지 본다.

초반(주문 2~5번)과 이후(6번~)의 새 상품 비율 상관을 본다. 주문 8회 이상만 사용.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

from eda.data import load_orders
from eda import plotstyle  # noqa: F401

FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
TABLE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"

MIN_ORDERS = 8
N_BINS = 10


def main():
    df = load_orders(columns=["user_id", "order_id", "reordered", "order_number"])

    # 주문 단위 새 상품 비율
    g = df.groupby("order_id")
    orders = pd.DataFrame({
        "user_id": g["user_id"].first(),
        "order_number": g["order_number"].first(),
        "new_share": 1 - g["reordered"].mean(),
    })

    umax = orders.groupby("user_id")["order_number"].max()
    cohort = umax[umax >= MIN_ORDERS].index
    o = orders[orders["user_id"].isin(cohort)]

    early = o[(o["order_number"] >= 2) & (o["order_number"] <= 5)]
    late = o[o["order_number"] >= 6]

    users = pd.DataFrame({
        "early": early.groupby("user_id")["new_share"].mean(),
        "late": late.groupby("user_id")["new_share"].mean(),
    }).dropna()

    pearson = users["early"].corr(users["late"])
    rho, _ = spearmanr(users["early"], users["late"])

    users["bin"] = pd.qcut(users["early"], N_BINS, labels=False, duplicates="drop")
    by_bin = users.groupby("bin").agg(early=("early", "mean"), late=("late", "mean"))
    baseline = users["late"].mean()

    print("=" * 60)
    print(f"cohort users (>= {MIN_ORDERS} orders) = {len(users):,}")
    print(f"early exploration vs late exploration:")
    print(f"  pearson r  = {pearson:.3f}")
    print(f"  spearman   = {rho:.3f}")
    print(f"  lowest decile late  = {by_bin['late'].iloc[0]:.3f}")
    print(f"  highest decile late = {by_bin['late'].iloc[-1]:.3f}")
    print(f"  baseline (overall late mean) = {baseline:.3f}")
    print("=" * 60)

    _plot(by_bin, baseline, pearson)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    by_bin.round(4).to_csv(TABLE_DIR / "05_exploration_persistence.csv", index=False)


def _plot(by_bin, baseline, pearson):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(by_bin["early"], by_bin["late"], color="#1f77b4", lw=2, marker="o", ms=6,
            label=f"실제 (r={pearson:.2f})")
    ax.axhline(baseline, color="0.5", lw=1.5, ls="--",
               label="성향 없을 때 기준선")
    ax.set_title("탐색 성향은 시간이 지나도 유지")
    ax.set_xlabel("초반 탐색 (주문 2-5번, 새 상품 비율)")
    ax.set_ylabel("이후 탐색 (주문 6번~, 새 상품 비율)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "05_exploration_persistence.png"
    fig.savefig(out, dpi=150)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
