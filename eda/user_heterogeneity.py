"""고객별 앞쪽과 뒤쪽 절반의 재구매율 차이(하락폭) 분포.

고객마다 pos<0.5 재구매율에서 pos>=0.5 재구매율을 뺀 값을 구해 분포를 그린다.
첫 주문(order_number=1)은 제외한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda.data import add_cart_position, load_orders
from eda import plotstyle  # noqa: F401

FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
TABLE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"

MIN_ITEMS_PER_HALF = 10  # 앞/뒤 각각 최소 상품 수


def main():
    df = load_orders(
        columns=["user_id", "order_id", "add_to_cart_order", "reordered", "order_number"]
    )
    df = add_cart_position(df)
    df = df[(df["order_size"] > 1) & (df["order_number"] >= 2)].copy()

    # 고객별 앞/뒤 절반 재구매율과 표본 수
    df["half"] = np.where(df["pos"] < 0.5, "front", "back")
    g = df.groupby(["user_id", "half"], observed=True)["reordered"].agg(["mean", "size"])
    wide = g.unstack("half")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.dropna(subset=["mean_front", "mean_back"])
    wide = wide[(wide["size_front"] >= MIN_ITEMS_PER_HALF)
                & (wide["size_back"] >= MIN_ITEMS_PER_HALF)]

    drop = (wide["mean_front"] - wide["mean_back"]) * 100  # 하락폭(%p)

    _report(drop)
    _plot(drop)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "metric": ["n_users", "mean_pp", "median_pp", "std_pp",
                   "share_flat_<5pp", "share_strong_>20pp", "share_negative"],
        "value": [
            len(drop), drop.mean(), drop.median(), drop.std(),
            (drop.abs() < 5).mean(), (drop > 20).mean(), (drop < 0).mean(),
        ],
    }).round(3).to_csv(TABLE_DIR / "03_user_heterogeneity.csv", index=False)


def _report(drop):
    print("=" * 60)
    print(f"per-customer front-back reorder drop (n={len(drop):,})")
    print(f"  mean   = {drop.mean():.1f}%p")
    print(f"  median = {drop.median():.1f}%p")
    print(f"  std    = {drop.std():.1f}%p")
    print(f"  share flat (|drop|<5%p)   = {(drop.abs()<5).mean()*100:.1f}%")
    print(f"  share strong (drop>20%p)  = {(drop>20).mean()*100:.1f}%")
    print(f"  share negative (drop<0)   = {(drop<0).mean()*100:.1f}%")
    print("=" * 60)


def _plot(drop):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(drop, bins=60, range=(-40, 80), color="#4c72b0", edgecolor="white", lw=0.3)
    ax.axvline(0, color="0.4", lw=1)
    ax.axvline(drop.mean(), color="#d62728", lw=2, ls="--",
               label=f"평균 {drop.mean():.1f}%p")
    ax.set_title("고객별 앞뒤 재구매율 하락폭")
    ax.set_xlabel("고객별 앞뒤 재구매율 차이 (%p)")
    ax.set_ylabel("고객 수")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "03_user_heterogeneity.png"
    fig.savefig(out, dpi=150)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
