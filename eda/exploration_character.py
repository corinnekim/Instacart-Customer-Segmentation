"""위치별 department 구성과 상품 등장 빈도.

Panel A 위치 bin별 department 점유율.
Panel B 위치 bin별 상품 전체 등장 빈도(log10) 평균.
department 선택은 front에서 back으로의 점유율 변화량 상위 N개와 하위 N개.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda.data import add_cart_position, load_orders
from eda import plotstyle  # noqa: F401

FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
TABLE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"

N_BINS = 10
N_PICK = 3  # 뽑을 department 수


def main():
    df = load_orders(columns=["order_id", "add_to_cart_order", "department", "product_id"])
    df = add_cart_position(df)
    df = df[df["order_size"] > 1].copy()  # 단일 상품 주문 제외

    edges = np.linspace(0, 1, N_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    df["pos_bin"] = pd.cut(df["pos"], edges, include_lowest=True, labels=False)

    # 위치 bin별 department 점유율(%)
    counts = pd.crosstab(df["pos_bin"], df["department"])
    share = counts.div(counts.sum(axis=1), axis=0) * 100
    delta = share.iloc[-1] - share.iloc[0]   # back - front (%p)
    risers = delta.sort_values(ascending=False).head(N_PICK).index.tolist()
    fallers = delta.sort_values().head(N_PICK).index.tolist()

    # 위치 bin별 상품 등장 빈도(log10) 평균
    freq = df.groupby("product_id")["order_id"].transform("size")
    df["log_pop"] = np.log10(freq)
    pop_by_bin = df.groupby("pos_bin")["log_pop"].mean()

    _report(share, delta, risers, fallers, pop_by_bin)
    _plot(centers, share, risers, fallers, pop_by_bin)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    tbl = pd.DataFrame({
        "front_share_%": share.iloc[0],
        "back_share_%": share.iloc[-1],
        "delta_pp": delta,
    }).sort_values("delta_pp", ascending=False).round(2)
    tbl.index.name = "department"
    tbl.to_csv(TABLE_DIR / "02_exploration_character.csv")


def _report(share, delta, risers, fallers, pop_by_bin):
    print("=" * 64)
    print("department share: front bin -> back bin (%p change)")
    print("  rises toward back (exploration-leaning):")
    for d in risers:
        print(f"    {d:<16} {share.iloc[0][d]:5.1f}% -> {share.iloc[-1][d]:5.1f}%  ({delta[d]:+.1f}%p)")
    print("  concentrates at front (staple-leaning):")
    for d in fallers:
        print(f"    {d:<16} {share.iloc[0][d]:5.1f}% -> {share.iloc[-1][d]:5.1f}%  ({delta[d]:+.1f}%p)")
    print(f"\nproduct popularity (log10 freq): front {pop_by_bin.iloc[0]:.2f} "
          f"-> back {pop_by_bin.iloc[-1]:.2f}")
    print("=" * 64)


def _plot(centers, share, risers, fallers, pop_by_bin):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    for d in fallers:
        axL.plot(centers, share[d], lw=2, marker="o", ms=4, label=d)
    for d in risers:
        axL.plot(centers, share[d], lw=2, ls="--", marker="s", ms=4, label=d)
    axL.set_title("장바구니 위치별 카테고리(department) 점유율")
    axL.set_xlabel("장바구니 위치 (0=앞, 1=뒤)")
    axL.set_ylabel("상품 비율 (%)")
    axL.legend()
    axL.grid(alpha=0.3)

    axR.plot(centers, pop_by_bin.to_numpy(), color="#d62728", lw=2, marker="o", ms=4)
    axR.set_title("장바구니 위치별 상품 대중성")
    axR.set_xlabel("장바구니 위치 (0=앞, 1=뒤)")
    axR.set_ylabel("평균 log10(상품 등장 빈도)")
    axR.grid(alpha=0.3)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "02_exploration_character.png"
    fig.savefig(out, dpi=150)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
