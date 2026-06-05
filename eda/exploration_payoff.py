"""첫 구매 상품의 재구매율(정착률).

고객이 처음 산 상품(reordered=0)이 다시 구매되는 비율을 첫 구매 위치별로 계산한다.
정착을 두 가지로 본다.
- ever: 이후 주문 내력 중 한 번이라도 재구매
- next: 바로 다음 주문에서 재구매
첫 주문과 마지막 주문의 첫 구매는 제외한다.
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


def main():
    df = load_orders(
        columns=["user_id", "product_id", "order_id",
                 "add_to_cart_order", "reordered", "order_number"]
    )
    df = add_cart_position(df)
    df = df[df["order_size"] > 1].copy()  # 단일 상품 주문 제외

    # 고객별 마지막 주문 번호
    df["user_max_order"] = df.groupby("user_id")["order_number"].transform("max")

    # 첫 구매(reordered=0) 행
    firsts = df.loc[
        df["reordered"] == 0,
        ["user_id", "product_id", "pos", "order_number", "user_max_order"],
    ].copy()
    # 첫 주문과 마지막 주문의 첫 구매 제외
    firsts = firsts[(firsts["order_number"] >= 2)
                    & (firsts["order_number"] < firsts["user_max_order"])]

    # ever: 재구매된 적 있는 (user, product)
    adopted_pairs = df.loc[df["reordered"] == 1, ["user_id", "product_id"]].drop_duplicates()
    adopted_pairs["adopted_ever"] = 1.0
    firsts = firsts.merge(adopted_pairs, on=["user_id", "product_id"], how="left")
    firsts["adopted_ever"] = firsts["adopted_ever"].fillna(0.0)

    # next: 바로 다음 주문(order_number+1)에 같은 상품 구매
    purchases = df[["user_id", "product_id", "order_number"]].drop_duplicates()
    firsts["next_order"] = firsts["order_number"] + 1
    firsts = firsts.merge(
        purchases.rename(columns={"order_number": "_matched"}),
        left_on=["user_id", "product_id", "next_order"],
        right_on=["user_id", "product_id", "_matched"],
        how="left",
    )
    firsts["adopted_next"] = firsts["_matched"].notna().astype(float)

    edges = np.linspace(0, 1, N_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    firsts["pos_bin"] = pd.cut(firsts["pos"], edges, include_lowest=True, labels=False)
    ever_by_pos = firsts.groupby("pos_bin")["adopted_ever"].mean() * 100
    next_by_pos = firsts.groupby("pos_bin")["adopted_next"].mean() * 100

    _report(firsts, "ever", ever_by_pos)
    _report(firsts, "next", next_by_pos)
    _plot(centers, ever_by_pos, next_by_pos,
          firsts["adopted_ever"].mean() * 100, firsts["adopted_next"].mean() * 100)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "pos_bin": ever_by_pos.index,
        "adoption_ever_%": ever_by_pos.values,
        "adoption_next_%": next_by_pos.values,
    }).round(2).to_csv(TABLE_DIR / "04_exploration_payoff.csv", index=False)


def _report(firsts, name, by_pos):
    overall = firsts[f"adopted_{name}"].mean() * 100
    print(f"[{name}] n={len(firsts):,}  overall={overall:.1f}%  "
          f"front={by_pos.iloc[0]:.1f}%  back={by_pos.iloc[-1]:.1f}%  "
          f"front-back={by_pos.iloc[0] - by_pos.iloc[-1]:.1f}%p")


def _panel(ax, centers, by_pos, overall, color, title):
    vals = by_pos.to_numpy()
    ax.plot(centers, vals, color=color, lw=2, marker="o", ms=5)
    ax.axhline(overall, color="0.5", lw=1, ls="--", label=f"전체 {overall:.1f}%")
    margin = (vals.max() - vals.min()) * 0.4 + 0.5
    ax.set_ylim(vals.min() - margin, vals.max() + margin)
    ax.set_title(title)
    ax.set_xlabel("첫 구매 시 장바구니 위치 (0=앞, 1=뒤)")
    ax.set_ylabel("재구매율 (%)")
    ax.legend()
    ax.grid(alpha=0.3)


def _plot(centers, ever_by_pos, next_by_pos, overall_ever, overall_next):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
    _panel(axL, centers, ever_by_pos, overall_ever, "#2a9d8f",
           "첫 구매 위치별 이후 재구매율")
    _panel(axR, centers, next_by_pos, overall_next, "#e76f51",
           "첫 구매 위치별 다음 주문 재구매율")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "04_exploration_payoff.png"
    fig.savefig(out, dpi=150)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
