"""장바구니 크기를 고정한 위치별 재구매율.

Panel A 고정 크기(5/10/20)에서 add_to_cart_order별 재구매율.
Panel B 장바구니 크기별 첫 상품과 마지막 상품의 재구매율 차이.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from eda.data import add_cart_position, load_orders

FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
TABLE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"

CURVE_SIZES = [5, 10, 20]   # Panel A 대표 장바구니 크기
MIN_ORDERS = 2000           # Panel B 최소 주문 수


def user_avg_by_position(df):
    """(order_size, add_to_cart_order)별 user 평균 후 user 간 평균낸 재구매율."""
    per_user = df.groupby(
        ["order_size", "add_to_cart_order", "user_id"], observed=True
    )["reordered"].mean()
    return per_user.groupby(["order_size", "add_to_cart_order"]).mean()


def main():
    df = load_orders(columns=["user_id", "order_id", "add_to_cart_order", "reordered"])
    df = add_cart_position(df)
    df = df[df["order_size"] > 1].copy()  # 단일 상품 주문 제외

    # 크기별 주문 수
    order_counts = (
        df.drop_duplicates("order_id").groupby("order_size")["order_id"].size()
    )

    # 첫 위치와 마지막 위치의 user-averaged 재구매율
    fronts = df[df["add_to_cart_order"] == 1]
    backs = df[df["add_to_cart_order"] == df["order_size"]]
    front_rate = (
        fronts.groupby(["order_size", "user_id"])["reordered"].mean()
        .groupby("order_size").mean()
    )
    back_rate = (
        backs.groupby(["order_size", "user_id"])["reordered"].mean()
        .groupby("order_size").mean()
    )

    drop = pd.DataFrame(
        {
            "n_orders": order_counts,
            "front_rate": front_rate,
            "back_rate": back_rate,
        }
    ).dropna()
    drop["drop_pp"] = (drop["front_rate"] - drop["back_rate"]) * 100
    drop = drop[(drop.index >= 2) & (drop["n_orders"] >= MIN_ORDERS)]

    # Panel A 위치별 곡선
    curves = {}
    ua = user_avg_by_position(df[df["order_size"].isin(CURVE_SIZES)])
    for s in CURVE_SIZES:
        c = ua.loc[s]
        curves[s] = (c.index.to_numpy(), c.to_numpy() * 100)

    _report(drop)
    _plot(curves, drop)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = drop.reset_index().rename(columns={"order_size": "basket_size"})
    out[["basket_size", "n_orders", "front_rate", "back_rate", "drop_pp"]].round(4).to_csv(
        TABLE_DIR / "01_position_confound.csv", index=False
    )


def _report(drop):
    print("=" * 60)
    print("front-back reorder drop by basket size (user-averaged)")
    show = [2, 3, 5, 10, 15, 20, 30]
    for s in show:
        if s in drop.index:
            r = drop.loc[s]
            print(f"  size {s:>3}: front {r.front_rate*100:5.1f}%  "
                  f"back {r.back_rate*100:5.1f}%  drop {r.drop_pp:5.1f}%p  "
                  f"(n_orders={int(r.n_orders):,})")
    print(f"\nrange of drop: {drop['drop_pp'].min():.1f}%p (small) "
          f"-> {drop['drop_pp'].max():.1f}%p (large)")
    print("=" * 60)


def _plot(curves, drop):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    cmap = plt.cm.viridis([0.2, 0.5, 0.8])
    for color, (s, (pos, rate)) in zip(cmap, curves.items()):
        axL.plot(pos, rate, color=color, lw=2, marker="o", ms=4, label=f"size {s}")
    axL.set_title("Reorder Rate by Cart Position (fixed basket size)")
    axL.set_xlabel("Add-to-cart order")
    axL.set_ylabel("Reorder rate (%)")
    axL.set_ylim(0, 100)
    axL.legend()
    axL.grid(alpha=0.3)

    axR.plot(drop.index, drop["drop_pp"], color="#1f77b4", lw=2, marker="o", ms=3)
    axR.set_title("Front-Back Reorder Gap Widens with Basket Size")
    axR.set_xlabel("Basket size (items)")
    axR.set_ylabel("Front - back reorder gap (%p)")
    axR.set_ylim(0, None)
    axR.grid(alpha=0.3)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "01_position_confound.png"
    fig.savefig(out, dpi=150)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
