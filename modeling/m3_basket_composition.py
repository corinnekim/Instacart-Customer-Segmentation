"""세그먼트별 장바구니 크기/구조를 필수재·탐색재 비중으로 나눠 막대로 그린다."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "modeling"
FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

ESS_COLOR = "#d1495b"  # 필수재
EXP_COLOR = "#2a9d8f"  # 탐색재


def main():
    cust = pd.read_parquet(OUT_DIR / "customer_segments.parquet")
    cust["ess_items"] = cust["avg_basket"] * cust["reorder_rate"]
    cust["exp_items"] = cust["avg_basket"] * cust["explore"]
    g = cust.groupby("segment").agg(
        basket=("avg_basket", "mean"),
        ess=("ess_items", "mean"),
        exp=("exp_items", "mean"),
    ).sort_values("basket")
    g["exp_share"] = g["exp"] / g["basket"]
    print(g.round(2).to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    y = range(len(g))
    ax.barh(y, g["ess"], color=ESS_COLOR, label="필수재 (재구매)")
    ax.barh(y, g["exp"], left=g["ess"], color=EXP_COLOR, label="탐색재 (새 상품)")
    for i, (_, r) in enumerate(g.iterrows()):
        ax.text(r["basket"] + 0.15, i, f"{r['exp_share']*100:.0f}% 탐색",
                va="center", fontsize=9, color="0.35")

    ax.set_yticks(list(y))
    ax.set_yticklabels(g.index)
    ax.set_xlabel("장바구니 크기 (평균 품목 수)")
    ax.set_title("세그먼트별 장바구니 구성: 필수재 vs 탐색재")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3, axis="x")
    ax.set_xlim(0, g["basket"].max() * 1.2)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "m3_basket_composition.png"
    fig.savefig(out, dpi=150)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
