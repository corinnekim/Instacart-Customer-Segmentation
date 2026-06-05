"""세그먼트별 정착률과 시도 수로 추천 기회 크기를 추정한다.

정착률이 전체 평균보다 낮은 세그먼트에 추천을 적용하는 시나리오를 본다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "modeling"
FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

UPLIFT = 0.01
SEG_COLORS = {
    "A 탐색형": "#2a9d8f",
    "B 습관형": "#1f77b4",
    "C 대형단골": "#9467bd",
    "D 저활동형": "#e76f51",
}


def main():
    cust = pd.read_parquet(OUT_DIR / "customer_segments.parquet")
    g = cust.groupby("segment").agg(
        customers=("segment", "size"),
        trials=("trials", "sum"),
        adopted=("adopted", "sum"),
    )
    g["adoption_rate"] = g["adopted"] / g["trials"]
    g = g.sort_values("adoption_rate")
    overall = g["adopted"].sum() / g["trials"].sum()

    print("=== segment funnel (value = adopted products) ===")
    print(g.round(3).to_string())
    print(f"overall adoption rate = {overall:.3f}")

    _scenario(g, overall)
    _plot(g, overall)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g.round(4).to_csv(OUT_DIR / "m5_opportunity.csv")
    print(f"\nsaved -> {OUT_DIR/'m5_opportunity.csv'}")


def _scenario(g, overall):
    # 정착률이 전체 평균보다 낮은 세그먼트 = 끌어올릴 여지가 있는 곳 (객관 기준)
    target = g[g["adoption_rate"] < overall]
    base = g["adopted"].sum()
    gain = target["trials"].sum() * UPLIFT
    print(f"\nscenario: recommend sticky items -> +{UPLIFT*100:.0f}%p adoption on below-average segments")
    for s in target.index:
        print(f"  {s}: {target.loc[s,'adoption_rate']:.1%} on {target.loc[s,'trials']:,.0f} trials")
    print(f"  +{UPLIFT*100:.0f}%p => +{gain:,.0f} adopted products (+{gain/base:.1%} over current {base:,.0f})")
    print("  assumption-based projection, not causal. validate with A/B.")


def _plot(g, overall):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    axL.bar(range(len(g)), g["adoption_rate"], color=[SEG_COLORS[s] for s in g.index], width=0.6)
    axL.axhline(overall, color="0.5", ls="--", lw=1, label=f"전체 {overall:.0%}")
    axL.set_xticks(range(len(g)))
    axL.set_xticklabels(g.index)
    axL.set_title("세그먼트별 정착률")
    axL.set_ylabel("정착률 (새 상품이 재구매된 비율)")
    axL.legend()
    axL.grid(alpha=0.3, axis="y")

    axR.bar(range(len(g)), g["trials"], color=[SEG_COLORS[s] for s in g.index], width=0.6)
    axR.set_xticks(range(len(g)))
    axR.set_xticklabels(g.index)
    axR.set_title("세그먼트별 신상품 시도량")
    axR.set_ylabel("시도 수 (첫 구매)")
    axR.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "m5_opportunity.png"
    fig.savefig(out, dpi=150)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
