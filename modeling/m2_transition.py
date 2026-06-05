"""모델링 2단계: 정규화 위치별 평균 필수도 곡선.

필수도 점수를 장바구니 위치(앞=0, 뒤=1)로 10등분해 평균을 낸다.
앞에서 뒤로 갈수록 낮아지며, 필수재에서 탐색재로 넘어가는 모양을 보인다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda import plotstyle  # noqa: F401

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "modeling"
FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

N_BINS = 10


def main():
    df = pd.read_parquet(OUT_DIR / "necessity_valid.parquet")
    df["order_size"] = df.groupby("order_id")["add_to_cart_order"].transform("size")
    df = df[df["order_size"] > 1].copy()
    df["pos"] = (df["add_to_cart_order"] - 1) / (df["order_size"] - 1)

    edges = np.linspace(0, 1, N_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    df["bin"] = pd.cut(df["pos"], edges, include_lowest=True, labels=False)
    curve = df.groupby("bin")["necessity_score"].mean()
    front, back = curve.iloc[0], curve.iloc[-1]

    print(f"necessity by cart position: front {front:.3f} -> back {back:.3f} "
          f"(drop {front - back:.3f})")
    _plot(centers, curve.to_numpy(), front, back)


def _plot(centers, vals, front, back):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(centers, vals, marker="o", color="#2a9d8f", lw=2.5)
    ax.set_title("장바구니 앞에서 뒤로 갈수록 필수도가 내려간다")
    ax.set_xlabel("장바구니 위치 (0=앞, 1=뒤)")
    ax.set_ylabel("평균 필수도 점수")
    ax.annotate("필수재", xy=(0.04, front), xytext=(0.04, front),
                fontsize=10, color="0.3")
    ax.annotate("탐색재", xy=(0.78, back), xytext=(0.78, back),
                fontsize=10, color="0.3")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "m2_transition.png"
    fig.savefig(out, dpi=150)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
