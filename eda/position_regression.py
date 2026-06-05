"""위치 효과를 다른 요인과 함께 통제하는 로지스틱 회귀.

reordered를 위치(pos)와 다른 요인(장바구니 크기, 주문 수, 경과일)을 함께 넣어
위치의 순수 효과를 본다. odds ratio 표와 예측 재구매 확률 그래프를 만든다.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.append(str(Path(__file__).resolve().parents[1]))  # 실행 버튼(직접 실행)에서도 임포트되게
from eda.data import add_cart_position, load_orders
from eda import plotstyle  # noqa: F401

FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
TABLE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"

SAMPLE_N = 2_000_000
PREDICTORS = ["pos", "order_size", "order_number", "days_since_prior_order"]


def main():
    df = load_orders(
        columns=["order_id", "add_to_cart_order", "reordered",
                 "order_number", "days_since_prior_order"]
    )
    df = add_cart_position(df)
    df = df[(df["order_size"] > 1) & (df["order_number"] >= 2)]
    df = df.dropna(subset=["days_since_prior_order"])

    sample = df.sample(n=SAMPLE_N, random_state=42)  # 회귀 적합용 표본

    X = sample[PREDICTORS].astype(float)
    mean, sd = X.mean(), X.std()
    Xz = sm.add_constant((X - mean) / sd)  # 표준화
    y = sample["reordered"].astype(int)

    res = sm.Logit(y, Xz).fit(disp=False)

    coef = res.params.drop("const")
    ci = res.conf_int().drop("const")
    out = pd.DataFrame({
        "coef": coef,
        "odds_ratio_per_1sd": np.exp(coef),
        "or_ci_low": np.exp(ci[0]),
        "or_ci_high": np.exp(ci[1]),
        "p_value": res.pvalues.drop("const"),
    })

    # pos는 0~1 범위라 front->back 전체 이동의 odds ratio도 계산
    pos_full_or = np.exp(coef["pos"] / sd["pos"])

    print("=" * 64)
    print(f"logistic regression on reordered (n={SAMPLE_N:,})")
    print(out.round(4).to_string())
    print(f"\npos front->back odds ratio (full 0->1 move) = {pos_full_or:.3f}")
    print("=" * 64)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out.round(5).to_csv(TABLE_DIR / "01b_position_regression.csv")

    _plot_effects(res, mean, sd)


def _predict_grid(res, mean, sd, var, grid):
    """var만 grid로 움직이고 나머지 변수는 평균(z=0)에 고정해 예측 확률을 낸다."""
    design = pd.DataFrame(0.0, index=np.arange(len(grid)), columns=["const"] + PREDICTORS)
    design["const"] = 1.0
    design[var] = (grid - mean[var]) / sd[var]
    return res.predict(design) * 100


def _plot_effects(res, mean, sd):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    pos_grid = np.linspace(0, 1, 50)
    axL.plot(pos_grid, _predict_grid(res, mean, sd, "pos", pos_grid),
             color="#2a9d8f", lw=2.5)
    axL.set_title("장바구니 위치별 예측 재구매 확률")
    axL.set_xlabel("장바구니 위치 (0=앞, 1=뒤)")
    axL.set_ylabel("예측 재구매 확률 (%)")
    axL.grid(alpha=0.3)

    on_grid = np.linspace(2, 50, 50)
    axR.plot(on_grid, _predict_grid(res, mean, sd, "order_number", on_grid),
             color="#1f77b4", lw=2.5)
    axR.set_title("주문 수별 예측 재구매 확률")
    axR.set_xlabel("주문 수")
    axR.set_ylabel("예측 재구매 확률 (%)")
    axR.grid(alpha=0.3)

    fig.tight_layout()
    out_path = FIG_DIR / "01b_position_regression.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved figure -> {out_path}")


if __name__ == "__main__":
    main()
