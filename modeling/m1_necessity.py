"""reordered 확률을 필수도 점수로 쓰는 모델.

누설 없는 feature로 LogReg·LightGBM·XGBoost를 비교하고, 검증셋 점수를 저장한다.
"""

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modeling.data_features import add_causal_user_features, add_category_rates, load_sample
from eda import plotstyle  # noqa: F401

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "modeling"
FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

FEATURES = [
    "add_to_cart_order", "order_number", "days_since_prior_order",
    "order_dow", "order_hour_of_day", "prior_orders",
    "user_reorder_rate_prior", "user_avg_basket_prior",
    "department_reorder_rate", "aisle_reorder_rate", "product_id_reorder_rate",
]


def main():
    df = add_causal_user_features(load_sample())
    df = df[df["order_number"] >= 2].copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_idx, va_idx = next(splitter.split(df, groups=df["user_id"]))
    train, valid = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
    train, valid = add_category_rates(train, [train, valid])
    print(f"train {len(train):,} / valid {len(valid):,} "
          f"(users {train['user_id'].nunique():,}/{valid['user_id'].nunique():,})")

    Xtr, ytr = train[FEATURES], train["reordered"].astype(int)
    Xva, yva = valid[FEATURES], valid["reordered"].astype(int)

    preds = {}

    logreg = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=1000, n_jobs=-1))
    logreg.fit(Xtr, ytr)
    preds["LogReg"] = logreg.predict_proba(Xva)[:, 1]

    lgbm = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=64,
        min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbose=-1)
    lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)],
             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    preds["LightGBM"] = lgbm.predict_proba(Xva)[:, 1]

    xgbm = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
        n_jobs=-1, random_state=42, early_stopping_rounds=50)
    xgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    preds["XGBoost"] = xgbm.predict_proba(Xva)[:, 1]

    rows = []
    for name, p in preds.items():
        rows.append({
            "model": name,
            "AUC": roc_auc_score(yva, p),
            "PR_AUC": average_precision_score(yva, p),
            "Brier": brier_score_loss(yva, p),
        })
    report = pd.DataFrame(rows).round(4)
    print("\n=== model comparison (valid, leakage-safe features) ===")
    print(report.to_string(index=False))
    print(f"\nbaseline rate (P(reorder)) = {yva.mean():.4f}")

    imp = pd.Series(lgbm.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n=== LightGBM feature importance (gain) ===")
    print(imp.to_string())

    best = max(preds, key=lambda k: roc_auc_score(yva, preds[k]))
    valid["necessity_score"] = preds[best]
    _plot(valid, best)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT_DIR / "m1_model_comparison.csv", index=False)
    valid[["user_id", "order_id", "add_to_cart_order", "reordered",
           "order_number", "necessity_score"]].to_parquet(
        OUT_DIR / "necessity_valid.parquet", index=False)
    print(f"\nbest model = {best}. saved scores -> {OUT_DIR/'necessity_valid.parquet'}")


def _plot(valid, best):
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, color, name in [(1, "#1f77b4", "재구매 = 필수재"),
                               (0, "#e76f51", "신규 = 탐색재")]:
        s = valid.loc[valid["reordered"] == label, "necessity_score"]
        ax.hist(s, bins=50, density=True, alpha=0.55, color=color, label=name)
    ax.set_title(f"필수도 점수 분포 ({best})")
    ax.set_xlabel("예측 필수도 점수 (0=탐색재, 1=필수재)")
    ax.set_ylabel("밀도")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "m1_necessity.png"
    fig.savefig(out, dpi=150)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
