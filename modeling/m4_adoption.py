"""신규 상품(reordered=0)이 이후 다시 구매되는지(정착) 예측하는 모델.

LogReg·LightGBM·XGBoost를 누설 없는 feature로 학습·평가한다.
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

from eda.data import add_cart_position
from eda import plotstyle  # noqa: F401
from modeling.data_features import add_causal_user_features, add_category_rates, load_sample

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "modeling"
FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

FEATURES = [
    "pos", "add_to_cart_order", "order_size", "order_number",
    "days_since_prior_order", "order_dow", "order_hour_of_day",
    "prior_orders", "user_reorder_rate_prior", "user_avg_basket_prior",
    "product_id_reorder_rate", "department_reorder_rate", "aisle_reorder_rate",
]


def main():
    df = add_cart_position(add_causal_user_features(load_sample()))
    df["user_max_order"] = df.groupby("user_id")["order_number"].transform("max")

    # 정착 라벨: 한 번이라도 재구매된 (user, product)
    adopted = df.loc[df["reordered"] == 1, ["user_id", "product_id"]].drop_duplicates()
    adopted["adopted"] = 1

    firsts = df[(df["reordered"] == 0)
                & (df["order_number"] >= 2)
                & (df["order_number"] < df["user_max_order"])].copy()
    firsts = firsts.merge(adopted, on=["user_id", "product_id"], how="left")
    firsts["adopted"] = firsts["adopted"].fillna(0).astype(int)

    # user 단위 분리 후, 카테고리 재구매율은 train user의 전체 행으로만 계산
    users = df["user_id"].unique()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_u, va_u = next(gss.split(users, groups=users))
    train_users = set(users[tr_u])
    train_full = df[df["user_id"].isin(train_users)]

    f_train = firsts[firsts["user_id"].isin(train_users)].copy()
    f_valid = firsts[~firsts["user_id"].isin(train_users)].copy()
    f_train, f_valid = add_category_rates(train_full, [f_train, f_valid])
    print(f"first-purchases  train {len(f_train):,} / valid {len(f_valid):,}")
    print(f"adoption base rate (valid) = {f_valid['adopted'].mean():.4f}")

    Xtr, ytr = f_train[FEATURES], f_train["adopted"]
    Xva, yva = f_valid[FEATURES], f_valid["adopted"]

    preds = {}
    logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(max_iter=1000))
    logreg.fit(Xtr, ytr)
    preds["LogReg"] = logreg.predict_proba(Xva)[:, 1]

    lgbm = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=64,
                              min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, verbose=-1)
    lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)],
             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    preds["LightGBM"] = lgbm.predict_proba(Xva)[:, 1]

    xgbm = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
                             random_state=42, early_stopping_rounds=50)
    xgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    preds["XGBoost"] = xgbm.predict_proba(Xva)[:, 1]

    rows = [{"model": k, "AUC": roc_auc_score(yva, p),
             "PR_AUC": average_precision_score(yva, p),
             "Brier": brier_score_loss(yva, p)} for k, p in preds.items()]
    report = pd.DataFrame(rows).round(4)
    print("\n=== adoption model comparison (valid) ===")
    print(report.to_string(index=False))

    imp = pd.Series(lgbm.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n=== LightGBM feature importance (gain) ===")
    print(imp.to_string())

    best = max(preds, key=lambda k: roc_auc_score(yva, preds[k]))
    f_valid["score"] = preds[best]
    _plot(f_valid, imp, best)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT_DIR / "m4_adoption_comparison.csv", index=False)
    print(f"\nbest model = {best}")


def _plot(f_valid, imp, best):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    top = imp.head(8).iloc[::-1]
    axL.barh(top.index, top.values, color="#1f77b4")
    axL.set_title("변수 중요도 (Feature Importance)")
    axL.set_xlabel("Gain")
    axL.grid(alpha=0.3, axis="x")

    dec = pd.qcut(f_valid["score"], 10, labels=False, duplicates="drop")
    lift = f_valid.groupby(dec)["adopted"].mean() * 100
    base = f_valid["adopted"].mean() * 100
    axR.plot(lift.index + 1, lift.values, marker="o", color="#2a9d8f", lw=2)
    axR.axhline(base, color="0.5", ls="--", lw=1, label=f"전체 {base:.0f}%")
    axR.set_title(f"예측 점수 10등분별 정착률 ({best})")
    axR.set_xlabel("예측 정착 점수 10등분 (1=낮음, 10=높음)")
    axR.set_ylabel("실제 정착률 (%)")
    axR.legend()
    axR.grid(alpha=0.3)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "m4_adoption.png"
    fig.savefig(out, dpi=150)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
