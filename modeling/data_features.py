"""모델링용 데이터 로딩과 누설 없는(causal) feature 생성.

user 단위 통계는 '과거 주문만으로' 계산해 현재 행의 정답이 새지 않게 한다.
department/aisle/product 재구매율은 split 이후 train으로만 계산해 매핑한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "instacart_full_data.parquet"

BASE_COLS = [
    "user_id", "order_id", "product_id", "add_to_cart_order", "reordered",
    "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order",
    "department", "aisle",
]
CAT_RATE_COLS = ["department", "aisle", "product_id"]


def load_sample(n_users=None, seed=42):
    """user 단위로 데이터를 읽는다. n_users=None이면 전체, 숫자면 그만큼만 표본.
    한 user의 전체 주문 이력은 통째로 유지한다."""
    df = pd.read_parquet(DATA_PATH, columns=BASE_COLS)
    if n_users is None:
        return df
    users = df["user_id"].unique()
    rng = np.random.default_rng(seed)
    keep = rng.choice(users, size=min(n_users, len(users)), replace=False)
    return df[df["user_id"].isin(keep)].copy()


def add_causal_user_features(df):
    """현재 주문 직전까지의 이력으로 user 통계를 만든다 (order_number=1 포함해 계산)."""
    o = (df.groupby(["user_id", "order_number"])
           .agg(r=("reordered", "sum"), n=("reordered", "size"))
           .reset_index()
           .sort_values(["user_id", "order_number"]))
    g = o.groupby("user_id")
    o["cum_r"] = g["r"].cumsum() - o["r"]          # 직전까지 재구매 수
    o["cum_n"] = g["n"].cumsum() - o["n"]          # 직전까지 상품 수
    o["prior_orders"] = g.cumcount()               # 직전까지 주문 수
    o["user_reorder_rate_prior"] = o["cum_r"] / o["cum_n"].replace(0, np.nan)
    o["user_avg_basket_prior"] = o["cum_n"] / o["prior_orders"].replace(0, np.nan)

    feats = o[["user_id", "order_number", "prior_orders",
               "user_reorder_rate_prior", "user_avg_basket_prior"]]
    return df.merge(feats, on=["user_id", "order_number"], how="left")


def add_category_rates(train, frames, cols=CAT_RATE_COLS):
    """train에서만 카테고리 재구매율을 계산해 각 frame에 매핑한다 (미등장은 train 전체 평균)."""
    gmean = train["reordered"].mean()
    rate_maps = {c: train.groupby(c, observed=True)["reordered"].mean() for c in cols}
    out = []
    for d in frames:
        d = d.copy()
        for c in cols:
            mapped = d[c].astype(object).map(rate_maps[c])
            d[f"{c}_reorder_rate"] = pd.to_numeric(mapped, errors="coerce").fillna(gmean)
        out.append(d)
    return out
