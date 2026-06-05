"""Kaggle raw CSV 6개를 조인해 분석용 parquet 하나로 만든다.

order_products(prior+train)에 주문·상품·카테고리 정보를 붙여
instacart_full_data.parquet(약 3,382만 행 × 12열)을 생성한다.
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"

KEEP = [
    "order_id", "product_id", "add_to_cart_order", "reordered",
    "user_id", "order_number", "order_dow", "order_hour_of_day",
    "days_since_prior_order", "product_name", "department", "aisle",
]


def main():
    order_products = pd.concat([
        pd.read_csv(DATA / "order_products__prior.csv"),
        pd.read_csv(DATA / "order_products__train.csv"),
    ], ignore_index=True)

    orders = pd.read_csv(DATA / "orders.csv")
    products = pd.read_csv(DATA / "products.csv")
    aisles = pd.read_csv(DATA / "aisles.csv")
    departments = pd.read_csv(DATA / "departments.csv")

    df = (order_products
          .merge(orders, on="order_id", how="left")
          .merge(products, on="product_id", how="left")
          .merge(aisles, on="aisle_id", how="left")
          .merge(departments, on="department_id", how="left"))

    out = DATA / "instacart_full_data.parquet"
    df[KEEP].to_parquet(out, index=False)
    print(f"saved {len(df):,} rows x {len(KEEP)} cols -> {out}")


if __name__ == "__main__":
    main()
