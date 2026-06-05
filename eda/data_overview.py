"""데이터 규모와 핵심 변수 분포(중앙값·사분위)를 출력한다."""

from pathlib import Path

import pandas as pd

from eda.data import load_orders

TABLE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"


def _quants(s):
    q = s.quantile([0.25, 0.5, 0.75, 0.95])
    return {"median": s.median(), "q1": q.loc[0.25], "q3": q.loc[0.75],
            "p95": q.loc[0.95], "max": s.max()}


def main():
    df = load_orders(columns=["user_id", "order_id", "product_id", "order_number",
                              "days_since_prior_order", "reordered"])

    orders_per_user = df.groupby("user_id")["order_id"].nunique()
    basket = df.groupby("order_id").size()

    scale = {
        "rows": len(df),
        "users": df["user_id"].nunique(),
        "orders": df["order_id"].nunique(),
        "products": df["product_id"].nunique(),
        "overall_reorder_rate": round(df["reordered"].mean(), 4),
        "days_since_null_%": round(df["days_since_prior_order"].isna().mean() * 100, 2),
    }

    dist = pd.DataFrame({
        "orders_per_user": _quants(orders_per_user),
        "basket_size": _quants(basket),
        "days_since_prior": _quants(df["days_since_prior_order"].dropna()),
    }).T.round(1)

    print("scale:")
    for k, v in scale.items():
        print(f"  {k:>22}: {v:,}" if isinstance(v, int) else f"  {k:>22}: {v}")
    print("\ndistribution (median/quartile, not min-max):")
    print(dist.to_string())
    print("\nnote: orders/user max 100, days_since max 30 are dataset caps.")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    dist.to_csv(TABLE_DIR / "00_data_overview.csv")
    print(f"\nsaved -> {TABLE_DIR/'00_data_overview.csv'}")


if __name__ == "__main__":
    main()
