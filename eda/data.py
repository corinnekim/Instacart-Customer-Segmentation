"""데이터 로딩과 장바구니 위치(cart position) 정규화 유틸."""

from pathlib import Path

import pandas as pd

# 프로젝트 루트 기준으로 데이터 경로 고정 (실행 위치와 무관하게 동작)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "instacart_full_data.parquet"


def load_orders(columns=None):
    """주문-상품 단위 parquet을 읽는다. columns 지정 시 해당 열만 로드해 메모리를 아낀다."""
    return pd.read_parquet(DATA_PATH, columns=columns)


def add_cart_position(df):
    """주문별 장바구니 크기와 정규화 위치(pos)를 붙인다.

    order_size = 한 주문(order_id)에 담긴 상품 수.
    pos = (add_to_cart_order - 1) / (order_size - 1), 0=맨 앞 ~ 1=맨 뒤.
    상품이 1개뿐인 주문은 위치 변화가 없으므로 pos=0으로 둔다.
    """
    order_size = df.groupby("order_id")["add_to_cart_order"].transform("size")
    df = df.assign(order_size=order_size)

    denom = (df["order_size"] - 1).where(df["order_size"] > 1, other=1)
    df["pos"] = (df["add_to_cart_order"] - 1) / denom
    df.loc[df["order_size"] == 1, "pos"] = 0.0
    return df
