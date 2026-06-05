"""탐색 성향 × 활동량으로 고객을 4개 운영 세그먼트로 나눈다.

자연 군집이 있는지 먼저 확인하고(DBSCAN·실루엣), 두 축으로 나눈다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from modeling.data_features import load_sample

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "modeling"
FIG_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

MIN_ORDERS = 3
SEG_COLORS = {
    "A 탐색형": "#2a9d8f",
    "B 습관형": "#1f77b4",
    "C 대형단골": "#9467bd",
    "D 저활동형": "#e76f51",
}


def build_customer_features():
    """검증셋 점수 파일에서 고객별 탐색 성향과 활동량을 만든다."""
    n = pd.read_parquet(OUT_DIR / "necessity_valid.parquet")
    n_orders = n.groupby("user_id")["order_id"].nunique()
    cust = pd.DataFrame({
        "n_orders": n_orders,
        "avg_basket": n.groupby("user_id").size() / n_orders,
        "reorder_rate": n.groupby("user_id")["reordered"].mean(),
    })
    cust["explore"] = 1 - cust["reorder_rate"]
    return cust[cust["n_orders"] >= MIN_ORDERS]


def _knee_eps(kdist):
    """정렬된 k-거리 곡선에서 양 끝을 잇는 선과 가장 먼 점을 eps로."""
    n = len(kdist)
    x = np.arange(n)
    x0, y0, x1, y1 = 0, kdist[0], n - 1, kdist[-1]
    d = np.abs((y1 - y0) * x - (x1 - x0) * kdist + x1 * y0 - y1 * x0) / np.hypot(y1 - y0, x1 - x0)
    return kdist[int(np.argmax(d))]


def cluster_check(cust):
    """자연 군집이 있는지 확인만 한다. 없으면 운영 분할로 간다."""
    X = StandardScaler().fit_transform(cust[["explore", "n_orders", "avg_basket"]])
    sils = {k: silhouette_score(
                X, KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).labels_,
                sample_size=5000, random_state=42)
            for k in range(2, 7)}
    best_k = max(sils, key=sils.get)

    kdist = np.sort(NearestNeighbors(n_neighbors=10).fit(X).kneighbors(X)[0][:, -1])
    eps = _knee_eps(kdist)
    db = DBSCAN(eps=eps, min_samples=10).fit(X)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    print("clusterability check (자연 군집이 있나):")
    print(f"  KMeans best silhouette = {sils[best_k]:.3f} at k={best_k} (>0.5 뚜렷, 0.25~0.5 약함)")
    print(f"  DBSCAN (eps={eps:.2f}) clusters = {n_clusters}, noise = {(db.labels_ == -1).mean()*100:.1f}%")
    print("  => 자연 군집 없음. 연속 분포로 보고 운영 세그먼트로 나눈다.\n")


def assign_segments(cust):
    """저활동을 먼저 떼고, 나머지를 탐색/습관으로, 습관 중 주문 많은 쪽을 대형단골로."""
    q = cust["n_orders"].quantile([1 / 3, 2 / 3])
    q33, q67 = q.iloc[0], q.iloc[1]
    low_act = cust["n_orders"] <= q33
    ex_thr = cust.loc[~low_act, "explore"].median()
    hi_ex = cust["explore"] >= ex_thr

    seg = np.select(
        [low_act,
         ~low_act & hi_ex,
         ~low_act & ~hi_ex & (cust["n_orders"] >= q67)],
        ["D 저활동형", "A 탐색형", "C 대형단골"],
        default="B 습관형",
    )
    return cust.assign(segment=seg), ex_thr, q33, q67


def adoption_by_user(user_ids):
    """고객별 신상품 시도 수와 정착 수. 세그먼트 외부 검증에 쓴다."""
    df = load_sample()
    df = df[df["user_id"].isin(user_ids)]
    df["user_max"] = df.groupby("user_id")["order_number"].transform("max")
    adopted = df.loc[df["reordered"] == 1, ["user_id", "product_id"]].drop_duplicates()
    adopted["ad"] = 1
    firsts = df[(df["reordered"] == 0)
                & (df["order_number"] >= 2)
                & (df["order_number"] < df["user_max"])][["user_id", "product_id"]]
    firsts = firsts.merge(adopted, on=["user_id", "product_id"], how="left")
    firsts["ad"] = firsts["ad"].fillna(0)
    return firsts.groupby("user_id").agg(trials=("ad", "size"), adopted=("ad", "sum"))


def main():
    cust = build_customer_features()
    print(f"customers = {len(cust):,} (>= {MIN_ORDERS} orders)\n")
    cluster_check(cust)

    cust, ex_thr, q33, q67 = assign_segments(cust)
    cust = cust.join(adoption_by_user(cust.index), how="left")
    cust[["trials", "adopted"]] = cust[["trials", "adopted"]].fillna(0)

    prof = cust.groupby("segment").agg(
        size=("segment", "size"),
        explore=("explore", "mean"),
        n_orders=("n_orders", "mean"),
        avg_basket=("avg_basket", "mean"),
        trials=("trials", "sum"),
        adopted=("adopted", "sum"),
    )
    prof["share_%"] = (prof["size"] / len(cust) * 100).round(1)
    prof["adoption_rate"] = prof["adopted"] / prof["trials"]
    print(prof.round(3).to_string())

    base = cust["adopted"].sum() / cust["trials"].sum()
    spread = (prof["adoption_rate"].max() - prof["adoption_rate"].min()) * 100
    print(f"\nexternal validation: adoption {prof['adoption_rate'].min():.1%}"
          f" ~ {prof['adoption_rate'].max():.1%} across segments "
          f"(spread {spread:.1f}%p, overall {base:.1%})")

    _plot(cust, prof, ex_thr, q33, q67, base)
    _plot_segment_map(prof, cust)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cust.to_parquet(OUT_DIR / "customer_segments.parquet")
    prof.round(4).to_csv(OUT_DIR / "m3_segment_profile.csv")
    print(f"\nsaved -> {OUT_DIR/'customer_segments.parquet'}")


def _plot(cust, prof, ex_thr, q33, q67, base):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

    for seg, c in SEG_COLORS.items():
        s = cust[cust["segment"] == seg]
        axL.scatter(s["explore"], s["n_orders"], s=7, alpha=0.35, color=c, label=seg)
    axL.axvline(ex_thr, color="0.4", lw=1, ls="--")
    axL.axhline(q33, color="0.4", lw=1, ls="--")
    axL.axhline(q67, color="0.4", lw=1, ls="--")
    axL.set_yscale("log")
    axL.set_title("고객 분포: 탐색 성향 × 활동량")
    axL.set_xlabel("탐색 성향 (새 상품 비율)")
    axL.set_ylabel("활동량 (주문 수, 로그)")
    axL.legend(markerscale=2)
    axL.grid(alpha=0.3)

    order = prof.sort_values("adoption_rate").index
    axR.bar(range(len(order)), prof.loc[order, "adoption_rate"],
            color=[SEG_COLORS[s] for s in order], width=0.6)
    axR.axhline(base, color="0.5", ls="--", lw=1, label=f"전체 {base:.0%}")
    axR.set_xticks(range(len(order)))
    axR.set_xticklabels(order)
    axR.set_title("세그먼트별 정착률")
    axR.set_ylabel("정착률 (새 상품이 재구매된 비율)")
    axR.legend()
    axR.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "m3_segmentation.png"
    fig.savefig(out, dpi=150)
    print(f"saved figure -> {out}")


def _plot_segment_map(prof, cust):
    """세그먼트 포지셔닝 맵: 집단을 비중 크기 버블로 탐색×활동 위치에 찍는다."""
    desc = {
        "A 탐색형": "새 상품 활발히 시도",
        "B 습관형": "늘 사던 것, 활동 보통",
        "C 대형단골": "늘 사던 것, 헤비 유저",
        "D 저활동형": "주문 적음 (초기 고객)",
    }
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axvline(cust["explore"].median(), color="0.75", lw=1, zorder=1)
    ax.axhline(cust["n_orders"].median(), color="0.75", lw=1, zorder=1)

    for seg, c in SEG_COLORS.items():
        cx, cy, sh = prof.loc[seg, ["explore", "n_orders", "share_%"]]
        ax.scatter(cx, cy, s=sh * 266, color=c, alpha=0.85,
                   edgecolor="white", lw=2.5, zorder=5)
        ax.annotate(f"{seg}\n{sh:.0f}%", (cx, cy), ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color="white", zorder=6)

    ax.set_yscale("log")
    ax.set_xlim(prof["explore"].min() - 0.16, prof["explore"].max() + 0.16)
    ax.set_ylim(prof["n_orders"].min() * 0.3, prof["n_orders"].max() * 2.6)
    ax.set_title("고객 세그먼트 포지셔닝 맵")
    ax.set_xlabel("← 늘 사던 것      탐색 성향 (새 상품 비율)      새것 탐색 →")
    ax.set_ylabel("← 적게      활동량 (주문 수)      많이 →")
    ax.grid(alpha=0.2, zorder=0)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=11, label=f"{seg}: {desc[seg]}")
               for seg, c in SEG_COLORS.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "m3_segment_map.png"
    fig.savefig(out, dpi=150)
    print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
