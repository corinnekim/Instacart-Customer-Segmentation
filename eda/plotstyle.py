"""matplotlib 한글 폰트 설정. import 시 rcParams에 적용된다."""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
