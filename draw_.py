import numpy as np
import matplotlib.pyplot as plt

# ======================= 数据 ======================= #
methods = ["PromptEOL", "Improved"]
tasks   = ["STSB", "STS12", "STS13", "STS14", "STS15"]

pearson = {
    "PromptEOL"  : [72.6, 63.70, 74.14, 68.87, 72.72],
    "Improved": [72.83, 64.97, 75.38, 70.31, 75.56],
}

spearman = {
    "PromptEOL"  : [71.29, 61.09, 72.54, 66.22, 71.79],
    "Improved": [71.31, 62.88, 72.85, 66.96, 75.71],
}

# 为了配色一致，可以手动指定颜色，也可使用 Matplotlib 默认循环
colors = ["#137cbc", "#1ca71c", "#8b5246", "#6c6c6c", "#1ab0c6"]

# ======================= 作图 ======================= #
plt.rcParams["font.family"] = "DejaVu Sans"           # 避免中文或 - 破折号乱码
plt.figure(figsize=(10, 12))

bar_width = 0.15
x0 = np.arange(len(methods))               # [0, 1] 两个柱组的中心 x 坐标

def add_subplot(pos, data, title, ylabel):
    """绘制单个子图，并在柱子上方标注数值"""
    ax = plt.subplot(pos)
    for i, task in enumerate(tasks):
        # 每个任务相对于组中心的位置：左负右正
        offset = (i - 2) * bar_width
        xs = x0 + offset
        ys = [data[m][i] for m in methods]
        bars = ax.bar(xs, ys, width=bar_width, label=task, color=colors[i])
        # 在柱顶写数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    height + 0.8,
                    f"{height:.1f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x0)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(title="STS Task", fontsize=9, title_fontsize=9, loc="upper left")

# Pearson 子图（上）
add_subplot(211,
            pearson,
            "",
            "Pearson")

# Spearman 子图（下）
add_subplot(212,
            spearman,
            "",
            "Spearman")

plt.tight_layout()
plt.show()
