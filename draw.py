import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (5,3)


# # 准备数据
# Prompts = ['PromptEOL', 'CoT', 'KE']
# STSB = [72.69, 73.52, 68.95]
# STS12 = [63.70, 65.19, 63.53]
# STS13 = [74.14, 75.23, 71.27]
# STS14 = [68.87, 71.17, 69.06]
#
# # 设置柱状图的宽度
# width = 0.2
#
# # 计算每个柱状图的x轴位置
# x = np.arange(len(Prompts))
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width, STSB, width, label='STSB')
# rects2 = ax.bar(x - 0.001, STS12, width, label='STS12')
# rects3 = ax.bar(x + width, STS13, width, label='STS13')
# rects4 = ax.bar(x + width*2, STS14, width, label='STS14')
#
#
#
#
# ax.set_ylabel('分数', fontsize=12)
# ax.set_xlabel('使用的Prompt', fontsize=12)
# ax.set_title('不同Prompt在STS数据集上的pearson系数得分')
# ax.set_xticks(x)
# ax.set_xticklabels(Prompts)
# ax.legend()
#
#
# def autolabel(rects):
#     """在*rects*中的每个柱状条上方附加一个文本标签，显示其高度"""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3点垂直偏移
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)
#
# fig.tight_layout()
#
# plt.show()


# 准备数据
layers = ['last', '27', '25' , '23']
pearson = [72.69, 72.71, 68.11, 55.37]
spearman = [71.29, 71.40, 67.62, 55.37]

x = np.arange(len(layers))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, pearson, width, label='pearson')
rects2 = ax.bar(x + width / 2, spearman, width, label='spearman')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('分数')
ax.set_xlabel('抽取层')
ax.set_title('STSB集上抽取不同层嵌入')
ax.set_xticks(x)
ax.set_xticklabels(layers)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()