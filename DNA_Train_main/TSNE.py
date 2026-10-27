from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def Tsne_cluster(Stre_cluster, pred_final, datasets, v):
    """
    绘制t-SNE聚类图：将文本标签更改为不同颜色的圆点（散点图）
    """
    # 1. 执行 t-SNE 降维
    print("正在进行 t-SNE 降维...")
    tsne = TSNE(n_components=2, random_state=42)
    fea_tsne = tsne.fit_transform(Stre_cluster)

    # 2. 准备数据
    # 确保 pred_final 是 numpy array，方便后续筛选
    pred_final = np.array(pred_final)
    unique_labels = sorted(set(pred_final))
    num_classes = len(unique_labels)

    # 3. 定义新的一组高对比度颜色 (SCI论文常用配色，对比更鲜明)
    # 顺序：红, 蓝, 绿, 紫, 橙, 黄, 棕, 粉, 灰, 青
    new_colors = [
        '#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00',
        '#FFFF33', '#A65628', '#F781BF', '#999999', '#17BECF'
    ]

    # 如果类别超过10个，自动扩展颜色
    if num_classes > len(new_colors):
        # 使用 matplotlib 的 tab20 色板补充
        extra_colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
        final_colors = extra_colors
    else:
        final_colors = new_colors[:num_classes]

    # 4. 开始绘图
    plt.figure(figsize=(8, 6))

    # 按类别循环绘制（这样可以自动生成图例）
    for i, label in enumerate(unique_labels):
        # 找到当前类别的所有点的索引
        idxs = (pred_final == label)

        plt.scatter(
            fea_tsne[idxs, 0],  # 当前类别的 x 坐标
            fea_tsne[idxs, 1],  # 当前类别的 y 坐标
            c=[final_colors[i]],  # 颜色
            label=f'Class {label}',  # 图例标签
            s=50,  # 点的大小 (圆饼大小)
            alpha=0.75,  # 透明度 (0-1)，防止重叠看不清
            edgecolors='white',  # 虽然是圆饼，加个白边会更有质感
            linewidth=0.5
        )

    # 5. 美化与保存
    # 移除坐标轴刻度，通常 t-SNE 图不需要具体数值
    plt.xticks([])
    plt.yticks([])

    # 添加图例 (位置在右边外侧，避免遮挡数据)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Categories")

    plt.title(f'T-SNE Visualization: {datasets} (View {v})')
    plt.tight_layout()

    # 保存图片
    save_name = f"{str(datasets)}_v={str(v)}_scatter.pdf"
    plt.savefig(save_name, format='pdf', bbox_inches='tight')
    print(f"图像已保存为: {save_name}")
    plt.show()


 