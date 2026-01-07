"""
得到的双星模式中每个星型结构为多跳
功能同rule_discovery_inter_doc/multi_hop_dual_graph/dual_star_discovery_multi_hop.py
在star_discovery_multi_hop_graph1.py和star_discovery_multi_hop_graph2.py后执行
生成的双星json保存在dual_star_json文件夹下，双星graph保存在dual_star_graph文件夹下
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 创建输出文件夹
os.makedirs("dual_star_json", exist_ok=True)
os.makedirs("dual_star_graph", exist_ok=True)


def load_star_structures_from_directory(directory):
    """加载指定目录下所有的星型结构文件"""
    star_structures = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                star_data = json.load(f)
                # 如果是列表形式，按中心节点分类
                if isinstance(star_data, list):
                    for star in star_data:
                        center = star["center"]
                        if center not in star_structures:
                            star_structures[center] = []
                        star_structures[center].append(star)
                else:
                    # 单个星型结构
                    center = star_data["center"]
                    star_structures[center] = [star_data]
    return star_structures


def load_star_images_from_directory(directory, centers):
    """加载指定目录下所有星型结构图像文件，并根据 center 检索"""
    star_images = {}
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # 假设图像文件是 PNG 格式
            file_path = os.path.join(directory, filename)
            for center in centers:
                # 替换 center 中的空格为下划线
                center_with_underscore = center.replace(" ", "_")
                # 检查图像文件名是否包含 center 的值
                if center_with_underscore in filename:
                    star_images[center] = file_path
                    break  # 找到匹配的图像后跳出循环
    return star_images


def combine_dual_star_structure(star1, star2):
    """将两个星型结构尾加，而不是合并属性"""
    return [star1, star2]


def save_dual_star_structure(dual_star, center_name):
    """保存双星结构到 dual_star_json 文件夹"""
    file_name = f"dual_star_json/{center_name.replace(' ', '_')}_dual_star_structure.json"
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(dual_star, f, ensure_ascii=False, indent=4)
    print(f"双星 JSON 文件已保存：{file_name}")


def combine_and_save_dual_stars(star_structures_1, star_structures_2, star_images_1, star_images_2):
    """匹配两个文件夹的星型结构和图像，生成双星并保存"""
    for center, stars_1 in star_structures_1.items():
        if center in star_structures_2:
            stars_2 = star_structures_2[center]

            # 假设每个文件夹下同中心的星型数量相同，且可以一一匹配
            for star1, star2 in zip(stars_1, stars_2):
                # 直接将第二个星型结构尾加到第一个星型结构后面
                dual_star = combine_dual_star_structure(star1, star2)

                # 保存双星结构为 JSON 文件
                save_dual_star_structure(dual_star, center)

                # 图像文件名根据中心节点的名称生成
                if center in star_images_1 and center in star_images_2:
                    image_path_1 = star_images_1[center]
                    image_path_2 = star_images_2[center]

                    # 创建 subplot 显示双星图像
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

                    # 加载并显示第一个图像
                    img1 = mpimg.imread(image_path_1)
                    ax[0].imshow(img1)
                    ax[0].set_title(f"Star 1: {center}")
                    ax[0].axis('off')  # 关闭坐标轴

                    # 加载并显示第二个图像
                    img2 = mpimg.imread(image_path_2)
                    ax[1].imshow(img2)
                    ax[1].set_title(f"Star 2: {center}")
                    ax[1].axis('off')  # 关闭坐标轴

                    # 保存组合后的图像
                    output_image = os.path.join("dual_star_graph", f"{center.replace(' ', '_')}_dual_star.png")
                    plt.savefig(output_image, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"双星图像已保存：{output_image}")
                else:
                    # 提示图像文件缺失
                    print(f"警告: 图像文件 {center} 不存在!")


def main():
    # 加载两个文件夹下的星型结构文件
    star_structures_1 = load_star_structures_from_directory("star_json_graph1")
    star_structures_2 = load_star_structures_from_directory("star_json_graph2")

    # 提取所有的 center 值
    centers = list(star_structures_1.keys())  # 假设两个文件夹中的中心节点名称是相同的

    # 加载两个文件夹下的图像文件
    star_images_1 = load_star_images_from_directory("star_graph1", centers)
    star_images_2 = load_star_images_from_directory("star_graph2", centers)

    # 合并并保存双星结构和图像
    combine_and_save_dual_stars(star_structures_1, star_structures_2, star_images_1, star_images_2)


if __name__ == "__main__":
    main()
