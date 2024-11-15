def morton_code(i, j):
    """计算二维坐标(i, j)的Morton码。"""
    def split_bits(n):
        """将整数n的二进制位展开并插入0位。"""
        n &= 0xFFFF  # 确保n为16位整数
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n
    return split_bits(i) | (split_bits(j) << 1)

def morton_sort_indices(grid_size):
    """根据Morton码对网格中的顶点进行排序。

    参数：
    - grid_size: 网格尺寸，表示网格为grid_size x grid_size

    返回：
    - sorted_indices: 按照Morton码排序的顶点索引列表
    """
    indices = []
    for i in range(grid_size):
        for j in range(grid_size):
            code = morton_code(i, j)
            index = i * grid_size + j  # 原始索引
            indices.append((code, index))
    # 根据Morton码排序
    indices.sort()
    # 提取排序后的索引
    sorted_indices = [index for (code, index) in indices]
    return sorted_indices

# 示例：对4x4网格进行排序
grid_size = 4
sorted_indices = morton_sort_indices(grid_size)
print("排序后的索引顺序：", sorted_indices)
