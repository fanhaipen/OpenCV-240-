# day09_pytorch_basics.py
import torch
import numpy as np

print("🎯 第9天：PyTorch基础学习")
print("=" * 50)

# 1. 张量创建的不同方式
print("1. 张量创建")
print("-" * 30)

# 从列表创建
tensor_from_list = torch.tensor([1, 2, 3, 4])
print(f"从列表创建: {tensor_from_list}")

# 从NumPy数组创建
numpy_array = np.array([5, 6, 7, 8])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"从NumPy创建: {tensor_from_numpy}")

# 特殊张量
zeros_tensor = torch.zeros(2, 3)  # 2x3的全0张量
ones_tensor = torch.ones(2, 3)  # 2x3的全1张量
random_tensor = torch.randn(2, 3)  # 2x3的正态分布随机数
print(f"全0张量:\n{zeros_tensor}")
print(f"全1张量:\n{ones_tensor}")
print(f"随机张量:\n{random_tensor}")

# 2. 张量属性
print("\n2. 张量属性")
print("-" * 30)

tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(f"张量: {tensor}")
print(f"形状(shape): {tensor.shape}")
print(f"数据类型(dtype): {tensor.dtype}")
print(f"设备(device): {tensor.device}")
print(f"是否需要梯度(requires_grad): {tensor.requires_grad}")

# 3. 张量运算
print("\n3. 张量运算")
print("-" * 30)

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 基本运算
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")

# 矩阵运算
matrix_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
matrix_b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(f"\n矩阵A:\n{matrix_a}")
print(f"矩阵B:\n{matrix_b}")
print(f"矩阵乘法:\n{torch.matmul(matrix_a, matrix_b)}")

# 4. 自动求导
print("\n4. 自动求导")
print("-" * 30)

# 创建一个需要梯度的张量
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x}")

# 定义函数
y = 2 * x ** 2 + 3 * x + 1
print(f"y = 2*x^2 + 3*x + 1")

# 计算梯度
y.backward()
print(f"在x={x.item()}时，梯度dy/dx = {x.grad.item()}")

# 5. 改变形状
print("\n5. 改变张量形状")
print("-" * 30)

original = torch.arange(12)  # 0到11
print(f"原始张量: {original}")
print(f"原始形状: {original.shape}")

reshaped = original.view(3, 4)  # 改为3x4
print(f"改变形状后(3x4):\n{reshaped}")

flattened = reshaped.flatten()  # 展平
print(f"展平后: {flattened}")

# 6. 索引和切片
print("\n6. 索引和切片")
print("-" * 30)

tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"原始张量:\n{tensor}")
print(f"第一行: {tensor[0]}")
print(f"第一列: {tensor[:, 0]}")
print(f"子张量(1:3, 1:3):\n{tensor[1:3, 1:3]}")

# 7. GPU操作（如果可用）
print("\n7. GPU操作")
print("-" * 30)

if torch.cuda.is_available():
    print("GPU可用，正在测试GPU操作...")
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
    gpu_tensor = cpu_tensor.cuda()  # 移动到GPU
    print(f"CPU张量: {cpu_tensor} (设备: {cpu_tensor.device})")
    print(f"GPU张量: {gpu_tensor} (设备: {gpu_tensor.device})")

    # 移回CPU
    back_to_cpu = gpu_tensor.cpu()
    print(f"移回CPU: {back_to_cpu} (设备: {back_to_cpu.device})")
else:
    print("GPU不可用，使用CPU计算")

print("\n" + "=" * 50)
print("✅ PyTorch基础学习完成！")
print("=" * 50)