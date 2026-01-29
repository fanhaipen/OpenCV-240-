# 2_神经网络模型.py
print("=== 第2步：理解神经网络模型 ===")
print("\n想象神经网络就像一个'魔法盒子'：")
print("输入 → [ 魔法盒子 ] → 输出")
print("(体重,身高) → [ 神经网络 ] → 猫(0)或狗(1)")

# 导入PyTorch
import torch
import torch.nn as nn

print("\n1. 创建最简单的神经网络（一层）")


class SimpleBrain(nn.Module):
    """最简单的神经网络，只有一层"""

    def __init__(self):
        super().__init__()
        # Linear层：2个输入（体重、身高）-> 1个输出（猫/狗）
        self.layer = nn.Linear(2, 1)  # 2个特征输入，1个输出

    def forward(self, x):
        # 前向传播：数据流过网络
        return self.layer(x)


# 创建模型实例
model = SimpleBrain()
print(f"\n模型结构：{model}")

# 查看模型的参数（权重）
print("\n模型的'知识'（权重）：")
for name, param in model.named_parameters():
    print(f"  {name}: {param.data}")
    print(f"  形状: {param.shape}")

# 2. 手动创建一个样本
print("\n2. 测试模型预测")
sample = torch.tensor([[3.5, 25.0]], dtype=torch.float32)  # 3.5kg, 25cm
print(f"输入样本: {sample}")  # 一只猫的特征

# 用模型预测
with torch.no_grad():  # 不计算梯度，只是预测
    prediction = model(sample)
    print(f"模型输出: {prediction.item():.4f}")

    # 解释输出
    if prediction.item() > 0.5:
        print("预测: 狗 (1)")
    else:
        print("预测: 猫 (0)")

print("\n🔍 注意：现在模型是随机初始化的，还没有学习，所以预测是随机的！")
print("   我们需要'训练'它，让它学会区分猫狗。")

# 3. 再看一个样本
sample2 = torch.tensor([[12.0, 45.0]], dtype=torch.float32)  # 12kg, 45cm
print(f"\n第二个样本: {sample2}")  # 一只狗的特征

with torch.no_grad():
    prediction2 = model(sample2)
    print(f"模型输出: {prediction2.item():.4f}")
    if prediction2.item() > 0.5:
        print("预测: 狗 (1)")
    else:
        print("预测: 猫 (0)")