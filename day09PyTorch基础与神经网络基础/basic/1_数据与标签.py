# 1_数据与标签.py
print("=== 第1步：理解数据和标签 ===")
print("\n想象一下，你要教电脑认识猫和狗...")

# 猫的特征：体重小(2-5kg)，身高矮(20-30cm)
# 狗的特征：体重大(5-20kg)，身高高(30-60cm)
print("\n1. 数据 = 动物的特征（体重、身高）")
print("2. 标签 = 答案（0=猫，1=狗）")

# 创建一些猫狗数据
cat_weight = 3.5  # 3.5kg
cat_height = 25   # 25cm
dog_weight = 12.0 # 12kg
dog_height = 45   # 45cm

print(f"\n猫的特征：体重={cat_weight}kg, 身高={cat_height}cm")
print(f"狗的特征：体重={dog_weight}kg, 身高={dog_height}cm")

# 用简单的规则判断
def is_dog(weight, height):
    """简单的判断规则"""
    if weight > 5 and height > 30:
        return "狗"
    else:
        return "猫"

print(f"\n用简单规则判断：")
print(f"  {cat_weight}kg, {cat_height}cm → {is_dog(cat_weight, cat_height)}")
print(f"  {dog_weight}kg, {dog_height}cm → {is_dog(dog_weight, dog_height)}")