from generate_dataset import generate_dataset

data_prefix = "/root/autodl-tmp/data/fruit30"
output_dir = "/root/autodl-tmp/data/my_fruit30_dataset"
train_ratio = 0.7  # 训练集所占比例
val_ratio = 0.15  # 验证集所占比例
test_ratio = 0.15  # 测试集所占比例

generate_dataset(data_prefix, output_dir, train_ratio, val_ratio, test_ratio)
print(f"数据集已生成，\n保存在：{output_dir}\n训练集比例：{train_ratio}\n验证集比例：{val_ratio}\n测试集比例：{test_ratio}")
