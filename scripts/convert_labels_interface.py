from convert_labels import convert_labels

label_files = [
    '/root/autodl-tmp/data/my_fruit30_dataset/test.txt',
    '/root/autodl-tmp/data/my_fruit30_dataset/train.txt',
    '/root/autodl-tmp/data/my_fruit30_dataset/val.txt'
]
output_dir = '/root/autodl-tmp/data/my_fruit30_dataset/converted'
mapping_file = '/root/autodl-tmp/data/my_fruit30_dataset/label_mapping.json'

convert_labels(label_files, output_dir, mapping_file)
print(f"转换完成，\n转换后文件保存在{output_dir}\n映射表保存在{mapping_file}")
