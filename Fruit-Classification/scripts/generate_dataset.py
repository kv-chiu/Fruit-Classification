import os
import random
import shutil

def generate_dataset(data_prefix, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    train_file = os.path.join(output_dir, "train.txt")
    val_file = os.path.join(output_dir, "val.txt")
    test_file = os.path.join(output_dir, "test.txt")

    class_labels = sorted(os.listdir(data_prefix))

    for class_label in class_labels:
        class_path = os.path.join(data_prefix, class_label)

        file_list = []
        for root, dirs, files in os.walk(class_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                file_list.append(file_path)

        random.shuffle(file_list)  # 随机打乱文件列表

        num_files = len(file_list)
        num_train = int(num_files * train_ratio)
        num_val = int(num_files * val_ratio)
        num_test = num_files - num_train - num_val

        train_list = file_list[:num_train]
        val_list = file_list[num_train:num_train + num_val]
        test_list = file_list[num_train + num_val:]

        os.makedirs(os.path.join(train_dir, class_label), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_label), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_label), exist_ok=True)

        train_res_list = []
        val_res_list = []
        test_res_list = []

        for file_path in train_list:
            dst_path = os.path.join(train_dir, class_label, os.path.basename(file_path))
            train_res_list.append(dst_path)
            shutil.copy(file_path, dst_path)

        for file_path in val_list:
            dst_path = os.path.join(val_dir, class_label, os.path.basename(file_path))
            val_res_list.append(dst_path)
            shutil.copy(file_path, dst_path)

        for file_path in test_list:
            dst_path = os.path.join(test_dir, class_label, os.path.basename(file_path))
            test_res_list.append(dst_path)
            shutil.copy(file_path, dst_path)

        with open(train_file, "a") as train_f, open(val_file, "a") as val_f, open(test_file, "a") as test_f:
            for file_path in train_res_list:
                train_f.write(f"{file_path} {class_label}\n")
            for file_path in val_res_list:
                val_f.write(f"{file_path} {class_label}\n")
            for file_path in test_res_list:
                test_f.write(f"{file_path} {class_label}\n")
