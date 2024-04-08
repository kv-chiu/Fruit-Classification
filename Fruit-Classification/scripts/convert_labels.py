import os
import json
import numpy as np

def convert_labels(label_files, output_dir, mapping_file):
    if isinstance(label_files, str):
        label_files = [label_files]

    label_dict = {}
    label_counter = {}

    for label_file in label_files:
        output_lines = []

        with open(label_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    file_path, label = line.split(' ')
                    if label not in label_dict:
                        label_dict[label] = len(label_dict) + 1
                        label_counter[label] = 1
                    else:
                        label_counter[label] += 1
                    label_number = label_dict[label]
                    output_line = f'{file_path} {label_number}\n'
                    output_lines.append(output_line)

        base_name = os.path.basename(label_file)
        output_file = os.path.join(output_dir, f'converted_{base_name}')
        with open(output_file, 'w') as file:
            file.writelines(output_lines)

    mapping_dict = {'label_dict': label_dict, 'label_counter': label_counter}
    mapping_output_file = os.path.join(output_dir, f'mapping_output.json')
    with open(mapping_output_file, 'w') as file:
        json.dump(mapping_dict, file, indent=4)
