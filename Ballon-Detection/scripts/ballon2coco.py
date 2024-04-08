import os
import json
from PIL import Image
from tqdm import tqdm

def convert_balloon_to_coco(ann_file, out_file, image_prefix):
    with open(ann_file, 'r') as f:
        data_infos = json.load(f)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in tqdm(enumerate(data_infos.values()), desc='Converting'):
        filename = v['filename']
        img_path = os.path.join(image_prefix, filename)
        img = Image.open(img_path)
        width, height = img.size

        images.append(
            {
                'id': idx,
                'file_name': filename,
                'height': height,
                'width': width
            }
        )

        for _, obj in v['regions'].items():
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = min(px), min(py), max(px), max(py)

            data_anno = {
                'image_id': idx,
                'id': obj_count,
                'category_id': 0,
                'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                'area': (x_max - x_min) * (y_max - y_min),
                'segmentation': [poly],
                'iscrowd': 0
            }
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = {
        'images': images,
        'annotations': annotations,
        'categories': [{
            'id': 0,
            'name': 'balloon'
        }]
    }

    with open(out_file, 'w') as f:
        json.dump(coco_format_json, f)
