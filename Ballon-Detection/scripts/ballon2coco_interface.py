from ballon2coco import convert_balloon_to_coco

ann_files = [
    '/root/autodl-tmp/data/balloon_dataset/balloon/train/via_region_data.json',
    '/root/autodl-tmp/data/balloon_dataset/balloon/val/via_region_data.json'
]
out_files = [
    '/root/autodl-tmp/data/balloon_dataset/balloon/coco_train.json',
    '/root/autodl-tmp/data/balloon_dataset/balloon/coco_val.json'
]
image_prefix = [
    '/root/autodl-tmp/data/balloon_dataset/balloon/train/',
    '/root/autodl-tmp/data/balloon_dataset/balloon/val/'
]

if __name__ == '__main__':
    for ann_file, out_file, prefix in zip(ann_files, out_files, image_prefix):
        convert_balloon_to_coco(
            ann_file=ann_file,
            out_file=out_file,
            image_prefix=prefix
            )
