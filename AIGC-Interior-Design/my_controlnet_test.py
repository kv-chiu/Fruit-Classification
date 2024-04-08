import cv2
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()

cfg = Config.fromfile('/root/autodl-tmp/work/mmagic-main/configs/controlnet/controlnet-canny.py')
controlnet = MODELS.build(cfg.model).cuda()

prompt = 'A spacious room with blue walls and a vibrant yellow ceiling. The walls are painted in a calming shade of blue, creating a soothing atmosphere. The ceiling features a bold yellow color, adding a touch of energy and warmth to the space. The room is well-lit with large windows, allowing plenty of natural light to fill the room. It is furnished with comfortable seating and tasteful decor, creating an inviting and stylish ambiance. The combination of blue walls and a yellow ceiling creates a harmonious and visually striking color scheme. Overall, it is a delightful room that combines relaxation and vibrancy.'
control_url = '/root/autodl-tmp/data/Rough_housing.jpg'
control_img = mmcv.imread(control_url)
control = cv2.Canny(control_img, 100, 200)
control = control[:, :, None]
control = np.concatenate([control] * 3, axis=2)
control = Image.fromarray(control)

output_dict = controlnet.infer(prompt, control=control)
samples = output_dict['samples']
for idx, sample in enumerate(samples):
    sample.save(f'sample_{idx}.png')
controls = output_dict['controls']
for idx, control in enumerate(controls):
    control.save(f'control_{idx}.png')