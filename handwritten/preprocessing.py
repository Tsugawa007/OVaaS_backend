import numpy as np
import cv2
#from PIL import Image
#import base64
#import io


def to_pil_image(img_bin):
    _decoded = io.BytesIO(img_bin)
    return Image.open(_decoded)

####1106
def get_characters(char_list_path):
    with open(char_list_path, 'r', encoding='utf-8') as f:
        return ''.join(line.strip('\n') for line in f)

##This function change the type of picture into the type of Openvinomodel
def preprocess_input(image, height, width):
    src = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    ratio = float(src.shape[1]) / float(src.shape[0])
    tw = int(height * ratio)
    rsz = cv2.resize(src, (tw, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    # [h,w] -> [c,h,w]
    img = rsz[None, :, :]
    _, h, w = img.shape
    # right edge padding
    pad_img = np.pad(img, ((0, 0), (0, height - h), (0, width -  w)), mode='edge')
    return pad_img
####

