import numpy as np
from PIL import Image

# KU.raw (720x560 image, each pixel is an 8-bit number)
# Gundam.raw (600x600 image, each pixel is an 8-bit number)
# Golf.raw (800x540 image, each pixel is an 8-bit number)

def convert_and_save_image(filename, shape):
    rawData = open(filename, 'rb').read()
    img = Image.frombytes('L', shape, rawData)
    img_name = filename.split('.raw')[0].replace('./', '')
    img.save("{}.bmp".format(img_name))

def load_image_asarray(filename, shape):
    rawData = open(filename, 'rb').read()
    img = Image.frombytes('L', shape, rawData)
    array = np.asarray(img)
    array = array.reshape((-1, 1))
    print(array.shape)
    return array


convert_and_save_image('./source/KU.raw', (720, 560))
convert_and_save_image('./source/Gundam.raw', (600, 600))
convert_and_save_image('./source/Golf.raw', (800, 540))