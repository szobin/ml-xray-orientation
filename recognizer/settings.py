import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

MODEL_DIR = os.path.join(BASE_DIR, 'model')
IMAGE_DIR = os.path.join(BASE_DIR, 'images')

TRAIN_DIR = os.path.join(IMAGE_DIR, 'train')
VAL_DIR = os.path.join(IMAGE_DIR, 'val')

IMG_EXT = '.png'

IMG_WIDTH = 32 #128
IMG_HEIGHT = 32 # 128

TYPES = ({"name": "frontal view", "folder": 'f'},
         {"name": "sagittal view", "folder": 's'})
