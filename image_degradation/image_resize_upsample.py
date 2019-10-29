import glob
import os
import cv2
import random as r

RESIZE_FACTOR = [2, 4, 6, 8, 32]
EXTENSIONS = ('jpg', 'png')
LENGTH_OF_NAME = 5
PATH = "C:\\Users\\Yuliya_Harshkova\\PycharmProjects\\first_work\\sources\\input"
OUTPUT_PATH = "C:\\Users\\Yuliya_Harshkova\\PycharmProjects\\first_work\\sources\\output\\downscale"



def process():
    for file in os.listdir(PATH):
        #img = cv2.imread(PATH, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(PATH + file)
        #img = glob.glob(os.path.join(PATH, '*.' + extension))
        batch_resize(img, file)


def resize_single_img(img, factor, file):
    # find old and new image dimensions
   # assert not isinstance(img, type(None)), 'image not found'
    h, w, _ = img.shape
    new_height = int(h * factor)
    new_width = int(w * factor)

    # resize the image - down
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    # img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    # save the image
    print('Saving {}'.format(file))
    name_of_processed_image = file  + ".upscaled" + ".jpg"
    print(name_of_processed_image)
    cv2.imwrite(os.path.join(OUTPUT_PATH, name_of_processed_image), img)


def batch_resize(img, file):
    for i in RESIZE_FACTOR:
        resize_single_img(img, i, file)
