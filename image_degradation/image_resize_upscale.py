import glob
import os
import cv2
import random as r

RESIZE_FACTOR = [2, 4, 6, 8, 32]
EXTENSIONS = ('jpg', 'png')
LENGTH_OF_NAME = 5
PATH = "C:\\Users\\Yuliya_Harshkova\\PycharmProjects\\first_work\\sources\\output\\downscale\\"
OUTPUT_PATH = "C:\\Users\\Yuliya_Harshkova\\PycharmProjects\\first_work\\sources\\output\\upscale\\"


def generate_random_string():
    random_string = ''
    random_str_seq = "qwertyuiopasdfghjklzxcvbnm"
    for i in range(0, LENGTH_OF_NAME):
        if i % LENGTH_OF_NAME == 0 and i != 0:
            random_string += '-'
        random_string += str(random_str_seq[r.randint(0, len(random_str_seq) - 1)])
    return random_string


def upscale():
    for file in os.listdir(PATH):
        # img = cv2.imread(PATH, cv2.IMREAD_UNCHANGED)
        img = cv2.imread(PATH + file)
        # img = glob.glob(os.path.join(PATH, '*.' + extension))
        batch_resize(img, file)


def resize_single_img(img, factor, file):
    # find old and new image dimensions
    # assert not isinstance(img, type(None)), 'image not found'
    print
    h, w, _ = img.shape
    new_height = int(h * factor[0])
    new_width = int(w * factor[0])

    # resize the image - down
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    # img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    # save the image
    print('Saving {}'.format(file))
    name_of_processed_image = file[:-4] + "_upscale_" + ".jpg"
    print(name_of_processed_image)
    cv2.imwrite(os.path.join(OUTPUT_PATH, name_of_processed_image), img)


def batch_resize(img, file):
    resize_single_img(img, get_resize_format(file), file)


def get_resize_format(file):
    file = file[:-4]
    return [int(s) for s in file.split('_') if s.isdigit()]
