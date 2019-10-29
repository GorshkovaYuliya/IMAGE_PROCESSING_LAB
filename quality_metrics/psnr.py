import pandas as pd
import cv2, os, math, numpy

import matplotlib.pyplot as plt

UPSCALED_PATH = "C:\\Users\\Yuliya_Harshkova\\PycharmProjects\\first_work\\sources\\output\\upscale\\"
ORIGINAL_PATH = "C:\\Users\\Yuliya_Harshkova\\PycharmProjects\\first_work\\sources\\input\\"
GRAPHYC_PATH = "C:\\Users\\Yuliya_Harshkova\\PycharmProjects\\first_work\\graphycs\\"
possible_scales = [2, 4, 6, 8, 32]


def get_upscaled_images(image_name, image_value):
    PSNR_total = []
    for file in os.listdir(UPSCALED_PATH):
        PSNR_buffer= []
        img = cv2.imread(UPSCALED_PATH + file)
        if file[:-26] == image_name:
            print("test")
            print(PSNR(image_value, img))
            PSNR_buffer.append(PSNR(image_value, img))
        PSNR_total.append(PSNR_buffer)
    return PSNR_total

def PSNR(original_image, upscale_image):
    value_of_diff = original_image.astype(float) - upscale_image.astype(float)
    flatten_value = value_of_diff.flatten('C')
    RMSE_value = math.sqrt(numpy.mean(flatten_value ** 2.))
    PSNR_value = 20 * math.log10(255. / RMSE_value)

    return PSNR_value


def process():

    for file in os.listdir(ORIGINAL_PATH):
        img = cv2.imread(ORIGINAL_PATH + file)
        print(get_upscaled_images(file, img))
        total = get_upscaled_images(file, img)
    PSNR_frame = pd.DataFrame(total, columns=[str(scale) for scale in possible_scales])

    plt.ylabel('PSNR');
    plt.xlabel('Scale')
    PSNR_boxplot = PSNR_frame.boxplot(grid=False)
    plt.savefig(os.path.join(GRAPHYC_PATH, "PSNR_image"))


if __name__ == "__main__":
    process()
