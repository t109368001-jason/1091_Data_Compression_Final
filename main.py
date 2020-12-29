import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from jpeg_impl import jpeg_encoding, jpeg_decoding

input_folder = "./images/"

if __name__ == '__main__':
    for filename in os.listdir(input_folder):
        file_path = input_folder + filename
        print(file_path)

        img = np.array(Image.open(file_path).convert('RGB'), dtype=int)
        plt.imshow(img)
        plt.show()
        bits = jpeg_encoding(img)
        _img = jpeg_decoding(bits)
        plt.imshow(_img)
        plt.show()
        mse = np.mean(np.square(img - _img))
        print("mse={:.2f}".format(mse))
        break
