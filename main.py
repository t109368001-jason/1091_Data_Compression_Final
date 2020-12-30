import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from jpeg_impl import jpeg_encoding, jpeg_decoding

input_folder = "./images/"

if __name__ == '__main__':
    file_path = input_folder + "lena.bmp"
    print(file_path)

    img = np.array(Image.open(file_path).convert('RGB')).astype(int)
    plt.imshow(img)
    plt.show()

    bits = jpeg_encoding(img)
    _img = jpeg_decoding(bits)
    plt.imshow(_img)
    plt.show()
    mse = np.mean(np.square(img - _img))
    print("main() mse={:.2f}".format(mse))
