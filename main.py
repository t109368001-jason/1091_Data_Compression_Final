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

    h, w = img.shape[0:2]
    n = 8
    m = int(h / n)
    bits = jpeg_encoding(img=img, n=n)
    _img = jpeg_decoding(bits=bits, m=m, n=n)
    plt.imshow(_img)
    plt.show()
    mse = np.mean(np.square(img - _img))
    ori_bits = h * w * 8
    print("main() mse={:.2f}".format(mse))
    print("main() ratio={:.2f} ({}/{})".format(len(bits) / ori_bits, len(bits), ori_bits))
