import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from time import time

from jpeg_impl import jpeg_encoding, jpeg_decoding
from lbg import lbg_encoding, lbg_decoding

input_folder = "./images/"

if __name__ == '__main__':
    file_path = input_folder + "lena.bmp"
    print(file_path)

    img = np.array(Image.open(file_path).convert('RGB')).astype(int)
    plt.imshow(img)
    plt.show()
    img = img[:, :, 0]
    h, w = img.shape
    ori_bits = h * w * 8

    epsilon = 1e-2
    codeword_dim = (8, 8)
    codebook_size = 64
    start = time()
    lbg_bits, codebook = lbg_encoding(img=img, codeword_dim=codeword_dim, codebook_size=codebook_size,
                                      epsilon=epsilon)
    lbg_encoding_time = time() - start
    start = time()
    img_ = lbg_decoding(bits=lbg_bits, codebook=codebook)
    lbg_decoding_time = time() - start
    plt.imshow(np.dstack((img_, img_, img_)))
    plt.show()
    mse = np.mean(np.square(img - img_))
    psnr = 10 * np.log10(255 * 255 / mse)
    print("main() LBG encoding={:.2f}s, decoding={:.2f}s".format(lbg_encoding_time, lbg_decoding_time))
    print("main() LBG mse={:.2f}, psnr={:.2f}".format(mse, psnr))
    print("main() LBG ratio={:.2f} ({}/{})".format(len(lbg_bits) / ori_bits, len(lbg_bits), ori_bits))

    n = 8
    m = int(h / n)
    start = time()
    bits = jpeg_encoding(img=img, n=n)
    jpeg_encoding_time = time() - start
    start = time()
    img_ = jpeg_decoding(bits=bits, m=m, n=n)
    jpeg_decoding_time = time() - start
    plt.imshow(np.dstack((img_, img_, img_)))
    plt.show()
    mse = np.mean(np.square(img - img_))
    psnr = 10 * np.log10(255 * 255 / mse)
    print("main() JPEG encoding={:.2f}s, decoding={:.2f}s".format(jpeg_encoding_time, jpeg_decoding_time))
    print("main() JPEG mse={:.2f}, psnr={:.2f}".format(mse, psnr))
    print("main() JPEG ratio={:.2f} ({}/{})".format(len(bits) / ori_bits, len(bits), ori_bits))
