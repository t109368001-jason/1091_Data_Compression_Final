from time import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from jpeg_impl import jpeg_encoding, jpeg_decoding
from lbg_impl import lbg_encoding, lbg_decoding

input_folder = "./images/"

if __name__ == '__main__':
    file_path = input_folder + "lena.bmp"

    img = np.array(Image.open(file_path).convert('RGB')).astype(int)
    plt.imshow(img)
    plt.show()
    img = img[:, :, 0]
    h, w = img.shape
    ori_bits = h * w * 8
    print("main() file_path={}, bits={}".format(file_path, ori_bits))

    epsilon = 1e-2
    codeword_dim = (8, 8)
    codebook_size = 64
    start = time()
    lbg_bitstream, codebook = lbg_encoding(img=img, codeword_dim=codeword_dim, codebook_size=codebook_size,
                                           epsilon=epsilon)
    lbg_encoding_time = time() - start
    start = time()
    img_ = lbg_decoding(bitstream=lbg_bitstream, codebook=codebook)
    lbg_decoding_time = time() - start
    plt.imshow(np.dstack((img_, img_, img_)))
    plt.show()
    mse = np.mean(np.square(img - img_))
    psnr = 10 * np.log10(255 * 255 / mse)
    algorithm = "LBG"
    print("main() {} encoding={:.2f}s, decoding={:.2f}s".format(algorithm, lbg_encoding_time, lbg_decoding_time))
    print("main() {} mse={:.2f}, psnr={:.2f}".format(algorithm, mse, psnr))
    print("main() {} ratio={:.2f} ({})".format(algorithm, len(lbg_bitstream) / ori_bits, len(lbg_bitstream)))

    n = 8
    m = int(h / n)
    start = time()
    jpeg_bitstream = jpeg_encoding(img=img, n=n)
    jpeg_encoding_time = time() - start
    start = time()
    img_ = jpeg_decoding(bitstream=jpeg_bitstream, m=m, n=n)
    jpeg_decoding_time = time() - start
    plt.imshow(np.dstack((img_, img_, img_)))
    plt.show()
    mse = np.mean(np.square(img - img_))
    psnr = 10 * np.log10(255 * 255 / mse)
    algorithm = "JPEG"
    print("main() {} encoding={:.2f}s, decoding={:.2f}s".format(algorithm, jpeg_encoding_time, jpeg_decoding_time))
    print("main() {} mse={:.2f}, psnr={:.2f}".format(algorithm, mse, psnr))
    print("main() {} ratio={:.2f} ({})".format(algorithm, len(jpeg_bitstream) / ori_bits, len(jpeg_bitstream)))
