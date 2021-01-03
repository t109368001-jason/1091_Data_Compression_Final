from time import time

import matplotlib.pyplot as plt
import numpy as np

from jpeg_impl import jpeg_encoding, jpeg_decoding
from lbg_impl import lbg_encoding, lbg_decoding

algorithms = {
    "LGB": {"encoding": lbg_encoding, "decoding": lbg_decoding},
    "JPEG": {"encoding": jpeg_encoding, "decoding": jpeg_decoding}
}


def perf(img, algorithm_name, param):
    algorithm = algorithms.get(algorithm_name)
    encoding_f = algorithm["encoding"]
    decoding_f = algorithm["decoding"]
    start = time()
    bitstream = encoding_f(img=img, param=param)
    encoding_time = time() - start
    start = time()
    img_ = decoding_f(bitstream=bitstream, param=param)
    decoding_time = time() - start
    return bitstream, img_, encoding_time, decoding_time


def show(algorithm_name, img, img_, bitstream, encoding_time, decoding_time):
    h, w = img.shape
    ori_bits = h * w * 8
    plt.imshow(np.dstack((img_, img_, img_)))
    plt.show()
    mse = np.mean(np.square(img - img_))
    psnr = 10 * np.log10(255 * 255 / mse)
    print("main() {} encoding={:.2f}s, decoding={:.2f}s".format(algorithm_name, encoding_time, decoding_time))
    print("main() {} mse={:.2f}, psnr={:.2f}".format(algorithm_name, mse, psnr))
    print("main() {} ratio={:.2f} ({})".format(algorithm_name, len(bitstream) / ori_bits, len(bitstream)))
