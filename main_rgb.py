import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import compression

input_folder = "./images/"

if __name__ == '__main__':
    file_path = input_folder + "baboon.png"

    img = np.array(Image.open(file_path).convert('RGB')).astype(int)
    plt.imshow(img)
    plt.show()
    h, w, c = img.shape
    ori_bits = h * w * c * 8
    print("main() file_path={}, bits={}".format(file_path, ori_bits))

    # FIXME
    # lgb_param = {
    #     "codeword_dim": (8, 8),
    #     "codebook_size": 64,
    #     "epsilon": 1e-4
    # }
    # bitstream, img_, encoding_time, decoding_time = compression.perf(img=img, algorithm_name="LGB", param=lgb_param)
    # compression.show("LGB", img, img_, bitstream, encoding_time, decoding_time)
    n = 8
    jpeg_param = {
        "n": n,
        "m": (int(h / n)),
        "is_gray": False,
        "jab": (4, 4, 4)
    }
    algorithm_name = "JPEG"
    bitstream, img_, encoding_time, decoding_time = compression.perf(img=img, algorithm_name=algorithm_name,
                                                                     param=jpeg_param)
    # FIXME
    h, w, c = img.shape
    ori_bits = h * w * 8
    plt.imshow(img_)
    plt.show()
    mse = np.mean(np.square(img - img_))
    psnr = 10 * np.log10(255 * 255 / mse)
    print("main() {} encoding={:.2f}s, decoding={:.2f}s".format(algorithm_name, encoding_time, decoding_time))
    print("main() {} mse={:.2f}, psnr={:.2f}".format(algorithm_name, mse, psnr))
    print("main() {} ratio={:.2f} ({})".format(algorithm_name, len(bitstream) / ori_bits, len(bitstream)))
    # compression.show("JPEG", img, img_, bitstream, encoding_time, decoding_time)
