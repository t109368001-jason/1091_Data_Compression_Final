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
        "is_gray": False
    }
    bitstream, img_, encoding_time, decoding_time = compression.perf(img=img, algorithm_name="JPEG", param=jpeg_param)
    # FIXME
    # compression.show("JPEG", img, img_, bitstream, encoding_time, decoding_time)
