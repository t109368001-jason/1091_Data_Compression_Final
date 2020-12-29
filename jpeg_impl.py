import numpy as np

import utils

q = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]


def jpeg_encoding(img: np.ndarray, n: int = 8) -> np.ndarray:
    is_gray = utils.is_grey_scale(img)
    print("jpeg_encoding()", is_gray)
    if not is_gray:
        img = utils.rgb2ycbcr(img=img).astype(int)
    img = img - 128
    # TODO
    if is_gray:
        result = utils.dct(f=img, n=n).astype(int)
    else:
        y, cb, cr = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        y_result = utils.dct(f=y, n=n).astype(int)
        cb_result = utils.dct(f=cb, n=n).astype(int)
        cr_result = utils.dct(f=cr, n=n).astype(int)
        result = np.dstack((y_result, cb_result, cr_result))
    return result


def jpeg_decoding(bits: np.ndarray, n: int = 8) -> np.ndarray:
    is_gray = utils.is_grey_scale(bits)
    print("jpeg_decoding()", is_gray)
    if is_gray:
        result = utils.idct(f=bits, n=n).astype(int)
    else:
        y, cb, cr = bits[:, :, 0], bits[:, :, 1], bits[:, :, 2]
        y_result = utils.idct(f=y, n=n).astype(int)
        cb_result = utils.idct(f=cb, n=n).astype(int)
        cr_result = utils.idct(f=cr, n=n).astype(int)
        result = np.dstack((y_result, cb_result, cr_result))
    result = result + 128
    if not is_gray:
        result = utils.ycbcr2rgb(img=result).astype(int)
    result = result.clip(0, 255)
    return result
