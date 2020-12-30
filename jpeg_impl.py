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


def jpeg_level_offset(img: np.ndarray) -> np.ndarray:
    level_offset = img - 128
    return level_offset


def ijpeg_level_offset(level_offset: np.ndarray) -> np.ndarray:
    img = level_offset + 128
    return img


def jpeg_quantization(dct: np.ndarray, n: int) -> np.ndarray:
    quantization = np.zeros(shape=dct.shape)
    h, w = quantization.shape
    for i in range(0, h, n):
        for j in range(0, w, n):
            quantization[i:i + n, j:j + n] = dct[i:i + n, j:j + n] / q
    return quantization


def ijpeg_quantization(quantization: np.ndarray, n: int) -> np.ndarray:
    dct = np.zeros(shape=quantization.shape)
    h, w = dct.shape
    for i in range(0, h, n):
        for j in range(0, w, n):
            dct[i:i + n, j:j + n] = quantization[i:i + n, j:j + n] * q
    return dct


def jpeg_encoding(img: np.ndarray, n: int = 8) -> np.ndarray:
    temp_img = np.copy(img)
    if len(temp_img.shape) == 3:
        gray = temp_img[:, :, 0]
    else:
        gray = temp_img
    level_offset = jpeg_level_offset(img=gray)
    dct = utils.dct(f=level_offset, n=n).astype(int)
    quantization = jpeg_quantization(dct=dct, n=n).astype(int)
    # TODO
    return quantization


def jpeg_decoding(quantization: np.ndarray, n: int = 8) -> np.ndarray:
    temp_quantization = np.copy(quantization)
    # TODO
    dct = ijpeg_quantization(quantization=temp_quantization, n=n).astype(int)
    level_offset = utils.idct(f=dct, n=n).astype(int)
    gray = ijpeg_level_offset(level_offset=level_offset)
    gray = gray.clip(0, 255)
    img = np.dstack((gray, gray, gray))
    return img
