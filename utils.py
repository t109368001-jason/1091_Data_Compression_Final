from time import time

import numpy as np


def is_grey_scale(img: np.ndarray) -> bool:
    if len(img.shape) == 2:
        return True
    h, w, c = img.shape
    if c == 1:
        return True
    for i in range(w):
        for j in range(h):
            r, g, b = img[i, j]
            if r != g != b:
                return False
    return True


rgb2ycbcr_t = np.array([[0.299, -0.168636, 0.499813],
                        [0.587, -0.331068, - 0.418531],
                        [0.114, 0.499704, - 0.081282]])


def rgb2ycbcr(img: np.ndarray, delta: int = 128) -> np.ndarray:
    start = time()
    t2 = np.array([0, delta, delta])
    result = np.dot(img, rgb2ycbcr_t) + t2
    print("rgb2ycbcr()", time() - start)
    return result


def ycbcr2rgb(img: np.ndarray, delta: int = 128) -> np.ndarray:
    start = time()
    it = np.linalg.inv(rgb2ycbcr_t)
    t2 = np.array([0, delta, delta])
    a = (img - t2)
    result = np.dot(a, it)
    print("ycbcr2rgb()", time() - start)
    return result


b_cache = dict()


def get_b(n: int) -> np.ndarray:
    if b_cache.get(n) is not None:
        return np.copy(b_cache[n])
    a = np.array([1] + [2] * (n - 1))
    a = np.sqrt(a / n)
    cos_coe = np.dot((2 * np.arange(n).reshape(n, 1) + 1), np.arange(n).reshape(1, n))
    cos_coe_radius = cos_coe * np.pi / 2 / n
    b = np.cos(cos_coe_radius)
    for i in range(n):
        b[i, :] = a * b[i, :]
    b_cache[n] = np.copy(b)
    return b


def dct(f: np.ndarray, n: int) -> np.ndarray:
    result = np.zeros(shape=f.shape)
    h, w = f.shape
    b = get_b(n)
    for i in range(0, h, n):
        for j in range(0, w, n):
            result[i:i + n, j:j + n] = np.dot(np.dot(np.transpose(b), f[i:i + 8, j:j + 8]), b)
    return result


def idct(f: np.ndarray, n: int) -> np.ndarray:
    result = np.zeros(shape=f.shape)
    h, w = f.shape
    b = get_b(n)
    for i in range(0, h, n):
        for j in range(0, w, n):
            result[i:i + n, j:j + n] = np.dot(np.dot(b, f[i:i + 8, j:j + 8]), np.transpose(b))
    return result


def bitfield(n: int, length: int = None) -> np.ndarray:
    result = np.fromstring(np.binary_repr(n), dtype='S1').astype(int)
    if length is not None:
        if len(result) < length:
            result = np.append(np.zeros(shape=(length - len(result))), result)
    return result


def img2block(img: np.ndarray, block_shape: tuple):
    h, w = img.shape[0:2]
    h_m, w_m = int(h / block_shape[0]), int(w / block_shape[1])
    image_block = np.copy(img).reshape((h, w_m, block_shape[1]))
    image_block = image_block.reshape((h_m, block_shape[0], w_m, block_shape[1]))
    image_block = image_block.swapaxes(1, 2)
    return image_block


def block2img(block: np.ndarray):
    block_shape = block.shape[2:4]
    h_m, w_m = block.shape[0:2]
    h, w = h_m * block_shape[0], w_m * block_shape[1]
    img = np.copy(block)
    img = img.swapaxes(1, 2)
    img = img.reshape((h, w_m, block_shape[1]))
    img = img.reshape((h, w))
    return img


def ibitfield(b: np.ndarray, length: int = None) -> int:
    temp_b = np.copy(b)
    if length is not None:
        if len(temp_b) > length:
            temp_b = temp_b[0:length]
    return int(temp_b.dot(1 << np.arange(temp_b.size)[::-1]))
