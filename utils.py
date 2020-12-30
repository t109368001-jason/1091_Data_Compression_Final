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


def rgb2ycbcr(img: np.ndarray, delta: int = 128) -> np.ndarray:
    h, w, c = img.shape
    result = np.zeros(shape=img.shape)
    for i in range(h):
        for j in range(w):
            p = img[i, j]
            r, g, b = p[0], p[1], p[2]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cb = 0.564 * (b - y) + delta
            cr = 0.713 * (r - y) + delta
            result[i, j] = [y, cb, cr]
    return result


def ycbcr2rgb(img: np.ndarray, delta: int = 128) -> np.ndarray:
    h, w, c = img.shape
    result = np.zeros(shape=img.shape)
    for i in range(h):
        for j in range(w):
            p = img[i, j]
            y, cb, cr = p[0], p[1], p[2]
            r = y + 1.403 * (cr - delta)
            g = y - 0.714 * (cr - delta) - 0.344 * (cb - delta)
            b = y + 1.773 * (cb - delta)
            result[i, j] = [r, g, b]
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
