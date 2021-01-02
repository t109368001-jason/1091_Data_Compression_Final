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


def jpeg_dpcm(quantization: np.ndarray, n: int) -> np.ndarray:
    h, w = quantization.shape
    temp = 0
    dpcm = np.zeros(shape=(int(h / n), int(w / n)))
    for i in range(0, h, n):
        for j in range(0, w, n):
            dpcm[int(i / n), int(j / n)] = quantization[i, j] - temp
            temp = quantization[i, j]
    return dpcm


def ijpeg_dpcm(dpcm: np.ndarray, n: int) -> np.ndarray:
    quantization_dc = np.zeros(shape=(dpcm.shape[0] * n, dpcm.shape[1] * n))
    h, w = quantization_dc.shape
    temp = 0
    for i in range(0, h, n):
        for j in range(0, w, n):
            quantization_dc[i, j] = dpcm[int(i / n), int(j / n)] + temp
            temp = quantization_dc[i, j]
    return quantization_dc


def jpeg_zigzag(quantization_ac: np.ndarray, n: int):
    h, w = quantization_ac.shape
    h_m, w_m = int(h / n), int(w / n)
    zigzag = np.zeros(shape=(h_m, w_m, n * n - 1))
    for i in range(0, h, n):
        for j in range(0, w, n):
            for ii in range(0, n):
                for jj in range(0, n):
                    if ii == 0 and jj == 0:
                        continue
                    zigzag[int(i / n), int(j / n), ii * n + jj - 1] = quantization_ac[i + ii, j + jj]
    return zigzag


def ijpeg_zigzag(zigzag: np.ndarray, n: int):
    h_m, w_m = zigzag.shape[0:2]
    h, w = h_m * n, w_m * n
    quantization_ac = np.zeros(shape=(h, w))
    for i in range(0, h, n):
        for j in range(0, w, n):
            for ii in range(0, n):
                for jj in range(0, n):
                    if ii == 0 and jj == 0:
                        continue
                    quantization_ac[i + ii, j + jj] = zigzag[int(i / n), int(j / n), ii * n + jj - 1]
    return quantization_ac


def jpeg_run_length(zigzag: np.ndarray, n: int) -> np.ndarray:
    h_m, w_m, length = zigzag.shape[0:3]
    run_length = np.zeros(shape=(h_m, w_m, length, 2))
    for i in range(h_m):
        for j in range(w_m):
            index = 0
            zero_count = 0
            for ll in range(length):
                if zigzag[i, j, ll] == 0:
                    zero_count += 1
                else:
                    run_length[i, j, index] = [zero_count, zigzag[i, j, ll]]
                    index += 1
                    zero_count = 0
    return run_length


def ijpeg_run_length(run_length: np.ndarray, n: int) -> np.ndarray:
    h_m, w_m, length = run_length.shape[0:3]
    zigzag = np.zeros(shape=(h_m, w_m, length))
    for i in range(h_m):
        for j in range(w_m):
            index = 0
            for ll in range(length):
                zero_count, value = run_length[i, j, ll, 0], run_length[i, j, ll, 1]
                if value == 0 and zero_count == 0:
                    break
                index = int(index + zero_count)
                zigzag[i, j, index] = value
                index += 1
    return zigzag


def jpeg_huffman(dpcm: np.ndarray, run_length: np.ndarray, n: int):
    h_m, w_m = dpcm.shape[0:2]
    bits = np.array([], dtype=int)
    for i in range(h_m):
        for j in range(w_m):
            dc_word = dpcm[i, j]
            ac_words = run_length[i, j]


def jpeg_encoding(img: np.ndarray, n: int = 8) -> (np.ndarray, np.ndarray):
    temp_img = np.copy(img)
    if len(temp_img.shape) == 3:
        gray = temp_img[:, :, 0]
    else:
        gray = temp_img
    level_offset = jpeg_level_offset(img=gray)
    dct = utils.dct(f=level_offset, n=n).astype(int)
    quantization = jpeg_quantization(dct=dct, n=n).astype(int)
    dpcm = jpeg_dpcm(quantization=quantization, n=n).astype(int)
    quantization_ac = np.copy(quantization)
    h, w = quantization_ac.shape
    for i in range(0, h, n):
        for j in range(0, w, n):
            quantization_ac[i, j] = 0
    zigzag = jpeg_zigzag(quantization_ac=quantization_ac, n=n)
    run_length = jpeg_run_length(zigzag=zigzag, n=n)
    # TODO
    return dpcm, run_length


def jpeg_decoding(seq: (np.ndarray, np.ndarray), n: int = 8) -> np.ndarray:
    # TODO
    dpcm, run_length = seq
    zigzag = ijpeg_run_length(run_length=run_length, n=n)
    quantization_ac = ijpeg_zigzag(zigzag=zigzag, n=n)
    quantization_dc = ijpeg_dpcm(dpcm=dpcm, n=n).astype(int)
    quantization = quantization_ac + quantization_dc
    dct = ijpeg_quantization(quantization=quantization, n=n).astype(int)
    level_offset = utils.idct(f=dct, n=n).astype(int)
    gray = ijpeg_level_offset(level_offset=level_offset)
    gray = gray.clip(0, 255)
    img = np.dstack((gray, gray, gray))
    return img
