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


def bitfield(n):
    return np.fromstring(np.binary_repr(n), dtype='S1').astype(int)


def ibitfield(b):
    return b.dot(1 << np.arange(b.size)[::-1])


def com_1s(bits: np.ndarray) -> np.ndarray:
    return np.array([1 if bit == 0 else 0 for bit in bits]).astype(int)


def jpeg_cat(value: int):
    cat = np.floor(np.log2(np.abs(value)))
    cat = 00 if cat == -np.inf else cat + 1
    return cat


def jpeg_dc_encoding(value: int) -> np.ndarray:
    cat = jpeg_cat(value)
    cat_bits = np.array(dc_cat_code_dict.get(cat))
    value_bits = bitfield(abs(value))
    if value < 0:
        value_bits = com_1s(value_bits)
    bits = np.append(cat_bits, value_bits).astype(int)
    return bits


def jpeg_dc_decoding(bits: np.ndarray) -> (int, np.ndarray):
    temp_bits = np.copy(bits)
    length = 2
    while (True):
        for key, code in dc_cat_code_dict.items():
            if len(code) == length:
                if list(temp_bits[0:length]) == code:
                    temp_bits = temp_bits[length:]
                    cat = key
                    if cat == 0:
                        return 0, temp_bits
                    else:
                        value_bits = temp_bits[0:cat]
                        temp_bits = temp_bits[cat:]
                        if value_bits[0] == 0:
                            value_bits = com_1s(value_bits)
                        value = ibitfield(value_bits)
                        return value, temp_bits
        length += 1
        if length > 11:
            raise Exception("error")


def jpeg_ac_encoding(ac_words: np.ndarray):
    bits = np.array([]).astype(int)
    length = ac_words.shape[0]
    for word in ac_words:
        zero_count, value = word[0], word[1]
        cat = jpeg_cat(value=value)


def jpeg_huffman(dpcm: np.ndarray, run_length: np.ndarray, n: int):
    h_m, w_m = dpcm.shape[0:2]
    bits = np.array([], dtype=int)
    for i in range(h_m):
        for j in range(w_m):
            dc_word = dpcm[i, j]
            ac_words = run_length[i, j]
            dc_bits = jpeg_dc_encoding(dc_word)
            ac_bits = jpeg_ac_encoding(ac_words)


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
    jpeg_huffman(dpcm=dpcm, run_length=run_length, n=n)
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


dc_cat_code_dict = {
    0: [0, 0],
    1: [0, 1, 0],
    2: [0, 1, 1],
    3: [1, 0, 0],
    4: [1, 0, 1],
    5: [1, 1, 0],
    6: [1, 1, 1, 0],
    7: [1, 1, 1, 1, 0],
    8: [1, 1, 1, 1, 1, 0],
    9: [1, 1, 1, 1, 1, 1, 0],
    10: [1, 1, 1, 1, 1, 1, 1, 0],
    11: [1, 1, 1, 1, 1, 1, 1, 1, 0],
}

ac_run_cat_code_dict = {
    "00": [1, 0, 1, 0],
    "01": [0, 0],
    "02": [0, 1],
    "03": [1, 0, 0],
    "04": [1, 0, 1, 1],
    "05": [1, 1, 0, 1, 0],
    "06": [1, 1, 1, 1, 0, 0, 0],
    "07": [1, 1, 1, 1, 1, 0, 0, 0],
    "08": [1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    "09": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    "0A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    "11": [1, 1, 0, 0],
    "12": [1, 1, 0, 1, 1],
    "13": [1, 1, 1, 1, 0, 0, 1],
    "14": [1, 1, 1, 1, 1, 0, 1, 1, 0],
    "15": [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    "16": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    "17": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    "18": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    "19": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    "1A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    "21": [1, 1, 1, 0, 0],
    "22": [1, 1, 1, 1, 1, 0, 0, 1],
    "23": [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    "24": [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
    "25": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    "26": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
    "27": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
    "28": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    "29": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    "2A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    "31": [1, 1, 1, 0, 1, 0],
    "32": [1, 1, 1, 1, 1, 0, 1, 1, 1],
    "33": [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    "34": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    "35": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    "36": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    "37": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    "38": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    "39": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
    "3A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
    "41": [1, 1, 1, 0, 1, 1],
    "42": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    "43": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    "44": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    "45": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    "46": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    "47": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    "48": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    "49": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    "4A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    "51": [1, 1, 1, 1, 0, 1, 0],
    "52": [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    "53": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
    "54": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
    "55": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    "56": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    "57": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    "58": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    "59": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0],
    "5A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0],
    "61": [1, 1, 1, 1, 0, 1, 1],
    "62": [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    "63": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
    "64": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
    "65": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0],
    "66": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0],
    "67": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    "68": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    "69": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    "6A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    "71": [1, 1, 1, 1, 1, 0, 1, 0],
    "72": [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    "73": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    "74": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    "75": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    "76": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    "77": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    "78": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    "79": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
    "7A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
    "81": [1, 1, 1, 1, 1, 1, 0, 0, 0],
    "82": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "83": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    "84": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    "85": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
    "86": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
    "87": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    "88": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    "89": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    "8A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    "91": [1, 1, 1, 1, 1, 1, 0, 0, 1],
    "92": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    "93": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    "94": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "95": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "96": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    "97": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    "98": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    "99": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    "9A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    "A1": [1, 1, 1, 1, 1, 1, 0, 1, 0],
    "A2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    "A3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    "A4": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    "A5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
    "A6": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
    "A7": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    "A8": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    "A9": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    "AA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    "B1": [1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    "B2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    "B3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    "B4": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    "B5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    "B6": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
    "B7": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
    "B8": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    "B9": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    "BA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
    "C1": [1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    "C2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
    "C3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
    "C4": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
    "C5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    "C6": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    "C7": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    "C8": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    "C9": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    "CA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    "D1": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    "D2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
    "D3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
    "D4": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
    "D5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
    "D6": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
    "D7": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
    "D8": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    "D9": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    "DA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    "E1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    "E2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
    "E3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
    "E4": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    "E5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    "E6": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    "E7": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    "E8": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
    "E9": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
    "EA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
    "F0": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    "F1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
    "F2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    "F3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    "F4": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    "F5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    "F6": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    "F7": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    "F8": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    "F9": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    "FA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
}
