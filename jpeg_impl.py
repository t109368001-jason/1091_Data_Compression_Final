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
    dpcm = np.zeros(shape=(int(h / n), int(w / n))).astype(int)
    for i in range(0, h, n):
        for j in range(0, w, n):
            dpcm[int(i / n), int(j / n)] = quantization[i, j] - temp
            temp = quantization[i, j]
    return dpcm


def ijpeg_dpcm(dpcm: np.ndarray, n: int) -> np.ndarray:
    quantization_dc = np.zeros(shape=(dpcm.shape[0] * n, dpcm.shape[1] * n)).astype(int)
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


def jpeg_run_length(zigzag: np.ndarray) -> np.ndarray:
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
                    if zero_count > 15:
                        run_length[i, j, index] = [15, 0]
                        index += 1
                        zero_count -= 16
                    run_length[i, j, index] = [zero_count, zigzag[i, j, ll]]
                    index += 1
                    zero_count = 0
    return run_length


def ijpeg_run_length(run_length: np.ndarray) -> np.ndarray:
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


def com_1s(bitstream: np.ndarray) -> np.ndarray:
    return np.array([1 if bit == 0 else 0 for bit in bitstream]).astype(int)


def jpeg_cat(value: int):
    cat = np.floor(np.log2(np.abs(value)))
    cat = 00 if cat == -np.inf else cat + 1
    return int(cat)


def jpeg_dc_encoding(value: int) -> np.ndarray:
    cat = jpeg_cat(value)
    cat_bitstream = np.array(dc_cat_code_dict.get(cat))
    bitstream = cat_bitstream
    if value != 0:
        value_bitstream = utils.bitfield(abs(value))
        if value < 0:
            value_bitstream = com_1s(value_bitstream)
        bitstream = np.append(bitstream, value_bitstream).astype(int)
    return bitstream


def jpeg_dc_decoding(bitstream: np.ndarray) -> (int, np.ndarray):
    temp_bitstream = np.copy(bitstream)
    length = 2
    while True:
        for key, code in dc_cat_code_dict.items():
            if len(code) == length:
                if list(temp_bitstream[0:length]) == code:
                    temp_bitstream = temp_bitstream[length:]
                    cat = key
                    if cat == 0:
                        return 0, temp_bitstream
                    else:
                        value_bitstream = temp_bitstream[0:cat]
                        temp_bitstream = temp_bitstream[cat:]
                        if value_bitstream[0] == 0:
                            value_bitstream = com_1s(value_bitstream)
                            value = 0 - utils.ibitfield(value_bitstream)
                        else:
                            value = utils.ibitfield(value_bitstream)
                        return value, temp_bitstream
        length += 1
        if length > 11:
            raise Exception("error")


def jpeg_ac_encoding(ac_words: np.ndarray):
    bitstream = np.array([]).astype(int)
    for word in ac_words:
        zero_count, value = word[0], word[1]
        cat = jpeg_cat(value=value)
        key = "{:X}{:X}".format(zero_count, cat)
        run_cat_bitstream = ac_run_cat_code_dict.get(key)
        bitstream = np.append(bitstream, run_cat_bitstream)
        if zero_count == 0 and value == 0:
            break
        if value != 0:
            value_bitstream = utils.bitfield(abs(value))
            if value < 0:
                value_bitstream = com_1s(value_bitstream)
            bitstream = np.append(bitstream, value_bitstream)
    return bitstream


def jpeg_ac_decoding(bitstream: np.ndarray, n: int):
    temp_bitstream = np.copy(bitstream)
    length = 1
    ac_words = np.zeros(shape=(n * n - 1, 2))
    index = 0
    while len(temp_bitstream) > 0:
        for key, code in ac_run_cat_code_dict.items():
            if len(code) == length:
                if list(temp_bitstream[0:length]) == code:
                    temp_bitstream = temp_bitstream[length:]
                    run_cat = key
                    zero_count, cat = int(run_cat[0], 16), int(run_cat[1], 16)
                    if cat == 0:
                        if zero_count == 0:
                            return ac_words, temp_bitstream
                        else:
                            ac_words[index] = [zero_count, 0]
                            index += 1
                    else:
                        value_bitstream = temp_bitstream[0:cat]
                        temp_bitstream = temp_bitstream[cat:]
                        if value_bitstream[0] == 0:
                            value_bitstream = com_1s(value_bitstream)
                            value = 0 - utils.ibitfield(value_bitstream)
                        else:
                            value = utils.ibitfield(value_bitstream)
                        ac_words[index] = [zero_count, value]
                        index += 1
                    length = 0
        length += 1
        if length > 16:
            raise Exception("error")
    return ac_words, temp_bitstream


def jpeg_huffman(dpcm: np.ndarray, run_length: np.ndarray, n: int):
    h_m, w_m = dpcm.shape[0:2]
    bitstream = np.array([], dtype=int)
    for i in range(h_m):
        for j in range(w_m):
            dc_word = dpcm[i, j]
            ac_words = run_length[i, j]
            dc_bitstream = jpeg_dc_encoding(dc_word)
            ac_bitstream = jpeg_ac_encoding(ac_words)
            bitstream = np.append(bitstream, dc_bitstream)
            bitstream = np.append(bitstream, ac_bitstream)
    return bitstream


def ijpeg_huffman(bitstream: np.ndarray, n: int, m: int):
    h_m, w_m = m, m
    dpcm = np.zeros(shape=(m, m))
    run_length = np.zeros(shape=(m, m, n * n - 1, 2))
    for i in range(h_m):
        for j in range(w_m):
            dc_word, bitstream = jpeg_dc_decoding(bitstream=bitstream)
            ac_words, bitstream = jpeg_ac_decoding(bitstream=bitstream, n=n)
            dpcm[i, j] = dc_word
            run_length[i, j] = ac_words
    return dpcm, run_length


def jpeg_encoding(img: np.ndarray, n: int = 8, **kwargs) -> (np.ndarray, np.ndarray):
    level_offset = jpeg_level_offset(img=img)
    dct = np.round(utils.dct(f=level_offset, n=n)).astype(int)
    quantization = np.round(jpeg_quantization(dct=dct, n=n)).astype(int)
    dpcm = jpeg_dpcm(quantization=quantization, n=n)
    zigzag = jpeg_zigzag(quantization_ac=quantization, n=n)
    run_length = jpeg_run_length(zigzag=zigzag).astype(int)
    bitstream = jpeg_huffman(dpcm=dpcm, run_length=run_length, n=n)
    # TODO
    return bitstream


def jpeg_decoding(bitstream: np.ndarray, m: int, n: int = 8) -> np.ndarray:
    # TODO
    dpcm, run_length = ijpeg_huffman(bitstream=bitstream, n=n, m=m)
    zigzag = ijpeg_run_length(run_length=run_length)
    quantization_ac = ijpeg_zigzag(zigzag=zigzag, n=n)
    quantization_dc = ijpeg_dpcm(dpcm=dpcm, n=n).astype(int)
    quantization = quantization_ac + quantization_dc
    dct = ijpeg_quantization(quantization=quantization, n=n).astype(int)
    level_offset = utils.idct(f=dct, n=n).astype(int)
    img = ijpeg_level_offset(level_offset=level_offset)
    img = img.clip(0, 255)
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
    "0A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    "11": [1, 1, 0, 0],
    "12": [1, 1, 0, 1, 1],
    "13": [1, 1, 1, 1, 0, 0, 1],
    "14": [1, 1, 1, 1, 1, 0, 1, 1, 0],
    "15": [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    "16": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    "17": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    "18": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    "19": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    "1A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    "21": [1, 1, 1, 0, 0],
    "22": [1, 1, 1, 1, 1, 0, 0, 1],
    "23": [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    "24": [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
    "25": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    "26": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
    "27": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    "28": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    "29": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    "2A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    "31": [1, 1, 1, 0, 1, 0],
    "32": [1, 1, 1, 1, 1, 0, 1, 1, 1],
    "33": [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    "34": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    "35": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    "36": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1],
    "37": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    "38": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
    "39": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
    "3A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    "41": [1, 1, 1, 0, 1, 1],
    "42": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    "43": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    "44": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1],
    "45": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    "46": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    "47": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    "48": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1],
    "49": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    "4A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1],
    "51": [1, 1, 1, 1, 0, 1, 0],
    "52": [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    "53": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
    "54": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
    "55": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
    "56": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
    "57": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    "58": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1],
    "59": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0],
    "5A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1],
    "61": [1, 1, 1, 1, 0, 1, 1],
    "62": [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    "63": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
    "64": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1],
    "65": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0],
    "66": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
    "67": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    "68": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    "69": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    "6A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    "71": [1, 1, 1, 1, 1, 0, 1, 0],
    "72": [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    "73": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    "74": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    "75": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    "76": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1],
    "77": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0],
    "78": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
    "79": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0],
    "7A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    "81": [1, 1, 1, 1, 1, 1, 0, 0, 0],
    "82": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "83": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    "84": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    "85": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
    "86": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    "87": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    "88": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
    "89": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    "8A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    "91": [1, 1, 1, 1, 1, 1, 0, 0, 1],
    "92": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    "93": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    "94": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "95": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
    "96": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    "97": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
    "98": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    "99": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    "9A": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    "A1": [1, 1, 1, 1, 1, 1, 0, 1, 0],
    "A2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
    "A3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    "A4": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
    "A5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
    "A6": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    "A7": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    "A8": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
    "A9": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    "AA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    "B1": [1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    "B2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    "B3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
    "B4": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0],
    "B5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
    "B6": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
    "B7": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    "B8": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    "B9": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
    "BA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
    "C1": [1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    "C2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1],
    "C3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0],
    "C4": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
    "C5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    "C6": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    "C7": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    "C8": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    "C9": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    "CA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
    "D1": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    "D2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
    "D3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
    "D4": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
    "D5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1],
    "D6": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
    "D7": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    "D8": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    "D9": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1],
    "DA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    "E1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    "E2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0],
    "E3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
    "E4": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    "E5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    "E6": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    "E7": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
    "E8": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
    "E9": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
    "EA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
    "F0": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    "F1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    "F2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    "F3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    "F4": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    "F5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    "F6": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    "F7": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    "F8": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    "F9": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    "FA": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
}
