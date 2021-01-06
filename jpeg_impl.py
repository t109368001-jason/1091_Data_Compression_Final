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

q_c = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
]

index_to_zigzag_index_table = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
]
index_to_zigzag_index_table_63 = np.array(index_to_zigzag_index_table[1:]) - 1

zigzag_index_to_index_table = [
    00, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
]
zigzag_index_to_index_table_63 = np.array(zigzag_index_to_index_table[1:]) - 1


def jpeg_check_jab(j: int, a: int, b: int):
    if j < 2:
        raise Exception("error")
    if a > j or b > j:
        raise Exception("error")
    if a != b and b != 0:
        raise Exception("error")


def jpeg_down_sampling(img: np.ndarray, j: int, a: int, b: int) -> np.ndarray:
    jpeg_check_jab(j, a, b)
    h, w = img.shape[0:2]
    h_m = 2 if b == 0 else 1
    w_m = int(j / a)
    down_sampling = img
    if h_m != 1:
        down_sampling = down_sampling[0:h:h_m, ::]
    if w_m != 1:
        down_sampling = down_sampling[::, 0:w:w_m]
    return down_sampling


def jpeg_up_sampling(down_sampling: np.ndarray, j: int, a: int, b: int) -> np.ndarray:
    jpeg_check_jab(j, a, b)
    h, w = down_sampling.shape[0:2]
    h_m = 2 if b == 0 else 1
    w_m = int(j / a)
    h_index = np.array([[ii] * h_m for ii in range(0, h)]).flatten()
    w_index = np.array([[jj] * w_m for jj in range(0, w)]).flatten()
    img = down_sampling[h_index][::, w_index]
    return img


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


def jpeg_zigzag(quantization_ac: np.ndarray, n: int) -> np.ndarray:
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


def ijpeg_zigzag(zigzag: np.ndarray, n: int) -> np.ndarray:
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
                    while zero_count > 15:
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


def jpeg_run_length_block(zigzag: np.ndarray) -> np.ndarray:
    length = zigzag.shape[0]
    run_length = np.zeros(shape=(length, 2)).astype(zigzag.dtype)
    index = 0
    zero_count = 0
    for ll in range(length):
        if zigzag[ll] == 0:
            zero_count += 1
        else:
            while zero_count > 15:
                run_length[index] = [15, 0]
                index += 1
                zero_count -= 16
            run_length[index] = [zero_count, zigzag[ll]]
            index += 1
            zero_count = 0
    return run_length


def ijpeg_run_length_block(run_length: np.ndarray) -> np.ndarray:
    length = run_length.shape[0]
    zigzag = np.zeros(shape=length)
    index = 0
    for ll in range(length):
        zero_count, value = run_length[ll, 0], run_length[ll, 1]
        if value == 0 and zero_count == 0:
            break
        index = int(index + zero_count)
        zigzag[index] = value
        index += 1
    return zigzag


def com_1s(bitstream: np.ndarray) -> np.ndarray:
    return np.array([1 if bit == 0 else 0 for bit in bitstream]).astype(int)


def jpeg_cat(value: int) -> int:
    if value == 0:
        return 0
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
        key = utils.ibitfield(temp_bitstream[0:length])
        cat = dc_code_cat_dict.get(key + (1 << length))
        if cat is not None:
            temp_bitstream = temp_bitstream[length:]
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


def jpeg_ac_encoding(ac_words: np.ndarray) -> np.ndarray:
    bitstream = np.array([]).astype(int)
    for word in ac_words:
        zero_count, value = word[0], word[1]
        cat = jpeg_cat(value=value)
        key = "{:X}{:X}".format(zero_count, cat)
        run_cat_bitstream = ac_run_cat_code_dict.get(key)
        if run_cat_bitstream is None:
            raise Exception("error")
        bitstream = np.append(bitstream, run_cat_bitstream)
        if zero_count == 0 and value == 0:
            break
        if value != 0:
            value_bitstream = utils.bitfield(abs(value))
            if value < 0:
                value_bitstream = com_1s(value_bitstream)
            bitstream = np.append(bitstream, value_bitstream)
    return bitstream


def jpeg_ac_decoding(bitstream: np.ndarray, n: int) -> (np.ndarray, np.ndarray):
    temp_bitstream = np.copy(bitstream)
    length = 2
    ac_words = np.zeros(shape=(n * n - 1, 2))
    index = 0
    while len(temp_bitstream) > 0:
        key = utils.ibitfield(temp_bitstream[0:length])
        run_cat = ac_code_run_cat_dict.get(key)
        if run_cat is not None:
            temp_bitstream = temp_bitstream[length:]
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
            length = 2
            continue
        length += 1
        if length > 16:
            raise Exception("error")
    return ac_words, temp_bitstream


def jpeg_huffman(dpcm: np.ndarray, run_length: np.ndarray) -> np.ndarray:
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


def ijpeg_huffman(bitstream: np.ndarray, n: int, m: int) -> (np.ndarray, np.ndarray):
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


def jpeg_dct(y_block: np.ndarray) -> np.ndarray:
    return np.round(utils.dct_block(y_block)).astype(int)


def ijpeg_dct(y_dct):
    return np.round(utils.idct_block(y_dct)).astype(int)


def jpeg_quant_y(y_dct: np.ndarray) -> np.ndarray:
    return np.round(y_dct / q).astype(y_dct.dtype)


def ijpeg_quant_y(y_quant: np.ndarray) -> np.ndarray:
    return (y_quant * q).astype(y_quant.dtype)


def jpeg_quant_cbcr(cb_dct: np.ndarray) -> np.ndarray:
    return np.round(cb_dct / q_c).astype(cb_dct.dtype)


def ijpeg_quant_cbcr(cb_quant: np.ndarray) -> np.ndarray:
    return (cb_quant * q_c).astype(cb_quant.dtype)


def jpeg_dc_ac(block: np.ndarray) -> (np.ndarray, np.ndarray):
    flatten = block.flatten()
    return flatten[0], flatten[1:]


def ijpeg_dc_ac(dc: np.ndarray, ac: np.ndarray) -> np.ndarray:
    flatten = np.append(dc, ac)
    n = int(np.sqrt(len(flatten)))
    return flatten.reshape(n, n)


def jpeg_zigzag_block(y_ac):
    return y_ac[index_to_zigzag_index_table_63]


def ijpeg_zigazg_block(y_zigzag):
    return y_zigzag[zigzag_index_to_index_table_63]


def jpeg_encoding(img: np.ndarray, param: dict) -> (np.ndarray, np.ndarray):
    n = param["n"]
    is_gray = param["is_gray"]
    if is_gray:
        level_offset = jpeg_level_offset(img=img)
        dct = np.round(utils.dct(f=level_offset, n=n)).astype(int)
        quantization = np.round(jpeg_quantization(dct=dct, n=n)).astype(int)
        dpcm = jpeg_dpcm(quantization=quantization, n=n)
        zigzag = jpeg_zigzag(quantization_ac=quantization, n=n)
        run_length = jpeg_run_length(zigzag=zigzag).astype(int)
        bitstream = jpeg_huffman(dpcm=dpcm, run_length=run_length)
        # TODO
        return bitstream
    else:
        ycbcr = np.round(utils.rgb2ycbcr(img=img))
        ycbcr = ycbcr - 128
        y, cb, cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
        j, a, b = param["jab"]
        cb = jpeg_down_sampling(cb, j, a, b)
        cr = jpeg_down_sampling(cr, j, a, b)
        cb = jpeg_up_sampling(cb, j, a, b)
        cr = jpeg_up_sampling(cr, j, a, b)
        y_block = utils.img2block(img=y, block_shape=(n, n))
        cb_block = utils.img2block(img=cb, block_shape=(n, n))
        cr_block = utils.img2block(img=cr, block_shape=(n, n))
        y_dc_last: int = 0
        cb_dc_last: int = 0
        cr_dc_last: int = 0
        bitstream = np.array([]).astype(int)
        for ii in range(y_block.shape[0]):
            for jj in range(y_block.shape[1]):
                y_dct = jpeg_dct(y_block[ii, jj])
                cb_dct = jpeg_dct(cb_block[ii, jj])
                cr_dct = jpeg_dct(cr_block[ii, jj])
                y_quant = jpeg_quant_y(y_dct)
                cb_quant = jpeg_quant_cbcr(cb_dct)
                cr_quant = jpeg_quant_cbcr(cr_dct)
                y_dc, y_ac = jpeg_dc_ac(y_quant)
                y_zigzag = jpeg_zigzag_block(y_ac)
                cb_dc, cb_ac = jpeg_dc_ac(cb_quant)
                cb_zigzag = jpeg_zigzag_block(cb_ac)
                cr_dc, cr_ac = jpeg_dc_ac(cr_quant)
                cr_zigzag = jpeg_zigzag_block(cr_ac)
                y_run_length = jpeg_run_length_block(y_zigzag)
                cb_run_length = jpeg_run_length_block(cb_zigzag)
                cr_run_length = jpeg_run_length_block(cr_zigzag)
                y_dpcm = y_dc - y_dc_last
                cb_dpcm = cb_dc - cb_dc_last
                cr_dpcm = cr_dc - cr_dc_last
                y_dc_last = y_dc
                cb_dc_last = cb_dc
                cr_dc_last = cr_dc
                y_dpcm_bits = jpeg_dc_encoding(y_dpcm)
                y_ac_bits = jpeg_ac_encoding(y_run_length)
                cb_dpcm_bits = jpeg_dc_encoding(cb_dpcm)
                cb_ac_bits = jpeg_ac_encoding(cb_run_length)
                cr_dpcm_bits = jpeg_dc_encoding(cr_dpcm)
                cr_ac_bits = jpeg_ac_encoding(cr_run_length)
                bitstream = np.append(bitstream, y_dpcm_bits)
                bitstream = np.append(bitstream, y_ac_bits)
                bitstream = np.append(bitstream, cb_dpcm_bits)
                bitstream = np.append(bitstream, cb_ac_bits)
                bitstream = np.append(bitstream, cr_dpcm_bits)
                bitstream = np.append(bitstream, cr_ac_bits)
        return bitstream


def jpeg_decoding(bitstream: np.ndarray, param: dict) -> np.ndarray:
    # TODO
    n = param["n"]
    m = param["m"]
    is_gray = param["is_gray"]
    if is_gray:
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
    else:
        h, w = param["resolution"]
        h_m, w_m = int(h / n), int(w / n)
        bitstream = bitstream
        y_block = np.zeros(shape=(h_m, w_m, n, n))
        cb_block = np.zeros(shape=(h_m, w_m, n, n))
        cr_block = np.zeros(shape=(h_m, w_m, n, n))
        y_dc_last = 0
        cb_dc_last = 0
        cr_dc_last = 0
        for ii in range(h_m):
            for jj in range(w_m):
                y_dpcm, bitstream = jpeg_dc_decoding(bitstream)
                y_run_length, bitstream = jpeg_ac_decoding(bitstream, n)
                cb_dpcm, bitstream = jpeg_dc_decoding(bitstream)
                cb_run_length, bitstream = jpeg_ac_decoding(bitstream, n)
                cr_dpcm, bitstream = jpeg_dc_decoding(bitstream)
                cr_run_length, bitstream = jpeg_ac_decoding(bitstream, n)
                y_zigzag = ijpeg_run_length_block(y_run_length)
                cb_zigzag = ijpeg_run_length_block(cb_run_length)
                cr_zigzag = ijpeg_run_length_block(cr_run_length)
                y_ac = ijpeg_zigazg_block(y_zigzag)
                y_dc = y_dpcm + y_dc_last
                y_quant = ijpeg_dc_ac(y_dc, y_ac)
                cb_ac = ijpeg_zigazg_block(cb_zigzag)
                cb_dc = cb_dpcm + cb_dc_last
                cb_quant = ijpeg_dc_ac(cb_dc, cb_ac)
                cr_ac = ijpeg_zigazg_block(cr_zigzag)
                cr_dc = cr_dpcm + cr_dc_last
                cr_quant = ijpeg_dc_ac(cr_dc, cr_ac)
                y_dc_last = y_dc
                cb_dc_last = cb_dc
                cr_dc_last = cr_dc
                y_dct = ijpeg_quant_y(y_quant)
                cb_dct = ijpeg_quant_cbcr(cb_quant)
                cr_dct = ijpeg_quant_cbcr(cr_quant)
                y_block[ii, jj] = ijpeg_dct(y_dct)
                cb_block[ii, jj] = ijpeg_dct(cb_dct)
                cr_block[ii, jj] = ijpeg_dct(cr_dct)
        y = utils.block2img(block=y_block)
        cb = utils.block2img(block=cb_block)
        cr = utils.block2img(block=cr_block)
        ycbcr = np.dstack([y, cb, cr])
        ycbcr = ycbcr + 128
        img = np.round(utils.ycbcr2rgb(img=ycbcr))
        img = img.clip(0, 255).astype(int)
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

dc_code_cat_dict = {
    4: 0,
    10: 1,
    11: 2,
    12: 3,
    13: 4,
    14: 5,
    30: 6,
    62: 7,
    126: 8,
    254: 9,
    510: 10,
    1022: 11,
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

ac_code_run_cat_dict = {
    10: "00",
    0: "01",
    1: "02",
    4: "03",
    11: "04",
    26: "05",
    120: "06",
    248: "07",
    1014: "08",
    65410: "09",
    65411: "0A",
    12: "11",
    27: "12",
    121: "13",
    502: "14",
    2038: "15",
    65412: "16",
    65413: "17",
    65414: "18",
    65415: "19",
    65416: "1A",
    28: "21",
    249: "22",
    1015: "23",
    4084: "24",
    65417: "25",
    65418: "26",
    65419: "27",
    65420: "28",
    65421: "29",
    65422: "2A",
    58: "31",
    503: "32",
    4085: "33",
    65423: "34",
    65424: "35",
    65425: "36",
    65426: "37",
    65427: "38",
    65428: "39",
    65429: "3A",
    59: "41",
    1016: "42",
    65430: "43",
    65431: "44",
    65432: "45",
    65433: "46",
    65434: "47",
    65435: "48",
    65436: "49",
    65437: "4A",
    122: "51",
    2039: "52",
    65438: "53",
    65439: "54",
    65440: "55",
    65441: "56",
    65442: "57",
    65443: "58",
    65444: "59",
    65445: "5A",
    123: "61",
    4086: "62",
    65446: "63",
    65447: "64",
    65448: "65",
    65449: "66",
    65450: "67",
    65451: "68",
    65452: "69",
    65453: "6A",
    250: "71",
    4087: "72",
    65454: "73",
    65455: "74",
    65456: "75",
    65457: "76",
    65458: "77",
    65459: "78",
    65460: "79",
    65461: "7A",
    504: "81",
    32704: "82",
    65462: "83",
    65463: "84",
    65464: "85",
    65465: "86",
    65466: "87",
    65467: "88",
    65468: "89",
    65469: "8A",
    505: "91",
    65470: "92",
    65471: "93",
    65472: "94",
    65473: "95",
    65474: "96",
    65475: "97",
    65476: "98",
    65477: "99",
    65478: "9A",
    506: "A1",
    65479: "A2",
    65480: "A3",
    65481: "A4",
    65482: "A5",
    65483: "A6",
    65484: "A7",
    65485: "A8",
    65486: "A9",
    65487: "AA",
    1017: "B1",
    65488: "B2",
    65489: "B3",
    65490: "B4",
    65491: "B5",
    65492: "B6",
    65493: "B7",
    65494: "B8",
    65495: "B9",
    65496: "BA",
    1018: "C1",
    65497: "C2",
    65498: "C3",
    65499: "C4",
    65500: "C5",
    65501: "C6",
    65502: "C7",
    65503: "C8",
    65504: "C9",
    65505: "CA",
    2040: "D1",
    65506: "D2",
    65507: "D3",
    65508: "D4",
    65509: "D5",
    65510: "D6",
    65511: "D7",
    65512: "D8",
    65513: "D9",
    65514: "EA",
    65515: "E1",
    65516: "E2",
    65517: "E3",
    65518: "E4",
    65519: "E5",
    65520: "E6",
    65521: "E7",
    65522: "E8",
    65523: "E9",
    65524: "EA",
    2041: "F0",
    65525: "F1",
    65526: "F2",
    65527: "F3",
    65528: "F4",
    65529: "F5",
    65530: "F6",
    65531: "F7",
    65532: "F8",
    65533: "F9",
    65534: "FA",
}
