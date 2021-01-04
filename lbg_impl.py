import numpy as np

import utils


def lbg_encoding(img: np.ndarray, param: dict) -> (np.ndarray, np.ndarray):
    codeword_dim = param["codeword_dim"]
    codebook_size = param["codebook_size"]
    epsilon = param["epsilon"]
    codeword_bits = int(np.log2(codebook_size))
    h, w = img.shape[0:2]
    h_m, w_m = int(h / codeword_dim[0]), int(w / codeword_dim[1])
    index = 0
    codebook = np.zeros(shape=(codebook_size, (codeword_dim[0]), (codeword_dim[1]))).astype(int)
    indices = np.zeros(shape=(h_m, w_m)).astype(int)
    bitstream = np.array([]).astype(int)
    bitstream = np.append(bitstream, utils.bitfield(h, 16))
    bitstream = np.append(bitstream, utils.bitfield(w, 16))

    temp_img = np.copy(img).reshape((h, w_m, codeword_dim[1]))
    temp_img = temp_img.reshape((h_m, codeword_dim[0], w_m, codeword_dim[1]))
    temp_img = temp_img.swapaxes(1, 2)

    for i in range(0, h_m):
        for j in range(0, w_m):
            codebook[index] = np.copy(temp_img[i, j])
            index += 1
            if index == codebook_size:
                break
        if index == codebook_size:
            break
    e = epsilon
    d = 0
    while not e < epsilon:
        img_ = np.zeros(shape=img.shape)
        temp_codebook = np.zeros(shape=codebook.shape).astype(np.longlong)
        for i in range(0, h_m):
            for j in range(0, w_m):
                ii = i * codeword_dim[0]
                jj = j * codeword_dim[1]

                distances = codebook - temp_img[i, j]
                distances = distances * distances
                distances = np.sum(distances, axis=1)
                distances = np.sum(distances, axis=1)
                k = np.argmin(distances)
                indices[i, j] = k
                img_[ii:ii + codeword_dim[0], jj:jj + codeword_dim[1]] = codebook[k]
                temp_codebook[k] += temp_img[i, j]

        for k in range(codebook_size):
            count = len(indices[indices == k])
            if count != 0:
                temp_codebook[k] = temp_codebook[k] / count
            codebook[k] = np.round(temp_codebook[k])
        d_ = img - img_
        d_ = d_ * d_
        d_ = np.sum(np.sum(d_))

        e = abs(d - d_) / d_
        d = d_

    codebook_flatten = codebook.flatten()

    for codebook_element in codebook_flatten:
        bitstream = np.append(bitstream, utils.bitfield(codebook_element, 8))

    for i in range(h_m):
        for j in range(w_m):
            index_bitstream = utils.bitfield(indices[i, j], codeword_bits)
            bitstream = np.append(bitstream, index_bitstream)
    return bitstream


def lbg_decoding(bitstream: np.ndarray, param: dict) -> np.ndarray:
    codeword_dim = param["codeword_dim"]
    codebook_size = param["codebook_size"]
    h = utils.ibitfield(bitstream, 16)
    bitstream = bitstream[16:]
    w = utils.ibitfield(bitstream, 16)
    bitstream = bitstream[16:]
    codebook = np.zeros(shape=(codebook_size, codeword_dim[0], codeword_dim[1]))
    for k in range(codebook_size):
        for i in range(codeword_dim[0]):
            for j in range(codeword_dim[1]):
                codebook[k, i, j] = utils.ibitfield(bitstream, 8)
                bitstream = bitstream[8:]
    img = np.zeros(shape=(h, w)).astype(int)
    codebook_size, codeword_dim = codebook.shape[0], codebook.shape[1:]
    codebook_size_bits = int(np.log2(codebook_size))
    for i in range(0, h, codeword_dim[0]):
        for j in range(0, w, codeword_dim[1]):
            index = utils.ibitfield(bitstream, codebook_size_bits)
            bitstream = bitstream[codebook_size_bits:]
            img[i:i + codeword_dim[0], j:j + codeword_dim[1]] = codebook[index]
    img = img.clip(0, 255)
    return img
