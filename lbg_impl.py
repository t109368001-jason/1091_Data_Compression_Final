import numpy as np

import utils


def lbg_encoding(img: np.ndarray, codeword_dim: tuple, codebook_size: int, epsilon: float):
    codebook_size_bits = int(np.log2(codebook_size))
    h, w = img.shape[0:2]
    h_m, w_m = int(h / codeword_dim[0]), int(w / codeword_dim[1])
    index = 0
    codebook = np.zeros(shape=(codebook_size, (codeword_dim[0]), (codeword_dim[1]))).astype(int)
    indices = np.zeros(shape=(h_m, w_m)).astype(int)
    bitstream = np.array([]).astype(int)
    bitstream = np.append(bitstream, utils.bitfield(h, 16))
    bitstream = np.append(bitstream, utils.bitfield(w, 16))

    for i in range(0, h_m):
        for j in range(0, w_m):
            ii = i * codeword_dim[0]
            jj = j * codeword_dim[1]
            codebook[index] = np.copy(img[ii:ii + codeword_dim[0], jj:jj + codeword_dim[1]])
            index += 1
            if index == codebook_size:
                break
        if index == codebook_size:
            break
    e = epsilon
    d = 0
    while not e < epsilon:
        img_ = np.zeros(shape=img.shape)
        temp_codebook = np.zeros(shape=codebook.shape)
        for i in range(0, h_m):
            for j in range(0, w_m):
                ii = i * codeword_dim[0]
                jj = j * codeword_dim[1]

                distances = codebook - img[ii:ii + codeword_dim[0], jj:jj + codeword_dim[1]]
                distances = distances * distances
                distances = np.sum(distances, axis=1)
                distances = np.sum(distances, axis=1)
                k = np.argmin(distances)
                indices[i, j] = k
                img_[ii:ii + codeword_dim[0], jj:jj + codeword_dim[1]] = codebook[k]
                temp_codebook[k] += img[ii:ii + codeword_dim[0], jj:jj + codeword_dim[1]]

        for k in range(codebook_size):
            count = len(indices[indices == k])
            codebook[k] = temp_codebook[k] / count
        d_ = img - img_
        d_ = d_ * d_
        d_ = np.sum(np.sum(d_))

        e = abs(d - d_) / d_
        d = d_

    for i in range(h_m):
        for j in range(w_m):
            index_bitstream = utils.bitfield(indices[i, j], codebook_size_bits)
            bitstream = np.append(bitstream, index_bitstream)
    return bitstream, codebook


def lbg_decoding(bitstream: np.ndarray, codebook: np.ndarray):
    h = utils.ibitfield(bitstream, 16)
    bitstream = bitstream[16:]
    w = utils.ibitfield(bitstream, 16)
    bitstream = bitstream[16:]
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
