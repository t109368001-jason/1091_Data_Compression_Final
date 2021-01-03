import numpy as np

import utils


def lbg_encoding(img: np.ndarray, codeword_dim: tuple, codebook_size: int, epsilon: float):
    codebook_size_bits = int(np.log2(codebook_size))
    h, w = img.shape[0:2]
    h_m, w_m = int(h / codeword_dim[0]), int(w / codeword_dim[1])
    index = 0
    codebook = np.zeros(shape=(codebook_size, codeword_dim[0], codeword_dim[1])).astype(int)
    indices = np.zeros(shape=(h_m, w_m)).astype(int)
    bits = np.array([]).astype(int)
    bits = np.append(bits, utils.bitfield(h, 16))
    bits = np.append(bits, utils.bitfield(w, 16))
    for i in range(0, h, codeword_dim[0]):
        for j in range(0, w, codeword_dim[1]):
            codebook[index] = np.copy(img[i:i + codeword_dim[0], j:j + codeword_dim[1]])
            index += 1
            if index == codebook_size:
                break
        if index == codebook_size:
            break
    e = epsilon
    d = 0
    while not e < epsilon:
        for i in range(0, h, codeword_dim[0]):
            for j in range(0, w, codeword_dim[1]):
                distances = np.array(
                    [codeword - img[i:i + codeword_dim[0], j:j + codeword_dim[1]] for codeword in codebook])
                distances = distances * distances
                distances = np.sum(distances, axis=1)
                distances = np.sum(distances, axis=1)
                k = np.argmin(distances)
                indices[int(i / codeword_dim[0]), int(j / codeword_dim[1])] = k

        for k in range(codebook_size):
            sum = np.zeros(shape=codeword_dim)
            count = len(indices[indices == k])
            for i in range(h_m):
                for j in range(w_m):
                    if k == indices[i, j]:
                        ii = i * codeword_dim[0]
                        jj = j * codeword_dim[1]
                        sum += img[ii:ii + codeword_dim[0], jj:jj + codeword_dim[1]].astype(float) / count
            codebook[k] = sum.astype(int)
        d_ = 0
        for i in range(h_m):
            for j in range(w_m):
                k = indices[i, j]
                ii = i * codeword_dim[0]
                jj = j * codeword_dim[1]
                distance = codebook[k] - img[ii:ii + codeword_dim[0], jj:jj + codeword_dim[1]]
                distance = distance * distance
                distance = np.sum(np.sum(distance))
                d_ += distance
        e = abs(d - d_) / d_
        d = d_

    for i in range(h_m):
        for j in range(w_m):
            index_bits = utils.bitfield(indices[i, j], codebook_size_bits)
            bits = np.append(bits, index_bits)
    return bits, codebook


def lbg_decoding(bits: np.ndarray, codebook: np.ndarray):
    h = utils.ibitfield(bits, 16)
    bits = bits[16:]
    w = utils.ibitfield(bits, 16)
    bits = bits[16:]
    img = np.zeros(shape=(h, w)).astype(int)
    codebook_size, codeword_dim = codebook.shape[0], codebook.shape[1:]
    codebook_size_bits = int(np.log2(codebook_size))
    h_m, w_m = int(h / codeword_dim[0]), int(w / codeword_dim[1])

    for i in range(0, h, codeword_dim[0]):
        for j in range(0, w, codeword_dim[1]):
            index = utils.ibitfield(bits, codebook_size_bits)
            bits = bits[codebook_size_bits:]
            img[i:i + codeword_dim[0], j:j + codeword_dim[1]] = codebook[index]
    img = img.clip(0, 255)
    return img
