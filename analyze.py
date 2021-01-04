import os

import numpy as np
from PIL import Image

import compression

input_folder = "./images/"
# image_names = ["barbara.bmp", "boat.png", "goldhill.bmp", "lena.bmp"]
image_names = ["barbara.bmp", "boat.png", "goldhill.bmp", "lena.bmp"]
# codeword_dims = [(4, 4), (8, 8), (16, 16)]
codeword_dims = [(8, 8), (16, 16), (32, 32)]
# codebook_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
codebook_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
epsilon = 1e-4

if __name__ == '__main__':
    if not os.path.exists("output.csv"):
        with open("output.csv", "w") as output_csv:
            output_csv.write(
                "image,ori_bits,algorithm,bits,jpeg_n,codeword_dim,codebook_size,en_time,de_time,mse,psnr,r,entropy\n")
    for image_name in image_names:
        file_path = input_folder + image_name

        img = np.array(Image.open(file_path).convert('RGB')).astype(int)
        img = img[:, :, 0]
        h, w = img.shape
        ori_bits = h * w * 8

        n = 8
        jpeg_param = {
            "n": n,
            "m": (int(h / n))
        }
        algorithm_name = "JPEG"
        print("main() image={}, algorithm={}, n={}".format(image_name, algorithm_name, n))
        bitstream, img_, encoding_time, decoding_time = compression.perf(img=img, algorithm_name=algorithm_name,
                                                                         param=jpeg_param)
        mse = np.mean(np.square(img - img_))
        psnr = 10 * np.log10(255 * 255 / mse)
        hist, bins = np.histogram(img_, np.arange(257))
        prob = hist / np.sum(hist)
        prob[prob == 0] = 1
        prob_log2 = np.log2(prob)
        prob_log2[prob_log2 == -np.inf] = 0
        entropy = np.sum(-prob * prob_log2)
        with open("output.csv", "a") as output_csv:
            output_csv.write(
                "{},{},{},{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.6f},{:.6f}\n".format(image_name, ori_bits,
                                                                                          algorithm_name,
                                                                                          len(bitstream), n,
                                                                                          None,
                                                                                          None, encoding_time,
                                                                                          decoding_time, mse,
                                                                                          psnr,
                                                                                          len(bitstream) / h / w,
                                                                                          entropy))

        for codeword_dim in codeword_dims:
            for codebook_size in codebook_sizes:
                if codebook_size * codeword_dim[0] * codeword_dim[1] >= h * w:
                    continue
                lgb_param = {
                    "codeword_dim": codeword_dim,
                    "codebook_size": codebook_size,
                    "epsilon": epsilon
                }
                algorithm_name = "LGB"
                print("main() image={}, algorithm={}, codeword_dim={}, codebook_size={}".format(image_name,
                                                                                                algorithm_name,
                                                                                                codeword_dim,
                                                                                                codebook_size))
                bitstream, img_, encoding_time, decoding_time = compression.perf(img=img, algorithm_name=algorithm_name,
                                                                                 param=lgb_param)
                mse = np.mean(np.square(img - img_))
                psnr = 10 * np.log10(255 * 255 / mse)
                hist, bins = np.histogram(img_, np.arange(257))
                prob = hist / np.sum(hist)
                prob[prob == 0] = 1
                prob_log2 = np.log2(prob)
                prob_log2[prob_log2 == -np.inf] = 0
                entropy = np.sum(-prob * prob_log2)
                with open("output.csv", "a") as output_csv:
                    output_csv.write(
                        "{},{},{},{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.6f},{:.6f}\n".format(image_name, ori_bits,
                                                                                                  algorithm_name,
                                                                                                  len(bitstream), None,
                                                                                                  "{}x{}".format(
                                                                                                      codeword_dim[0],
                                                                                                      codeword_dim[1]),
                                                                                                  codebook_size,
                                                                                                  encoding_time,
                                                                                                  decoding_time, mse,
                                                                                                  psnr,
                                                                                                  len(
                                                                                                      bitstream) / h / w,
                                                                                                  entropy))
