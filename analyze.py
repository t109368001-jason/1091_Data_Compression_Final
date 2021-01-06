import os

import numpy as np
from PIL import Image
import utils

import compression

input_folder = "./images/"
output_folder = "./output/"
# image_names = ["barbara.bmp", "boat.png", "goldhill.bmp", "lena.bmp"]
image_names = ["baboon.png", "barbara.bmp", "boat.png", "goldhill.bmp", "lena.bmp", "lena.png", "peppers.png"]
# codeword_dims = [(8, 8), (32, 32), (16, 16), (4, 4)]
codeword_dims = [(8, 8), (32, 32), (16, 16), (4, 4)]
# codebook_sizes = [64, 4, 8, 16, 32, 128, 256, 512, 1024, 2048, 4096, 8192]
codebook_sizes = [64, 4, 8, 16, 32, 128, 256, 512, 1024, 2048, 4096, 8192]
epsilon = 1e-4

skip_jpeg = False
# skip_jpeg = True
# skip_lbg = False
skip_lbg = True
output_path = output_folder + "output.csv"

if __name__ == '__main__':
    if not os.path.exists(output_path):
        with open(output_path, "w") as output_csv:
            output_csv.write(
                "image,ori_bits,algorithm,bits,jpeg_n,codeword_dim,codebook_size,en_time,de_time,mse,psnr,r,entropy\n")
    if not skip_jpeg:
        for image_name in image_names:
            file_path = input_folder + image_name

            img = np.array(Image.open(file_path).convert('RGB')).astype(int)
            is_gray = utils.is_grey_scale(img)
            if is_gray:
                img = img[:, :, 0]
            channel = 1 if is_gray else 3
            h, w = img.shape[0:2]
            ori_bits = h * w * channel * 8
            n = 8
            jpeg_param = {
                "n": n,
                "m": (int(h / n)),
                "is_gray": is_gray,
                "jab": (4, 4, 4),
                "resolution": (h, w)
            }
            algorithm_name = "JPEG"
            print("main() image={}, algorithm={}, n={}".format(image_name, algorithm_name, n))
            bitstream, img_, encoding_time, decoding_time = compression.perf(img=img, algorithm_name=algorithm_name,
                                                                             param=jpeg_param)
            if is_gray:
                image_file = Image.fromarray(np.dstack((img_, img_, img_)).astype(np.uint8))
            else:
                image_file = Image.fromarray(img_.astype(np.uint8))
            image_file.save(output_folder + "{}_{}_{}.bmp".format(image_name, algorithm_name, n))
            mse = np.mean(np.square(img - img_))
            psnr = 10 * np.log10(255 * 255 * channel / mse)
            hist, bins = np.histogram(img_, np.arange(257))
            prob = hist / np.sum(hist)
            prob[prob == 0] = 1
            prob_log2 = np.log2(prob)
            prob_log2[prob_log2 == -np.inf] = 0
            entropy = np.sum(-prob * prob_log2)
            with open(output_path, "a") as output_csv:
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
    if not skip_lbg:
        for codeword_dim in codeword_dims:
            for codebook_size in codebook_sizes:
                for image_name in image_names:
                    file_path = input_folder + image_name

                    img = np.array(Image.open(file_path).convert('RGB')).astype(int)
                    img = img[:, :, 0]
                    h, w = img.shape
                    ori_bits = h * w * 8
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
                    bitstream, img_, encoding_time, decoding_time = compression.perf(img=img,
                                                                                     algorithm_name=algorithm_name,
                                                                                     param=lgb_param)
                    image_file = Image.fromarray(np.dstack((img_, img_, img_)).astype(np.uint8))
                    image_file.save(output_folder + "{}_{}_{}_{}.bmp".format(image_name,
                                                                             algorithm_name,
                                                                             "{}x{}".format(
                                                                                 codeword_dim[0],
                                                                                 codeword_dim[1]),
                                                                             codebook_size))
                    mse = np.mean(np.square(img - img_))
                    psnr = 10 * np.log10(255 * 255 / mse)
                    hist, bins = np.histogram(img_, np.arange(257))
                    prob = hist / np.sum(hist)
                    prob[prob == 0] = 1
                    prob_log2 = np.log2(prob)
                    prob_log2[prob_log2 == -np.inf] = 0
                    entropy = np.sum(-prob * prob_log2)
                    with open(output_path, "a") as output_csv:
                        output_csv.write(
                            "{},{},{},{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.6f},{:.6f}\n".format(image_name,
                                                                                                      ori_bits,
                                                                                                      algorithm_name,
                                                                                                      len(bitstream),
                                                                                                      None,
                                                                                                      "{}x{}".format(
                                                                                                          codeword_dim[
                                                                                                              0],
                                                                                                          codeword_dim[
                                                                                                              1]),
                                                                                                      codebook_size,
                                                                                                      encoding_time,
                                                                                                      decoding_time,
                                                                                                      mse,
                                                                                                      psnr,
                                                                                                      len(
                                                                                                          bitstream) / h / w,
                                                                                                      entropy))
