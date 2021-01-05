from unittest import TestCase

import numpy as np

import utils


class Test(TestCase):
    def test_img2block(self):
        img = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [1, 2, 3, 4],
                        [5, 6, 7, 8]])
        block1 = utils.img2block(img=img, block_shape=(2, 2))
        img_ = utils.block2img(block=block1)

        np.testing.assert_array_equal(block1, np.array([[[[1, 2], [5, 6]], [[3, 4], [7, 8]]],
                                                        [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]]))
        np.testing.assert_array_equal(img, img_)

    def test_rgb2ycbcr(self):
        img = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
                        [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]],
                        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
                        [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]]]).astype(float)
        ycbcr = utils.rgb2ycbcr(img)
        rgb = utils.ycbcr2rgb(ycbcr).astype(float)
        self.assertLessEqual(np.sum(np.abs(img - rgb), axis=None), 1e-6)
