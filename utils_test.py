import unittest

import numpy as np
from PIL import Image

import utils


class MyTestCase(unittest.TestCase):

    def test_is_grey_scale(self):
        ans = {
            "baboon": {"file_path": "images/baboon.png", "is_gray": False},
            "barbara": {"file_path": "images/barbara.bmp", "is_gray": True},
            "boat": {"file_path": "images/boat.png", "is_gray": True},
            "goldhill": {"file_path": "images/goldhill.bmp", "is_gray": True},
            "lenagray": {"file_path": "images/lena.bmp", "is_gray": True},
            "lena": {"file_path": "images/lena.png", "is_gray": False},
            "peppers": {"file_path": "images/peppers.png", "is_gray": False},
        }
        for ans in ans.values():
            img = np.array(Image.open(ans["file_path"]).convert('RGB'))
            result = utils.is_grey_scale(img)
            self.assertEqual(result, ans["is_gray"])

    def test_dct_idct(self):
        n = 8
        a = np.dot(np.arange(n).reshape(n, 1), np.arange(n).reshape(1, n))
        b = utils.dct(a, n)
        _a = utils.idct(b, n)
        np.testing.assert_array_equal(a, _a, )


if __name__ == '__main__':
    unittest.main()
