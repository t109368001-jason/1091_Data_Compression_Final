from unittest import TestCase

import numpy as np

from jpeg_impl import jpeg_down_sampling, jpeg_up_sampling


class Test(TestCase):
    def test_jpeg_down_sampling(self):
        img = [[1, 2, 3, 4],
               [5, 6, 7, 8]]
        result1 = jpeg_down_sampling(np.array(img), 4, 4, 4)
        result2 = jpeg_down_sampling(np.array(img), 4, 4, 0)
        result3 = jpeg_down_sampling(np.array(img), 4, 2, 2)
        result4 = jpeg_down_sampling(np.array(img), 4, 2, 0)
        result5 = jpeg_down_sampling(np.array(img), 4, 1, 1)
        result6 = jpeg_down_sampling(np.array(img), 4, 1, 0)

        np.testing.assert_array_equal(result1, np.array([[1, 2, 3, 4],
                                                         [5, 6, 7, 8]]))
        np.testing.assert_array_equal(result2, np.array([[1, 2, 3, 4]]))
        np.testing.assert_array_equal(result3, np.array([[1, 3],
                                                         [5, 7]]))
        np.testing.assert_array_equal(result4, np.array([[1, 3]]))
        np.testing.assert_array_equal(result5, np.array([[1],
                                                         [5]]))
        np.testing.assert_array_equal(result6, np.array([[1]]))

    def test_jpeg_up_sampling(self):
        result1 = jpeg_up_sampling(np.array([[1, 2, 3, 4],
                                             [5, 6, 7, 8]]), 4, 4, 4)
        result2 = jpeg_up_sampling(np.array([[1, 2, 3, 4]]), 4, 4, 0)
        result3 = jpeg_up_sampling(np.array([[1, 3],
                                             [5, 7]]), 4, 2, 2)
        result4 = jpeg_up_sampling(np.array([[1, 3]]), 4, 2, 0)
        result5 = jpeg_up_sampling(np.array([[1],
                                             [5]]), 4, 1, 1)
        result6 = jpeg_up_sampling(np.array([[1]]), 4, 1, 0)

        np.testing.assert_array_equal(result1, np.array([[1, 2, 3, 4],
                                                         [5, 6, 7, 8]]))
        np.testing.assert_array_equal(result2, np.array([[1, 2, 3, 4],
                                                         [1, 2, 3, 4]]))
        np.testing.assert_array_equal(result3, np.array([[1, 1, 3, 3],
                                                         [5, 5, 7, 7]]))
        np.testing.assert_array_equal(result4, np.array([[1, 1, 3, 3],
                                                         [1, 1, 3, 3]]))
        np.testing.assert_array_equal(result5, np.array([[1, 1, 1, 1],
                                                         [5, 5, 5, 5]]))
        np.testing.assert_array_equal(result6, np.array([[1, 1, 1, 1],
                                                         [1, 1, 1, 1]]))
