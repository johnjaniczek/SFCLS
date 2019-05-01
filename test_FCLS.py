from unittest import TestCase
from unmix import FCLS_unmix
import numpy as np

class TestFCLS(TestCase):
    def test_unmixing(self):
        # create random data
        m = 73
        n = 36
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        # test unmixing
        x, loss = FCLS_unmix(A, b)
        self.assertTrue(np.allclose(b, np.matmul(A, x), atol=loss))
        self.assertTrue(np.allclose(x.sum(), 1, atol=0.01))





