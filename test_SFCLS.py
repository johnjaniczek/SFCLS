from unittest import TestCase
from unmix import SFCLS_unmix
import numpy as np



class TestSFCLS(TestCase):
    def test_unmixing(self):
        # create random data
        m = 73
        n = 36
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        # test unmixing

        x, loss = SFCLS_unmix(A, b, lam=0.001)
        self.assertTrue(np.allclose(b, np.matmul(A, x), atol=loss))
        self.assertTrue(np.allclose(x.sum(), 1, atol=0.01))

