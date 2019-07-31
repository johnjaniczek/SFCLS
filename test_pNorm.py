from unittest import TestCase
from unmix import p_norm_unmix
import numpy as np



class TestGrav(TestCase):
    def test_unmixing(self):
        # create random data
        m = 73
        n = 36
        np.random.seed(1)
        A = np.random.random((m, n))
        x_true = np.random.random(n)
        x_true = x_true / sum(x_true)
        b = A@x_true


        # test unmixing

        x = p_norm_unmix(A, b, lam=0.0001, p=0.9)
        self.assertTrue(np.allclose(b, np.matmul(A, x), atol=0.01))
        self.assertTrue(np.allclose(x.sum(), 1, atol=0.01))

