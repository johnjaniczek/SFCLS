from unittest import TestCase
from unmix import L_infty_unmix
import numpy as np



class TestLinfty(TestCase):
    def test_unmixing(self):
        # create random data
        m = 73
        n = 36
        np.random.seed(1)
        A = np.random.random((m, n))
        x_true = np.random.random(n)
        x_true = x_true / sum(x_true)
        b = A @ x_true

        # test unmixing

        x = L_infty_unmix(A, b, lam=0.001)
        self.assertTrue(np.allclose(b, np.matmul(A, x), atol=0.1))
        self.assertTrue(np.allclose(x.sum(), 1, atol=0.01))

