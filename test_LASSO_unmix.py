from unittest import TestCase
from unmix import LASSO_unmix
import numpy as np

class TestLASSO(TestCase):
    def test_unmixing(self):
        # create random data
        m = 73
        n = 36
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        # test unmixing
        x, loss = LASSO_unmix(A, b, lam=0.1)
        self.assertTrue(np.allclose(b, np.matmul(A, x), atol=loss))