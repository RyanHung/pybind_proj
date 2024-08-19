import _functions as _func
import numpy as np
import pandas as pd

def check_symmetry(a, rtol=1e-5, atol=1e-8):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def get_lambda_matrix(lam1, S):
    if isinstance(lam1, float) | isinstance(lam1, int):
        lam_mat = np.full_like(S, lam1, order='F', dtype='float64')
    elif isinstance(lam1, np.ndarray):
        lam_mat = lam1
    return lam_mat

class TestClass:
    def testOpenMP(self, S):
        return _func.testOpenMPThreads(S)
    def testBLAS(self, S):
        return _func.testBLAS(S)