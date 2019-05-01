import numpy as np
import cvxpy as cp


def FCLS_unmix(A, b):
        # setup problem
        n = A.shape[1]
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(A * x - b))
        constraints = [0 <= x, cp.sum(x) == 1]
        prob = cp.Problem(objective, constraints)


        # find optimal solution
        loss = prob.solve()
        return x.value, loss


def LASSO_unmix(A, b, lam=0.01):
        m = A.shape[0]
        n = A.shape[1]

        # setup problem
        x = cp.Variable(n)
        lam_cp = cp.Parameter(nonneg=True)
        lam_cp.value = lam
        objective = cp.Minimize(cp.sum_squares(A*x - b) + lam_cp*cp.norm(x, 1))
        constraints = [0 <= x]
        prob = cp.Problem(objective, constraints)

        # find optimal solution
        loss = prob.solve()
        return x.value, loss


def SFCLS_unmix(A, b, lam=1e-6):
        A = A
        m = A.shape[0]
        n = A.shape[1]

        x = cp.Variable(n)
        c = cp.Parameter(n)
        objective = cp.Minimize(cp.sum_squares(A * x - b) + lam * cp.inv_pos(c*x))
        constraints = [x >= 0,
                       cp.sum(x) == 1]
        prob = cp.Problem(objective, constraints)
        # iterate over n convex programs
        temp_loss = np.zeros(n)
        for i in range(n):
                c_new = np.zeros(n)
                c_new[i] = 1
                c.value = c_new
                temp_loss[i] = prob.solve()

        # choose index with minimum loss
        i_min = temp_loss.argmin()
        c_new = np.zeros(n)
        c_new[i_min] = 1
        c.value = c_new
        loss = prob.solve()
        return x.value, loss








