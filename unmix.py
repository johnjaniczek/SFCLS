import numpy as np
from numpy.linalg import norm
import cvxpy as cp
from scipy.optimize import *

# make global variables to pass to
# scipy objective and jacobian functions
# because scipy only supports f(x)


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

def p_norm_unmix(A, b, lam=0.01, p=0.8):

        # setup problem
        m = A.shape[0]
        n = A.shape[1]

        def func(x, A=A, b=b, lam=lam, p=p):
                return np.sum((A@x - b)**2) + lam*norm(x, ord=p)

        def func_deriv(x, A=A, b=b, lam=lam, p=p):
                # sum of squares term
                dfdx = 2*A.T@A@x - 2*A.T@b

                # p norm term
                for i in range(x.shape[0]):
                        x_i = np.clip(x[i], 1e-10, None)
                        dfdx[i] += lam*np.sign(x_i)*(np.abs(x_i)/norm(x, ord=p))**(p-1)
                return dfdx

        cons = ({'type': 'eq',
                 'fun': lambda x: np.ones(n).T@x - 1,
                 'jac': lambda x: np.ones(n)},
                {'type': 'ineq',
                 'fun': lambda x: x,
                 'jac': lambda x: np.identity(n)})
        x0 = np.full(n, 1/n)
        res = minimize(func, x0, jac=func_deriv, constraints=cons,
                       method="SLSQP", options={'disp': False, 'maxiter': 500, 'ftol':1e-10})
        return res.x , res


def delta_norm_unmix(A, b, lam=1e-7, delta=1e-3):

        # setup problem
        m = A.shape[0]
        n = A.shape[1]

        def func(x, A=A, b=b, lam=lam, delta=delta):
                return np.sum((A@x - b)**2) + lam*x@np.reciprocal(x+delta)

        def func_deriv(x, A=A, b=b, lam=lam, delta=delta):
                # sum of squares term
                dfdx = 2*A.T@A@x - 2*A.T@b

                # L0 norm term
                for i in range(x.shape[0]):
                        dfdx[i] += lam*delta/((delta + x[i])**2)
                return dfdx

        cons = ({'type': 'eq',
                 'fun': lambda x: np.ones(n).T@x - 1,
                 'jac': lambda x: np.ones(n)},
                {'type': 'ineq',
                 'fun': lambda x: x,
                 'jac': lambda x: np.identity(n)})
        x0 = np.full(n, 1/n)
        res = minimize(func, x0, jac=func_deriv, constraints=cons,
                       method="SLSQP", options={'disp': False, 'maxiter': 500, 'ftol':1e-9,
                                                'eps': 1e-11})
        return res.x , res


def gravitron(A, b, infty_lam=1e-7, delta_lam=1e-7, p_lam=1e-6, p=0.8, delta=1e-3):

        # setup problem
        m = A.shape[0]
        n = A.shape[1]

        def func(x, A=A, b=b, delta_lam=delta_lam, infty_lam=infty_lam,
                 p_lam=p_lam, p=p, delta=delta):
                return np.sum((A@x - b)**2) + delta_lam*x@np.reciprocal(x+delta) + infty_lam/x.max() + p_lam*norm(x, ord=p)

        def func_deriv(x, A=A, b=b, infty_lam=infty_lam, delta_lam=delta_lam,
                       p_lam=p_lam, p=p, delta=delta):

                # sum of squares term
                dfdx = 2*A.T@A@x - 2*A.T@b

                # L infinity norm term
                max_i = np.argmax(x)
                dfdx[max_i] += -infty_lam/x[max_i]**2

                # delta norm and p_norm terms
                for i in range(x.shape[0]):
                        dfdx[i] += delta_lam*delta/((delta + x[i])**2)
                        x_i = np.clip(x[i], 1e-10, None)
                        dfdx[i] += p_lam * np.sign(x_i) * (np.abs(x_i) / norm(x, ord=p)) ** (p - 1)
                return dfdx



        cons = ({'type': 'eq',
                 'fun': lambda x: np.ones(n).T@x - 1,
                 'jac': lambda x: np.ones(n)},
                {'type': 'ineq',
                 'fun': lambda x: x,
                 'jac': lambda x: np.identity(n)})
        x0 = np.full(n, 1/n)
        res = minimize(func, x0, jac=func_deriv, constraints=cons,
                       method="SLSQP", options={'disp': False, 'maxiter': 500, 'ftol':1e-10,
                                                'eps': 1e-11})
        return res.x, res








