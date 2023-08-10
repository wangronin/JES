import autograd.numpy as anp
import numpy as np

from problems.problems import Problem

##TODO: The following codes is converted from mDTLZ.m. The g and h functions can be optimized

__authors__ = ["Kaifeng Yang"]
__version__ = ["0.0.1"]


class WOSGZ(Problem):
    def __init__(self, n_var, n_obj):
        self.n_obj = n_obj
        self.n_var = n_var

        self.D = n_var  #
        self.M = n_obj
        _xl = -np.ones((n_var))
        _xl[0] = 0
        self.elementwise_evaluation = False
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=_xl, xu=1, type_var=anp.double)
        ## TODO: xu should be redefined!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def _initialize(self, x):
        self.N = x.shape[1]
        self.X = np.zeros((self.M, self.N))  # M*N array
        self.g = np.zeros((self.M, self.N))  # M*N array

    # TODO: integratge this function to _compute_para
    def _compute_X(self, x):
        X = self.X
        X[0, :] = 1 - x[0, :]
        X[self.M - 1, :] = np.prod(x[0 : self.M - 1, :], axis=0)

        for i in range(1, self.M - 1):
            X[i, :] = np.prod(x[0:i, :], axis=0) * (1 - x[i, :])
        self.X = X

    def _compute_obj(self, x, A, B, C, D, E):
        J = [[0]] * self.M  # M*1 list
        Jsize = np.zeros((self.M, 1))  # M*1 array
        co_r = np.ones((self.M, self.M)) / (self.M - 1)  # M*M array
        cor = np.zeros((self.M, self.M, self.N)) / (self.M - 1)
        R = np.zeros((self.M, 1))  # M*1 array

        cor = -self.M * np.diag(np.diag(co_r)) + co_r
        r = [
            np.sqrt((self.M - 1) / self.M) * np.max(np.matmul(cor, self.X[:, i] - 1 / self.M))
            for i in range(0, self.N)
        ]  # N list, each item is 1*1 array

        R[0 : self.M - 1] = 1 / (self.M - 1) - 1 / self.M
        R[self.M - 1] = -1 / self.M
        R_long = np.sum(R**2) ** 0.5

        h = r / R_long  # N list, each item is 1*1 array
        theta = h ** (self.M - 1)  # N list, each item is 1*1 array

        Y_bias = A * (np.sin(0.5 * np.pi * theta) ** D) + 1  # N list, each item is 1*1 array
        X_bias = 0.9 * (np.sin(0.5 * np.pi * theta) ** B)  # N list, each item is 1*1 array

        M_to_D = np.array([i for i in range(self.M, self.D + 1)])
        t = [
            x[self.M - 1 :, i]
            - X_bias[i] * np.cos(E * np.pi * h[i] + 0.5 * np.pi * (self.D + 2) * M_to_D / self.D)
            for i in range(0, self.N)
        ]  # N list, each item is 1*3 array
        t = np.transpose(np.array(t))  # M*N array

        J[0], Jsize[0] = [j for j in range(self.M, self.D, self.M)], len(J[0])

        index_in_t = np.array(J[0]) - self.M
        n_in_t = index_in_t.shape[0]
        self.g[0, :] = (
            Y_bias / Jsize[0] * np.sum(np.abs(t[index_in_t, :]).reshape(n_in_t, self.N) ** C, axis=0)
        )

        J[self.M - 1] = np.arange(2 * self.M - 1, self.D + 1, self.M)
        Jsize[self.M - 1] = J[self.M - 1].shape[0]
        index_in_t = np.array(J[self.M - 1]) - self.M
        self.g[self.M - 1, :] = (
            Y_bias / Jsize[self.M - 1] * np.sum(np.abs(t[index_in_t, :]).reshape(n_in_t, self.N) ** C, axis=0)
        )

        for j in range(1, self.M):
            J[j] = np.arange(self.M + j, self.D + 1, self.M)
            Jsize[j] = J[j].shape[0]
            index_in_t = np.array(J[j]) - self.M
            self.g[j, :] = (
                Y_bias / Jsize[j] * np.sum(np.abs(t[index_in_t, :]).reshape(n_in_t, self.N) ** C, axis=0)
            )

        F = self.X + self.g
        return F


def generic_sphere(ref_dirs):
    return ref_dirs / anp.tile(anp.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))


class WOSGZ1(WOSGZ):
    def __init__(self, n_var=5, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    # def _calc_pareto_front(self, ref_dirs=None):
    #     return 0.5 * ref_dirs

    def _evaluate(self, x, out, *args, requires_F=True, **kwargs):
        A = 1.0
        B = 4.0
        C = 2.0
        D = 4.0
        E = 3.0

        x = np.transpose(x)  # tranpose
        self._initialize(x)  # initialize all
        self._compute_X(x)  # calculate X
        F = self._compute_obj(x, A, B, C, D, E)
        out["F"] = np.transpose(F)
        return out


class WOSGZ2(WOSGZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _evaluate(self, x, out, *args, requires_F=True, **kwargs):
        A = 2.0
        B = 4.0
        C = 2.0
        D = 4.0
        E = 3.0

        x = np.transpose(x)  # tranpose
        self._initialize(x)  # initialize all
        self._compute_X(x)  # calculate X
        F = self._compute_obj(x, A, B, C, D, E)
        out["F"] = np.transpose(F)
        return out


class WOSGZ3(WOSGZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _evaluate(self, x, out, *args, requires_F=True, **kwargs):
        A = 2.0
        B = 2.0
        C = 2.0
        D = 2.0
        E = 3.0

        x = np.transpose(x)  # tranpose
        self._initialize(x)  # initialize all
        self._compute_X(x)  # calculate X
        F = self._compute_obj(x, A, B, C, D, E)
        out["F"] = np.transpose(F)
        return out


class WOSGZ4(WOSGZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _evaluate(self, x, out, *args, requires_F=True, **kwargs):
        A = 3.0
        B = 2.0
        C = 2.0
        D = 2.0
        E = 3.0

        x = np.transpose(x)  # tranpose
        self._initialize(x)  # initialize all
        self._compute_X(x)  # calculate X
        F = self._compute_obj(x, A, B, C, D, E)
        out["F"] = np.transpose(F)
        return out


class WOSGZ5(WOSGZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _evaluate(self, x, out, *args, requires_F=True, **kwargs):
        A = 3.0
        B = 1.0
        C = 2.0
        D = 1.0
        E = 3.0

        x = np.transpose(x)  # tranpose
        self._initialize(x)  # initialize all
        self._compute_X(x)  # calculate X
        F = self._compute_obj(x, A, B, C, D, E)
        out["F"] = np.transpose(F)
        return out


class WOSGZ6(WOSGZ):
    def _evaluate(self, x, out, *args, requires_F=True, **kwargs):
        A = 4.0
        B = 1.0
        C = 2.0
        D = 1.0
        E = 3.0

        x = np.transpose(x)  # tranpose
        self._initialize(x)  # initialize all
        self._compute_X(x)  # calculate X
        F = self._compute_obj(x, A, B, C, D, E)
        out["F"] = np.transpose(F)
        return out


class WOSGZ7(WOSGZ):
    def _evaluate(self, x, out, *args, requires_F=True, **kwargs):
        A = 2.0
        B = 2.0
        C = 0.8
        D = 2.0
        E = 3.0

        x = np.transpose(x)  # tranpose
        self._initialize(x)  # initialize all
        self._compute_X(x)  # calculate X
        F = self._compute_obj(x, A, B, C, D, E)
        out["F"] = np.transpose(F)
        return out


class WOSGZ8(WOSGZ):
    def _evaluate(self, x, out, *args, requires_F=True, **kwargs):
        A = 2.0
        B = 2.0
        C = 2.0
        D = 2.0
        E = 3.0

        x = np.transpose(x)  # tranpose
        self._initialize(x)  # initialize all
        self._compute_X(x)  # calculate X
        F = self._compute_obj(x, A, B, C, D, E)
        F = self.X**2 + self.g
        out["F"] = np.transpose(F)
        return out
