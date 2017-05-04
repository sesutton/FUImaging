import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes as ct


def convex_cone(data, latents):
    """do maximum projection of the data"""

    data = data.copy()
    res = {'base': [], 'timecourses': []}

    for i in range(latents):
        # most interesting column
        col_concern = np.max(data, axis=0)
        best_col = np.argmax(col_concern)

        timecourse = data[:, best_col].copy()
        norm = np.dot(timecourse, timecourse)
        timecourse /= np.sqrt(norm)
        base = np.dot(data.T, timecourse)
        base[base < 0] = 0

        data -= np.outer(timecourse, base)

        res['base'].append(base)
        res['timecourses'].append(timecourse)

    return res


def get_nmf(self):

        floatp = ndpointer(ct.c_float, flags="F_CONTIGUOUS")

        lib = ct.CDLL('./cuda/libnmf.so')
        func = lib.nmf
        func.restype = None
        func.argtypes = [floatp, floatp, floatp, ct.c_int, ct.c_int, ct.c_int]
        return func



# Different sparsnes norms
NORMS = {'global_sparse': lambda new_vec, x: np.sum(x, 0),
         'local_sparse': lambda new_vec, x: 1}


class nmf_cuda(object):
    """NMF with regularized HALS algorithm"""

    def __init__(self, num_comp, **kwargs):
        """
        Parameter
        ---------
        num_comp: int
            number of components/latents in the factorization

        Keyword Args
        ------------
        maxcount: int, default 100
            maximal number of iterations
        smooth_param: float, default 0
            strength of smoothness regularisation
        sparse_param: float, default 0
            strength of sparseness regularisation
        sparse_fct: str, default 'global_sparse'
            * 'global_sparse': sparseness regularisation over all components
            * 'local_sparse': sparseness regularisation within a component
        neg_time: bool, default False
            it True negative activations are allowed
        init: 'convex' , 'random' or dictionary with keys 'A' and 'X'
	    default 'convex'
            inital guess of factorization
        eps: float, default 1E-5
            value to avoid divisions by zero
        verbose: int, default 0
           number of iterations with output to stdout
        shape: 2-tuple of int
           shape of input images for calculating neigborhood matrix

        """

        self.k = num_comp
        self.maxcount = kwargs.get('maxcount', 100)
        self.eps = kwargs.get('eps', 1E-5)
        self.verbose = kwargs.get('verbose', 0)
        self.shape = kwargs.get('shape', None)
        self.init = kwargs.get('init', 'convex')
        self.smooth_param = kwargs.get('smooth_param', 0)
        self.sparse_param = kwargs.get("sparse_param", 0)
        self.neg_time = kwargs.get("neg_time", False)
        self.timenorm = kwargs.get('timenorm', lambda x: np.sqrt(np.dot(x, x)))
        self.basenorm = kwargs.get("basenorm", lambda x: 1)
        self.sparse_fct = NORMS[kwargs.get("sparse_fct", 'global_sparse')]

    def nmf(self, Y, A, X):
        __nmf = get_nmf()
        m, n = Y.shape
        k, _ = X.shape
        __nmf(A, X, Y, m, n, k)
        return

    def fit(self, Y):
        """perform NMF of Y until stop criterion """
        self.psi = 1E-12  # numerical stabilization

        Y = np.asfortranarray(Y)
        A, X = self.init_factors(Y)
        # create neighborhood matrix
        if self.smooth_param:
            self.S = self.create_nn_matrix()

        count = 0
        obj_old = 1e99
        nrm_Y = np.linalg.norm(Y)

        if self.verbose:
            print 'init completed'

        self.nmf(Y, A, X)

        obj = np.linalg.norm(Y - np.dot(A, X)) / nrm_Y

        A = np.ascontiguousarray(A)
        X = np.ascontiguousarray(X)

        return A, X, obj

    def init_factors(self, Y):
        """ generate start matrices A, X """

        if self.init == 'random':
            m, n = Y.shape
            A = np.random.rand(m, self.k)
            X = np.zeros((self.k, n))

        elif self.init == 'convex':
            out = convex_cone(Y, self.k)
            X = np.array(out['base'])
            A = np.array(out['timecourses']).T

        elif type(self.init) == dict:
            X = self.init['X']
            A = self.init['A']

        return np.asfortranarray(A), np.asfortranarray(X)

    def create_nn_matrix(self):
        """creates neighborhood matrix"""

        nn_matrix = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                temp = np.zeros(self.shape)
                if i > 0:
                    temp[i - 1, j] = 1
                if i < self.shape[0] - 1:
                    temp[i + 1, j] = 1
                if j > 0:
                    temp[i, j - 1] = 1
                if j < self.shape[1] - 1:
                    temp[i, j + 1] = 1
                nn_matrix.append(1. * temp.flatten() / np.sum(temp))
        return np.array(nn_matrix)



