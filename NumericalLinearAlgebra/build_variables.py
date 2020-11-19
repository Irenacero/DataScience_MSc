import numpy as np


def create_test_problem(n):
    """
    :param n: dimension of the solution of the KKT system
    :return: matrices and vectors in the KKT system given the test problem.
    Values are returned in the following order G, C, d, g, x, s_0, lam_0
    """
    m = 2*n
    G = np.identity(n)
    C = np.concatenate((np.identity(n), -np.identity(n)), axis=1)
    d = -10*np.ones(m)
    g = np.random.randn(n)
    x_0 = np.zeros(n)
    s_0, lam_0 = np.ones(m), np.ones(m)

    return G, x_0, g, C, lam_0, s_0, d


def mkkt_matrix(G, C, s, lam, A):
    """
    :param G: matrix with dimension nxn
    :param A: matrix with dimension nxp
    :param C: matrix with dimension nxm
    :param s: vector with length m
    :param lam: vector with length m
    :return: mkkt matrix
    """
    n = G.shape[0]
    m = C.shape[1]
    p = A.shape[1]

    Mkkt_row1 = np.concatenate([G, -A, -C, np.zeros((n, m))],
                               axis=1)
    Mkkt_row2 = np.concatenate([(-A).transpose(), np.zeros((p, p)), np.zeros((p, m)),
                                np.zeros((p, m))], axis=1)
    Mkkt_row3 = np.concatenate([(-C).transpose(), np.zeros((m, p)), np.zeros((m, m)),
                                np.identity(m)], axis=1)
    Mkkt_row4 = np.concatenate([np.zeros((m, n)), np.zeros((m, p)),
                                np.diag(s), np.diag(lam)], axis=1)

    return np.concatenate([Mkkt_row1, Mkkt_row2, Mkkt_row3, Mkkt_row4], axis=0)


def mkkt_system_ldl(A, G, C, s, lam):
    n = G.shape[0]
    m = C.shape[1]
    p = A.shape[1]
    
    Mkkt_row1 = np.concatenate([G, -A, -C], axis=1)
    Mkkt_row2 = np.concatenate([-A.transpose(), np.zeros((p,p)), np.zeros((p,m))], axis = 1)
    Mkkt_row3 = np.concatenate([(-C).transpose(), np.zeros((m, p)),(np.diag(lam**(-1)*(-1))).dot(np.diag(s))], axis=1)
    return np.concatenate([Mkkt_row1, Mkkt_row2, Mkkt_row3], axis=0), \
		   np.diag(lam**(-1))

   
def mkkt_system_cholesky(G, C, s, lam):
    """
    :param G: nxn G matrix of the Mkkt matrix
    :param C: nxm C matrix of the Mkkt matrix
    :param s: vector of dimension m with s_i
    :param lam: vector of dimension m with s_i
    :return: G' matrix for the Cholesky factorization, -CS^-1 matrix, Lambda matrix
    """
    return G + C.dot(np.diag(s**(-1))).dot(np.diag(lam)).dot(C.transpose()), \
           np.diag(s**(-1)), \
           np.diag(lam)

