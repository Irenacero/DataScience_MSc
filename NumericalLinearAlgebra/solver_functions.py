import numpy as np
import scipy.linalg as la
from ex_c1 import newton_step
from build_variables import mkkt_matrix, mkkt_system_ldl, mkkt_system_cholesky


def update_mkkt(Mkkt, s_k, lam_k, n):
    """
    :param Mkkt: mkkt matrix by blocks
    :param s_k: updated s
    :param lam_k: updated lambda
    :return: updated mkkt matrix - two last blocks in the last row are updated by s_k and lam_k
    """
    Mkkt[-1 * 2 * n:, -1 * 2 * n * 2:-1 * 2 * n] = np.diag(s_k)
    Mkkt[-1 * 2 * n:, -1 * 2 * n:] = np.diag(lam_k)

    return Mkkt


def compute_mat_f(G, x, g, C, lam, s, d, A, b, gam):
    """
    :param G: matrix with dimension nxn
    :param x: vector with length n
    :param g: vector with length n
    :param C: matrix with dimension nxm
    :param lam: vector with length m
    :param s: vector with length m
    :param d: vector with length m
    :param A: matrix with dimension nxp. If None, computes the case for A=0
    :param b: vector with length p. If None, computes the case for A=0
    :param gam: vector with length p. If None, computes the case for A=0
    :return: matrix of the function F given the input values
    """
    block_row1 = G.dot(x) + g - A.dot(gam) - C.dot(lam)
    block_row2 = b - A.transpose().dot(x)
    block_row3 = s + d - C.transpose().dot(x)
    block_row4 = s*lam

    F = np.concatenate([block_row1, block_row2, block_row3, block_row4], axis=0)
    return F


def solve_ldl_system(Gstar, Lam_inv_ldl, F, s):
    # Compute factorization:
    m = Lam_inv_ldl.shape[1]
    n = F.size - 2*m
    p = F.size -2*m -n
    LU, D, perm = la.ldl(Gstar)
    # L = LU[perm, :]
    L = LU
    Lt = L.transpose()

    # Compute r* vector:
    r1 = F[:n]
    ra = F[n:n+p]
    r2 = F[-(m + m):-m]
    r3 = F[-m:]
    sys_sol = np.concatenate([-r1, ra, -(r2-Lam_inv_ldl.dot(r3))], axis=0)

    # solve the system:
    y = np.linalg.solve(L, sys_sol)
    z = np.linalg.solve(D, y)
    sub_delta = np.linalg.solve(Lt, z)
    delta_s = Lam_inv_ldl.dot(-r3 - (np.diag(s).dot(sub_delta[-m:])))

    return np.concatenate([sub_delta, delta_s])
    
def solve_cholesky_system(G_chol, C, Sinv_chol, Lam_chol, F):
    # Get dimensions
    n = G_chol.shape[0]
    m = C.shape[1]

    # Split F in r vectors and build right hand vector
    r1 = F[:n]
    r2 = F[-(m + m):-m]
    r3 = F[-m:]
    sys_sol = -r1 - (-C.dot(Sinv_chol).dot(-r3+Lam_chol.dot(r2)))

    # Compute Cholesky factorization
    L = np.linalg.cholesky(G_chol)
    Lt = L.transpose()

    # Solve the systems
    y = np.linalg.solve(L, sys_sol)
    delta_x = np.linalg.solve(Lt, y)
    delta_lam = Sinv_chol.dot(-r3+Lam_chol.dot(r2)) - Sinv_chol.dot(Lam_chol).dot(C.transpose()).dot(delta_x)
    delta_s = -r2 + C.transpose().dot(delta_x)

    return np.concatenate([delta_x, delta_lam, delta_s], axis=0)
    
def result_function(sol_n, G, g):

    result = 0.5*sol_n.transpose().dot(G).dot(sol_n) + g.transpose().dot(sol_n) 
    
    return result
    
def compute_cond_number(matrix):

    cond_number = np.linalg.cond(matrix)
    return (cond_number)

def solve_kkt_system(G, x_0, g, C, lam_0, s_0, d, A=None, gam_0=None, b=None,
                     tol=1e-16, max_iter=100, factorization=None):
    n = G.shape[0]
    m = C.shape[1]

    # Define empty A, b and gamma for if we solve the test problem
    if (A is None) or (b is None) or (gam_0 is None):
        A = np.array([]).reshape(n, 0)
        b = np.array([])
        gam_0 = np.array([])
    p = A.shape[1]

    # define the Mkkt function:
    if factorization is None:
        Mkkt = mkkt_matrix(G=G, C=C, s=s_0, lam=lam_0, A=A)
    elif factorization == 'ldl':
        Gstar, Lam_inv_ldl = mkkt_system_ldl(G=G, C=C, s=s_0, lam=lam_0, A=A)
    elif factorization == 'cholesky':
        G_chol, Sinv_chol, Lam_chol = mkkt_system_cholesky(G=G, C=C, s=s_0, lam=lam_0)

    # p = gam_0.size
    F = compute_mat_f(G=G, x=x_0, g=g, C=C, lam=lam_0, s=s_0, d=d,
                      A=A, b=b, gam=gam_0)

    x_k = x_0.copy()
    gam_k = gam_0.copy()
    lam_k = lam_0.copy()
    s_k = s_0.copy()
    e = np.repeat(1., m)
    mu = np.dot(s_0, lam_0)/m

    iters = 0
    while (iters < max_iter and
           np.linalg.norm(F[:n]) > tol and
           np.linalg.norm(F[-(m + m):-m]) > tol and
           np.linalg.norm(F[-m:]) > tol and
           abs(mu) > tol):

        # Predictor substep:+
        if factorization is None:
            delta = np.linalg.solve(Mkkt, -F)
        elif factorization == 'ldl':
            delta = solve_ldl_system(Gstar=Gstar, Lam_inv_ldl=Lam_inv_ldl, F=F, s=s_k)
        elif factorization == 'cholesky':
            delta = solve_cholesky_system(G_chol, C, Sinv_chol, Lam_chol, F)

        # Step-size correction substep:
        delta_lam = delta[-(m + m):-m]
        delta_s = delta[-m:]
        alp = newton_step(lam_k, delta_lam, s_k, delta_s)

        # compute mu, muhat, sigma
        mu = (s_k.transpose().dot(lam_k)) / m
        muhat = (s_k + alp*delta_s).dot((lam_k + alp*delta_lam)) / m
        sigma = (muhat / mu) ** 3

        # corrector step:
        rs_update = F[-m:] + np.diag(delta_s).dot(np.diag(delta_lam)).dot(e) - sigma*mu*(e)
        F_2 = np.concatenate([F[:-m], rs_update], axis=0)

        if factorization is None:
            delta2 = np.linalg.solve(Mkkt, -F_2)
        elif factorization == 'ldl':
            delta2 = solve_ldl_system(Gstar=Gstar, Lam_inv_ldl=Lam_inv_ldl, F=F_2, s=s_k)
        elif factorization == 'cholesky':
            delta2 = solve_cholesky_system(G_chol, C, Sinv_chol, Lam_chol, F_2)

        # Step-size correction substep:
        dlamb2 = delta2[-(m + m):-m]
        ds2 = delta2[-m:]
        alp2 = newton_step(lam_k, dlamb2, s_k, ds2)

        # update step:
        z_new = np.concatenate([x_k, gam_k, lam_k, s_k]) + 0.95 * alp2 * delta2
        x_k = z_new[:n]
        gam_k = z_new[n:n+p]
        lam_k = z_new[-(m + m):-m]
        s_k = z_new[-m:]

        if factorization is None:
            Mkkt = update_mkkt(Mkkt=Mkkt, s_k=s_k, lam_k=lam_k, n=n)
        elif factorization == 'ldl':
            Gstar, Lam_inv_ldl = mkkt_system_ldl(G=G, C=C, s=s_k, lam=lam_k, A=A)
        elif factorization == 'cholesky':
            G_chol, Sinv_chol, Lam_chol = mkkt_system_cholesky(G=G, C=C, s=s_k, lam=lam_k)

        F = compute_mat_f(G, x_k, g, C, lam_k, s_k, d, A, b, gam_k)
        iters += 1
    
    
    # compute condition number and relative error of the solution:
    if factorization is None:
        cond_number = compute_cond_number(matrix = Mkkt)
    elif factorization == 'ldl':
        cond_number = compute_cond_number(matrix = Gstar)
    elif factorization == 'cholesky':
        cond_number = compute_cond_number(matrix = G_chol)


    return x_k, iters, g, cond_number
    
