import numpy as np


def newton_step(lamb0, dlamb, s0, ds):
    alp = 1
    idx_lamb0 = np.array(np.where(dlamb < 0))
    if idx_lamb0.size > 0:
        alp = min(alp, np.min(-lamb0[idx_lamb0]/dlamb[idx_lamb0]))
    idx_s0 = np.array(np.where(ds < 0))
    if idx_s0.size > 0:
        alp = min(alp, np.min(-s0[idx_s0]/ds[idx_s0]))
    return alp
