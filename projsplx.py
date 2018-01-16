"""
    Project a vector to standard probability simplex
    x = projsplx(y)
    where y in R^n is given, and x is the closest point to y that satisifes x>=0 and sum(x)=1
    
    @Xiaojing Ye
    
    Derivation provided in
    Projection onto a simplex, Y. Chen and X. Ye, arXiv:1101.6081, 2011.
    Link to PDF: https://arxiv.org/pdf/1101.6081.pdf
"""

import numpy as np

def projsplx(y):
    
    nrow, ncol = np.shape(y)
    m = max([nrow, ncol])
    bget = False
    
    s = sorted(np.array(y), reverse=True)
    s = np.reshape(s, [m,1])
    tmpsum = 0.0

    for ii in range(m-1):
        tmpsum = tmpsum + s[ii]
        tmax = (tmpsum - 1.0) / (ii+1.0)
        if tmax >= s[ii+1]:
            bget = True
            break

    if bget == False:
        tmax = (tmpsum + s[m-1] - 1.0) / m

    x = np.maximum(np.array(y) - tmax, np.zeros([nrow,ncol]))
    x = np.reshape(x, [nrow, ncol])

    return x


def proj_l1(y):

    sgn = np.sign(y)
    x = abs(projsplx(y))
    x = sgn * x

    return x