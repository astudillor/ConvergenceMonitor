from __future__ import division, print_function, absolute_import
from ConvergenceMonitor import ConvergenceMonitor

import scipy.sparse.linalg as la
import scipy.io as io
import numpy as np

# Copy from scypy demo_lgmres
def setup_linear_system():
    problem = "SPARSKIT/drivcav/e05r0200"
    server = 'ftp://math.nist.gov/pub/MatrixMarket2/'
    mm = np.lib._datasource.Repository(server)
    f = mm.open('%s.mtx.gz' % problem)
    A = io.mmread(f).tocsr()
    f.close()

    f = mm.open('%s_rhs1.mtx.gz' % problem)
    b = np.array(io.mmread(f)).ravel()
    f.close()
    return A, b


def main():
    """docstring for main"""
    A, b = setup_linear_system()

    # using bicg
    conv = ConvergenceMonitor(action=lambda x: b - A.dot(x), increment=2)
    x, info = la.bicg(A, b, maxiter=1000000, callback=conv)
    conv.scale(1.0 / np.linalg.norm(b))
    print('bicg ', end='')
    conv.printInfo()

    # using bicgstab
    conv = ConvergenceMonitor(action=lambda x: b - A.dot(x), increment=2)
    x, info = la.bicgstab(A, b, maxiter=1000000, callback=conv)
    conv.scale(1.0 / np.linalg.norm(b))
    print('bicgstab ', end='')
    conv.printInfo()

    # using gmres
    conv = ConvergenceMonitor()
    x, info = la.gmres(A, b, maxiter=1000000, restart=250, callback=conv)
    conv.scale(1.0 / np.linalg.norm(b))
    print('full GMRES ', end='')
    conv.printInfo()


if __name__ == '__main__':
    main()
