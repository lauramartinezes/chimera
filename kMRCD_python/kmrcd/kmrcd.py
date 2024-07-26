import numpy as np
from scipy.linalg import svd
from scipy.optimize import fsolve
from scipy.stats import norm

# Importing necessary utilities and kernels
from kmrcd.utils import Utils
from kernels.lin_kernel import LinKernel
from kernels.rbf_kernel import RbfKernel
from kernels.auto_rbf_kernel import AutoRbfKernel
from standarization.unimcd import unimcd

class kMRCD:
    def __init__(self, kModel=None):
        self.kModel = kModel if kModel else LinKernel()
        self.cStepIterationsAllowed = 100
        self.maxcond = 50

    def runAlgorithm(self, x, alpha):
        if not (0.5 <= alpha <= 1):
            raise ValueError('The percentage of regular observations, alpha, should be in [0.5-1]')

        if isinstance(self.kModel, RbfKernel):
            print('Warning: kMRCD switches to AutoRbfKernel in case a RbfKernel was specified!!')
            self.kModel = AutoRbfKernel(x)

        K = self.kModel.compute(x, x)
        n, p = x.shape

        # Initial estimators
        solution = {}
        solution['SDO'] = {}
        solution['SpatialRank'] = {}
        solution['SpatialMedian'] = {}
        solution['SSCM'] = {}

        solution['SDO']['outlyingnessIndices'], _ = Utils.SDO(K, alpha)
        solution['SpatialRank']['outlyingnessIndices'], _ = Utils.SpatialRank(K, alpha)
        solution['SpatialMedian']['outlyingnessIndices'], _, _ = Utils.SpatialMedianEstimator(K, alpha)
        solution['SSCM']['outlyingnessIndices'], _ = Utils.SSCM(K)

        scfac = Utils.MCDcons(p, alpha)
        rhoL = np.full(len(solution), np.nan)

        for idx, (key, value) in enumerate(solution.items()):
            hsubsetIndices = value['outlyingnessIndices'][:int(np.ceil(n * alpha))]
            solution[key]['hsubsetIndices'] = hsubsetIndices

            Kx = self.kModel.compute(x[hsubsetIndices], x[hsubsetIndices])
            s = svd(Utils.center(Kx), compute_uv=False)
            e_min = np.min(s)
            e_max = np.max(s)
            fncond = lambda rho: (len(hsubsetIndices) * rho + (1 - rho) * scfac * e_max) / (len(hsubsetIndices) * rho + (1 - rho) * scfac * e_min) - self.maxcond

            try:
                rhoL[idx] = fsolve(fncond, [1e-6, 0.99])[0]
            except:
                grid = np.linspace(1e-6, 1 - 1e-6, 1000)
                objgrid = np.abs([fncond(r) for r in grid])
                rhoL[idx] = grid[np.argmin(objgrid)]

        rho = np.max(rhoL[rhoL <= np.max([0.1, np.median(rhoL)])])

        for key, value in solution.items():
            for iteration in range(self.cStepIterationsAllowed):
                hSubset = value['hsubsetIndices']
                Kx = self.kModel.compute(x[hSubset], x[hSubset])
                nx = Kx.shape[0]
                Kt = self.kModel.compute(x, x[hSubset])
                Kc = Utils.center(Kx)
                Kt_c = Utils.center(Kx, Kt)
                Ktt_diag = np.diag(K)
                Kxx = Ktt_diag - (2 / len(hSubset)) * np.sum(Kt, axis=1) + (1 / len(hSubset)**2) * np.sum(Kx)
                denominator = (1 - rho) * scfac * Kc + nx * rho * np.eye(nx)
                denominator_inv = np.linalg.inv(denominator)
                Kt_c_divided = np.dot(Kt_c, denominator_inv)
                smd = (1 / rho) * (Kxx - (1 - rho) * scfac * np.sum((Kt_c_divided * Kt_c), axis=1))
                
                indices = np.argsort(smd)
                solution[key]['hsubsetIndices'] = indices[:nx]
                
                if not np.setdiff1d(hSubset, solution[key]['hsubsetIndices']).size:
                    print(f'Convergence at iteration {iteration}, {key}')
                    sigma = svd(Kc, compute_uv=False)
                    sigma = (1 - rho) * scfac * sigma + len(solution[key]['hsubsetIndices']) * rho
                    solution[key]['obj'] = np.sum(np.log(sigma))
                    solution[key]['smd'] = smd
                    break
            assert iteration<self.cStepIterationsAllowed, 'no C-step convergence'

        min_idx = np.argmin([sol['obj'] for sol in solution.values()])
        solution_name = list(solution.keys())[min_idx]
        solution = list(solution.values())[min_idx]
        print(f'-> Best estimator is {solution_name}')

        solution['name'] = solution_name
        solution['rho'] = rho
        solution['scfac'] = scfac

        solution['rd'] = np.maximum(np.sqrt(solution['smd']), 0)
        solution['ld'] = np.log(0.1 + solution['rd'])
        tmcd, smcd, _, _, _, _, _ = unimcd(solution['ld'], len(solution['hsubsetIndices']))
        solution['cutoff'] = np.exp(tmcd + norm.ppf(0.995) * smcd) - 0.1
        solution['flaggedOutlierIndices'] = np.where(solution['rd'] > solution['cutoff'])[0]

        return solution
