import numpy as np
from scipy.stats import chi2, norm, rankdata
from scipy.linalg import eigh
from scipy.spatial.distance import mahalanobis
import numpy as np
from scipy.stats import chi2

from makeplot import makeplot


def weightmecov(data, weights, centered=0):
    if np.any(weights < 0):
        raise ValueError('The weights are negative')

    if weights.ndim == 1:
        weights = weights[:, np.newaxis]

    q = np.where(weights > 0)[0]

    if not centered:
        wmean = np.sum(weights * data, axis=0) / np.sum(weights)
    else:
        wmean = np.zeros(data.shape[1])

    centered_data = data[q, :] - wmean
    wcov = (centered_data.T @ (weights[q] * centered_data)) / (np.sum(weights**2) - 1)

    return wmean, wcov


def classSVD(X, all=0, centered=0):
    n, p = X.shape

    if n == 1:
        raise ValueError('The sample size is 1. No SVD can be performed.')

    if not centered:
        cX = np.mean(X, axis=0)
        centerX = X - cX
    else:
        cX = np.zeros(p)
        centerX = X

    # Perform SVD
    U, S, loadings = np.linalg.svd(centerX / np.sqrt(n - 1), full_matrices=False)
    
    # Eigenvalues are the squares of the singular values
    eigenvalues = S**2
    
    # Tolerance for determining the rank
    tol = max(n, p) * np.finfo(float).eps * eigenvalues[0]
    r = np.sum(eigenvalues > tol)

    if not all:
        L = eigenvalues[:r]
        P = loadings[:, :r]
    else:
        L = eigenvalues
        P = loadings

    T = centerX @ P

    return P, T, L, r, centerX, cX


def unimcd(y, h, centered=0):
    ncas = len(y)
    len_ = ncas - h + 1
    xorig = y.copy()

    if len_ == 1:
        if centered == 1:
            tmcd = 0
        else:
            tmcd = np.mean(y)
        smcd = np.sqrt(np.var(y, ddof=1))
        weights = np.ones(len(y))
    else:
        if centered == 1:
            y_sorted = np.sort(np.abs(y))
            I = np.argsort(np.abs(y))
            Hopt = I[:h]
            initmean = 0
            initcov = np.sum(y_sorted[:h] ** 2) / (h - 1)
        else:
            y_sorted = np.sort(y)
            I = np.argsort(y)
            ay = np.zeros(len_)
            ay[0] = np.sum(y_sorted[:h])
            for samp in range(1, len_):
                ay[samp] = ay[samp - 1] - y_sorted[samp - 1] + y_sorted[samp + h - 1]
            ay2 = ay ** 2 / h
            sq = np.zeros(len_)
            sq[0] = np.sum(y_sorted[:h] ** 2) - ay2[0]
            for samp in range(1, len_):
                sq[samp] = sq[samp - 1] - y_sorted[samp - 1] ** 2 + y_sorted[samp + h - 1] ** 2 - ay2[samp] + ay2[samp - 1]
            sqmin = np.min(sq)
            ii = np.where(sq == sqmin)[0][0]
            Hopt = I[ii:ii + h]
            ndup = len(np.where(sq == sqmin)[0])
            slutn = ay[ii:ii + ndup]
            initmean = slutn[int(np.floor((ndup + 1) / 2))] / h  # initial mean
            initcov = sqmin / (h - 1)  # initial variance

        # calculating consistency factor
        res = (xorig - initmean) ** 2 / initcov
        sortres = np.sort(res)
        factor = sortres[h - 1] / chi2.ppf(h / ncas, 1)
        initcov = factor * initcov
        res = (xorig - initmean) ** 2 / initcov  # raw_robdist^2
        quantile = chi2.ppf(0.975, 1)
        weights = res <= quantile  # raw-weights
        rawrd = np.sqrt(res)

        # reweighting procedure
        if weights.shape[0] != y.shape[0]:
            weights = weights.T
        if centered == 1:
            tmcd = 0
        else:
            tmcd = np.sum(xorig * weights) / np.sum(weights)
        smcd = np.sqrt(np.sum((xorig - tmcd) ** 2 * weights) / (np.sum(weights) - 1))

    return tmcd, smcd, weights, initmean, initcov, rawrd, Hopt


def quanf(alfa, n, rk):
    return int(np.floor(2 * np.floor((n + rk + 1) / 2) - n + 2 * (n - np.floor((n + rk + 1) / 2)) * alfa))

def W_scale(x):
    c = 4.5
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, p = x.shape
    Wc = lambda x, c: (1 - (x / c) ** 2) ** 2 * (np.abs(x) < c)
    sigma0 = np.median(np.abs(x - np.median(x, axis=0)), axis=0)
    w = Wc((x - np.median(x, axis=0)) / sigma0, c)
    loc = np.sum(x * w, axis=0) / np.sum(w, axis=0)

    c = 3
    rc = lambda x, c: np.minimum(x ** 2, c ** 2)
    sigma0 = np.median(np.abs(x - np.median(x, axis=0)), axis=0)
    b = c * norm.ppf(3 / 4)
    nes = n * (2 * ((1 - b ** 2) * norm.cdf(b) - b * norm.pdf(b) + b ** 2) - 1)
    scale = sigma0 ** 2 / nes * np.sum(rc((x - loc) / sigma0, c), axis=0)
    return np.sqrt(scale)

def qn(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, p = x.shape
    if n == 1 and p != 1:
        x = x.T
        n, p = x.shape
    h = int(np.floor(n / 2) + 1)
    Qn = np.zeros(p)
    for j in range(p):
        dist = np.abs(np.subtract.outer(x[:, j], x[:, j]))
        d = np.sort(dist[np.triu_indices(n, 1)])
        Qn[j] = 2.2219 * d[h * (h - 1) // 2]
    if n <= 9:
        dn = {2: 0.399, 3: 0.994, 4: 0.512, 5: 0.844, 6: 0.611, 7: 0.857, 8: 0.669, 9: 0.872}[n]
    else:
        if n % 2 == 1:
            dn = n / (n + 1.4)  
        else:
            dn = n / (n + 3.8)
    return dn * Qn

def ogkscatter(x, scales):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n, p = x.shape
    U = np.eye(p)
    for i in range(p+1):
        sYi = x[:, i]
        for j in range(i):
            sYj = x[:, j]
            sY = sYi + sYj
            dY = sYi - sYj
            U[i, j] = 0.25 * (scales(sY) ** 2 - scales(dY) ** 2)
    U = np.tril(U, -1) + U.T
    P, L = eigh(U)
    return P, L

def initset(data, scales, P, n, p):
    lambda_ = scales(data @ P)
    sqrtcov = P @ np.diag(lambda_) @ P.T
    sqrtinvcov = P @ np.diag(1.0 / lambda_) @ P.T
    estloc = np.median(data @ sqrtinvcov, axis=0) @ sqrtcov
    centeredx = (data - estloc) @ P
    dist = np.array([mahalanobis(centeredx[i], np.zeros(p), np.diag(lambda_) ** 2) for i in range(n)])
    ind = np.argsort(dist)
    return ind

def DetMCD(x, **kwargs):
    if (len(kwargs) % 2) != 0:
        raise ValueError('The number of input arguments should be odd!')

    data = x
    rew = {'plane': []}
    raw = {'cor': []}
    rew['cor'] = []
    if data.shape[0] == 1:
        data = data.T

    ok = np.all(np.isfinite(data), axis=1)
    data = data[ok, :]
    xx = data
    n, p = data.shape

    if n == 0:
        raise ValueError('All observations have missing or infinite values.')
    if n < p:
        raise ValueError('Need at least (number of variables) observations.')

    hmin = quanf(0.5, n, p)
    h = quanf(0.75, n, p)
    default = {'alpha': 0.75, 'h': h, 'plots': 1, 'scale_est': 1, 'cor': 0, 'hsetsfull': np.nan, 'classic': 0}
    options = default.copy()
    chklist = []
    i = 0

    if len(kwargs) > 0:
        for key in kwargs.keys():
            chklist.append(key)
        dummy = sum([chklist.count('h'), 2 * chklist.count('alpha')])
        if dummy == 0:
            alfa = options['alpha']
            h = options['h']
        elif dummy == 3:
            raise ValueError('Both input arguments alpha and h are provided. Only one is required.')

        for key, value in kwargs.items():
            if key in options:
                options[key] = value

        if dummy == 1:
            if options['h'] < hmin:
                print(f'Warning: The MCD must cover at least {hmin} observations.')
                print(f'The value of h is set equal to {hmin}')
                options['h'] = hmin
            elif options['h'] > n:
                raise ValueError('h is greater than the number of non-missings and non-infinites.')
            elif options['h'] < p:
                raise ValueError(f'h should be larger than the dimension {p}.')
            options['alpha'] = options['h'] / n
        elif dummy == 2:
            if options['alpha'] < 0.5:
                options['alpha'] = 0.5
                print('Attention (detmcd.m): Alpha should be larger than 0.5. It is set to 0.5.')
            if options['alpha'] > 1:
                options['alpha'] = 0.75
                print('Attention (detmcd.m): Alpha should be smaller than 1. It is set to 0.75.')
            options['h'] = quanf(options['alpha'], n, p)

    h = options['h']
    plots = options['plots']
    alfa = options['alpha']
    scale_est = options['scale_est']
    hsetsfull = options['hsetsfull']
    cor = options['cor']

    if scale_est == 1:
        if n >= 1000:
            scales = W_scale  
        else:
            scales = qn
    elif scale_est == 2:
        scales = qn
    elif scale_est == 3:
        scales = W_scale

    med = np.median(data, axis=0)
    sca = scales(data)
    if np.any(sca < np.finfo(float).eps):
        raise ValueError(f'DetMCD.m: Variable {np.where(sca < np.finfo(float).eps)[0][0]} has zero scale. MCD cannot be computed.')
    data = (data - med) / sca
    cutoff_rd = np.sqrt(chi2.ppf(0.975, p))
    cutoff_md = cutoff_rd
    clmean = np.mean(data, axis=0)
    clcov = np.cov(data, rowvar=False)

    if p == 1:
        rew_center, rewsca, weights, raw_center, raw_cov, raw_rd, Hopt = unimcd(data, h)
        rew['Hsubsets'] = {'Hopt': Hopt.T}
        raw['cov'] = raw_cov * sca ** 2
        raw['objective'] = raw['cov']
        raw['center'] = raw_center * sca + med
        raw['cutoff'] = cutoff_rd
        raw['wt'] = weights
        rew['cov'] = rewsca ** 2
        mah = (data - rew_center) ** 2 / rew['cov']
        rew['rd'] = np.sqrt(mah)
        rew['flag'] = rew['rd'] <= cutoff_rd
        rew['cutoff'] = cutoff_rd
        rew['center'] = rew_center * sca + med
        rew['cov'] = rew['cov'] * sca ** 2
        rew['mahalanobis'] = np.abs(data - clmean) / np.sqrt(clcov)

        if options['classic'] == 1:
            classic = {
                'cov': clcov,
                'center': clmean,
                'md': rew['mahalanobis'],
                'flag': rew['mahalanobis'] <= cutoff_md,
                'class': 'COV'
            }
        else:
            classic = 0

        rew = {
            'center': rew['center'],
            'cov': rew['cov'],
            'cor': rew['cor'],
            'h': h,
            'Hsubsets': rew['Hsubsets'],
            'alpha': alfa,
            'rd': rew['rd'],
            'cutoff': cutoff_rd,
            'flag': rew['flag'],
            'plane': rew['plane'],
            'class': 'MCDCOV',
            'md': rew['mahalanobis'],
            'classic': classic
        }
        raw = {
            'center': raw['center'],
            'cov': raw['cov'],
            'cor': raw['cor'],
            'objective': raw['objective'],
            'rd': raw['rd'],
            'wt': raw['wt']
        }
        if plots:
            makeplot(rew)
        return rew, raw

    if np.isnan(hsetsfull).all():
        hsetsfull = np.full((6, n), np.nan)
        y1 = np.tanh(data)
        R1 = np.corrcoef(y1, rowvar=False)
        P, L = eigh(R1)
        ind = initset(data, scales, P, n, p)
        hsetsfull[0, :] = ind

        y2 = np.apply_along_axis(rankdata, 0, data)
        R2 = np.corrcoef(y2, rowvar=False)
        P, L = eigh(R2)
        ind = initset(data, scales, P, n, p)
        hsetsfull[1, :] = ind

        y3 = norm.ppf((y2 - 1 / 3) / (n + 1 / 3))
        R3 = np.corrcoef(y3, rowvar=False)
        P, L = eigh(R3)
        ind = initset(data, scales, P, n, p)
        hsetsfull[2, :] = ind

        znorm = np.sqrt(np.sum(data ** 2, axis=1))
        ii = znorm > np.finfo(float).eps
        zznorm = data.copy()
        zznorm[ii, :] = data[ii, :] / znorm[ii][:, np.newaxis]
        SCM = (zznorm.T @ zznorm) / (n - 1)
        P, L = eigh(SCM)
        ind = initset(data, scales, P, n, p)
        hsetsfull[3, :] = ind

        ind5 = np.argsort(znorm)
        half = int(np.ceil(n / 2))
        Hinit = ind5[:half]
        covx = np.cov(data[Hinit, :], rowvar=False)
        P, L = eigh(covx)
        ind = initset(data, scales, P, n, p)
        hsetsfull[4, :] = ind

        P = ogkscatter(data, scales)
        ind = initset(data, scales, P, n, p)
        hsetsfull[5, :] = ind

    Isets = hsetsfull[:, :half]
    nIsets = Isets.shape[0]

    for k in range(nIsets):
        xk = data[Isets[k, :], :]
        P, T, L, r, centerX, meanvct = classSVD(xk)
        if r < p:
            raise ValueError('DetMCD.m: More than half of the observations lie on a hyperplane.')
        score = (data - meanvct) @ P
        dist = np.array([mahalanobis(score[i], np.zeros(score.shape[1]), np.diag(L)) for i in range(n)])
        sortdist = np.argsort(dist)
        hsetsfull[k, :] = sortdist

    Hsets = hsetsfull[:, :h]
    raw['wt'] = np.full(len(ok), np.nan)
    raw['rd'] = np.full(len(ok), np.nan)
    rew['rd'] = np.full(len(ok), np.nan)
    rew['mahalanobis'] = np.full(len(ok), np.nan)
    rew['flag'] = np.full(len(ok), np.nan)

    csteps = 100
    prevdet = 0
    bestobj = np.inf
    cutoff_rd = np.sqrt(chi2.ppf(0.975, p))
    cutoff_md = cutoff_rd

    for i in range(Hsets.shape[0]):
        for j in range(csteps):
            if j == 1:
                obs_in_set = Hsets[i, :]
            else:
                score = (data - meanvct) @ P
                mah = np.array([mahalanobis(score[k], np.zeros(score.shape[1]), np.diag(L)) for k in range(n)])
                sortdist = np.argsort(mah)
                obs_in_set = sortdist[:h]
            P, T, L, r, centerX, meanvct = classSVD(data[obs_in_set, :])
            obj = np.prod(L)

            if r < p:
                raise ValueError('DetMCD.m: More than h of the observations lie on a hyperplane.')
            if j >= 2 and obj == prevdet:
                break
            prevdet = obj

        if obj < bestobj:
            bestset = obs_in_set
            bestobj = obj
            initmean = meanvct
            initcov = P @ np.diag(L) @ P.T
            raw['initcov'] = initcov
            rew['Hsubsets'] = {'Hopt': bestset, 'i': i}
        rew['Hsubsets']['csteps'] = j

    P, T, L, r, centerX, meanvct = classSVD(data[bestset, :])
    mah = np.array([mahalanobis((data[k] - meanvct) @ P, np.zeros(P.shape[1]), np.diag(L)) for k in range(n)])
    sortmah = np.sort(mah)

    factor = sortmah[h - 1] / chi2.ppf(h / n, p)
    raw['cov'] = factor * initcov
    raw['cov'] = raw['cov'] * sca[:, np.newaxis] * sca
    raw['center'] = initmean * sca + med
    raw['objective'] = bestobj * np.prod(sca) ** 2
    mah = mah / factor
    raw['rd'] = np.sqrt(mah)
    weights = raw['rd'] <= cutoff_rd
    raw['wt'] = weights
    rew['center'], rew['cov'] = weightmecov(data, weights)
    trcov = rew['cov'] * sca[:, np.newaxis] * sca
    trcenter = rew['center'] * sca + med

    mah = np.array([mahalanobis(data[k], rew['center'], rew['cov']) for k in range(n)])
    rew['rd'] = np.sqrt(mah)
    rew['flag'] = rew['rd'] <= cutoff_rd

    rew['mahalanobis'] = np.array([mahalanobis(data[k], clmean, clcov) for k in range(n)])
    rawo = raw
    reso = rew

    if options['classic'] == 1:
        classic = {
            'center': clmean * sca + med,
            'cov': clcov * sca[:, np.newaxis] * sca,
            'md': rew['mahalanobis'],
            'flag': rew['mahalanobis'] <= cutoff_md,
            'class': 'COV'
        }
        if cor == 1:
            diagcl = np.sqrt(np.diag(clcov))
            classic['cor'] = clcov / (diagcl[:, np.newaxis] * diagcl)
    else:
        classic = 0

    if cor == 1:
        diagraw = np.sqrt(np.diag(raw['cov']))
        raw['cor'] = raw['cov'] / (diagraw[:, np.newaxis] * diagraw)
        diagrew = np.sqrt(np.diag(rew['cov']))
        rew['cor'] = rew['cov'] / (diagrew[:, np.newaxis] * diagrew)

    rew = {
        'center': trcenter,
        'cov': trcov,
        'cor': rew['cor'],
        'h': h,
        'Hsubsets': reso['Hsubsets'],
        'alpha': alfa,
        'rd': reso['rd'],
        'cutoff': cutoff_rd,
        'flag': reso['flag'],
        'plane': reso['plane'],
        'class': 'MCDCOV',
        'md': reso['mahalanobis'],
        'classic': classic,
        'X': xx
    }
    raw = {
        'center': rawo['center'],
        'cov': rawo['cov'],
        'cor': raw['cor'],
        'objective': rawo['objective'],
        'rd': rawo['rd'],
        'cutoff': cutoff_rd,
        'wt': rawo['wt']
    }

    if data.shape[1] != 2:
        del rew['X']

    if plots:
        makeplot(rew)

    return rew, raw

# example usage of DetMCD
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(0)
    n = 100
    p = 2
    x = np.random.randn(n, p)
    x[::2] = x[::2] + 2
    x[1::2] = x[1::2] - 2

    rew, raw = DetMCD(x, h=50, plots=1)
    plt.show()
