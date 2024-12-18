import matplotlib.pyplot as plt
import numpy as np

def makeplot(out, **kwargs):
    def whichplot(out, plotn, attrib, exitno, labsd, labod, labresd, labmcd, lablts, labclus, classic, classicplots):
        # Implement the plotting logic here
        pass

    # Initialization
    if isinstance(out, dict):
        innames = out.keys()
        if 'class' in innames:
            attrib = out['class']
        else:
            raise ValueError('The method had no class identifier.')
        if 'classic' in innames:
            classic = out['classic']
        elif attrib in {'CPCA', 'CPCR', 'CSIMPLS', 'MLR', 'CDA', 'LS', 'CSIMCA'}:
            classic = 1
        else:
            classic = 0
    else:
        raise ValueError('The first input argument is not a structure.')

    counter = 1
    default = {
        'nameplot': 0, 'labsd': 3, 'labod': 3, 'labresd': 3, 'labmcd': 3, 'lablts': 3, 'labclus': [], 'classic': 1
    }
    options = default.copy()
    chklist = []
    i = 1

    if kwargs:
        for key in kwargs.keys():
            chklist.append(key)
        while counter <= len(default):
            if list(default.keys())[counter - 1] in chklist:
                options[list(default.keys())[counter - 1]] = kwargs[list(default.keys())[counter - 1]]
            counter += 1

        choice = options['nameplot']
        if choice != 0:
            ask = 0
        else:
            ask = 1  # menu of plots
        labsd = options['labsd']
        labod = options['labod']
        labresd = options['labresd']
        labmcd = options['labmcd']
        lablts = options['lablts']
        labclus = options['labclus']
        classicplots = options['classic']
        if not isinstance(classic, dict) and options['classic'] == 1 and attrib in {'RAPCA', 'ROBPCA', 'RPCR', 'LTS', 'MCDREG', 'RSIMPLS', 'RDA', 'RSIMCA'}:
            print('The classical output is not available. Only robust plots will be shown.\nPlease rerun the preceding analysis with the option "classic" set to 1 if the classical plots are required.')
            classicplots = 0
    else:
        ask = 1  # menu of plots
        choice = 0
        labsd = 3
        labod = 3
        labresd = 3
        labmcd = 3
        lablts = 3
        labclus = []
        if not isinstance(classic, dict) and options['classic'] == 1 and attrib in {'RAPCA', 'ROBPCA', 'RPCR', 'LTS', 'MCDREG', 'RSIMPLS', 'RDA', 'RSIMCA', 'MCDCOV'}:
            print('The classical output is not available. Only robust plots will be shown.\nPlease rerun the preceding analysis with the option "classic" set to 1 if the classical plots are required.')
            classicplots = 0
        else:
            classicplots = 1

    # to initialize the correct menu of plots
    exitno = 0
    if attrib == 'CPCA':
        exitno = 4
        if choice in {'regdiag', '3ddiag', 'robdist', 'qqmcd', 'dd', 'ellipse', 'resfit', 'resindex', 'qqlts', 'diag', 'scatter', 'da', 'simca'}:
            raise ValueError('That kind of plot is not available for this method.')
    elif attrib == 'CPCR':
        exitno = 6
        if choice in {'robdist', 'qqmcd', 'dd', 'ellipse', 'resfit', 'resindex', 'qqlts', 'diag', 'scatter', 'da', 'simca'}:
            raise ValueError('That kind of plot is not available for this method.')
    elif attrib == 'LS':
        exitno = 7
        if choice in {'scree', 'pcadiag', '3ddiag', 'robdist', 'qqmcd', 'dd', 'ellipse', 'simca', 'da'}:
            raise ValueError('That kind of plot is not available for this method.')
    elif attrib == 'MCDREG':
        exitno = 3
        if choice in {'scree', 'pcadiag', '3ddiag', 'robdist', 'qqmcd', 'dd', 'ellipse', 'resfit', 'resindex', 'qqlts', 'diag', 'scatter', 'da', 'simca'}:
            raise ValueError('That kind of plot is not available for this method.')
    elif attrib == 'MLR':
        exitno = 3
        if choice in {'scree', 'pcadiag', '3ddiag', 'robdist', 'qqmcd', 'dd', 'ellipse', 'resfit', 'resindex', 'qqlts', 'diag', 'scatter', 'da', 'simca'}:
            raise ValueError('That kind of plot is not available for this method.')
    elif attrib == 'LTS':
        exitno = 7
        if choice in {'scree', 'pcadiag', '3ddiag', 'robdist', 'qqmcd', 'dd', 'ellipse', 'da', 'simca'}:
            raise ValueError('That kind of plot is not available for this method.')
        if 'rd' not in out:
            raise ValueError('Please rerun the LTS-regression again with the option plots equal to 1.')
    elif attrib in {'RAPCA', 'ROBPCA'}:
        exitno = 4
        if choice in {'regdiag', '3ddiag', 'robdist', 'qqmcd', 'dd', 'ellipse', 'resfit', 'resindex', 'qqlts', 'diag', 'scatter', 'da', 'simca'}:
            raise ValueError('That kind of plot is not available for this method.')
    elif attrib == 'RPCR':
        exitno = 6
        if choice in {'robdist', 'qqmcd', 'dd', 'ellipse', 'resfit', 'resindex', 'qqlts', 'diag', 'scatter', 'da', 'simca'}:
            raise ValueError('That kind of plot is not available for this method.')
    elif attrib in {'RSIMPLS', 'CSIMPLS'}:
        exitno = 5
        if choice in {'scree', 'robdist', 'qqmcd', 'dd', 'ellipse', 'resfit', 'resindex', 'qqlts', 'diag', 'scatter', 'da', 'simca'}:
            raise ValueError('That kind of plot is not available for this method.')
    elif attrib == 'MCDCOV':
        if choice in {'scree', 'pcadiag', 'regdiag', '3ddiag', 'resfit', 'resindex', 'qqlts', 'diag', 'scatter', 'da', 'simca'}:
            raise ValueError('That kind of plot is not available for this method.')
        exitno = 6
    elif attrib in {'RDA', 'CDA'}:
        exitno = 3
        if choice in {'scree', 'pcadiag', '3ddiag', 'robdist', 'qqmcd', 'dd', 'ellipse', 'resfit', 'resindex', 'qqlts', 'diag', 'scatter', 'regdiag', 'da', 'simca'}:
            raise ValueError('That kind of plot is not available for this method.')
        if out['center'].shape[1] > 2:
            print('Warning: Tolerance ellipses are only drawn for two-dimensional data sets.')
            return
    elif attrib in {'CSIMCA', 'RSIMCA'}:
        exitno = 3
        if choice in {'scree', 'pcadiag', '3ddiag', 'robdist', 'qqmcd', 'dd', 'ellipse', 'resfit', 'resindex', 'qqlts', 'diag', 'scatter', 'regdiag', 'da'}:
            raise ValueError('That kind of plot is not available for this method.')
        if out['pca'][0]['P'].shape[0] > 3:
            print('Warning: The dimension of the dataset is larger than 3.')
            return
    elif attrib in {'AGNES', 'DIANA'}:
        exitno = 4
        if choice in {'scree', 'pcadiag', '3ddiag', 'robdist', 'qqmcd', 'dd', 'ellipse', 'resfit', 'resindex', 'qqlts', 'diag', 'scatter', 'regdiag', 'da', 'silhouet', 'clus'}:
            raise ValueError('That kind of plot is not available for this method.')
    elif attrib == 'MONA':
        exitno = 3
        if choice in {'scree', 'pcadiag', '3ddiag', 'robdist', 'qqmcd', 'dd', 'ellipse', 'resfit', 'resindex', 'qqlts', 'diag', 'scatter', 'regdiag', 'da', 'silhouet', 'clus', 'tree'}:
            raise ValueError('That kind of plot is not available for this method.')
    elif attrib in {'PAM', 'FANNY'}:
        exitno = 4
        if choice in {'scree', 'pcadiag', '3ddiag', 'robdist', 'qqmcd', 'dd', 'ellipse', 'resfit', 'resindex', 'qqlts', 'diag', 'scatter', 'regdiag', 'da', 'tree', 'banner'}:
            raise ValueError('That kind of plot is not available for this method.')
    elif attrib == 'CLARA':
        exitno = 3
        if choice in {'scree', 'pcadiag', '3ddiag', 'robdist', 'qqmcd', 'dd', 'ellipse', 'resfit', 'resindex', 'qqlts', 'diag', 'scatter', 'regdiag', 'da', 'tree', 'silhouet', 'banner'}:
            raise ValueError('That kind of plot is not available for this method.')

    if exitno == 0:
        raise ValueError('Your attribute identifier must be one of the following names: CPCA, RAPCA, ROBPCA, CPCR, RPCR, LS, LTS, MCDREG, RSIMPLS, CSIMPLS, MCDCOV, CDA, RDA, CSIMCA, RSIMCA, AGNES, DIANA, MONA, PAM, CLARA, FANNY')

    # plotting what is asked for
    if ask == 0:
        whichplot(out, choice, attrib, exitno, labsd, labod, labresd, labmcd, lablts, labclus, classic, classicplots)
    else:
        # make menu of plots
        while choice != exitno:
            if attrib in {'CPCA', 'ROBPCA', 'RAPCA'}:
                choice = menuscore(out, attrib, exitno, labsd, labod, classic, classicplots)
            elif attrib in {'MCDREG', 'MLR'}:
                choice = menureg(out, attrib, exitno, labsd, labresd, classic, classicplots)
            elif attrib in {'COV', 'MCDCOV'}:
                choice = menucov(out, attrib, exitno, labmcd, classic, classicplots)
            elif attrib in {'RSIMPLS', 'CSIMPLS'}:
                choice = menupls(out, attrib, exitno, labsd, labod, labresd, classic, classicplots)
            elif attrib in {'CPCR', 'RPCR'}:
                choice = menuscoreg(out, attrib, exitno, labsd, labod, labresd, classic, classicplots)
            elif attrib in {'LTS', 'LS'}:
                choice = menuls(out, attrib, exitno, labsd, labresd, lablts, classic, classicplots)
            elif attrib in {'CDA', 'RDA'}:
                choice = menuda(out, attrib, exitno, classic, classicplots)
            elif attrib in {'CSIMCA', 'RSIMCA'}:
                choice = menusimca(out, attrib, exitno, classic, classicplots)
            elif attrib in {'AGNES', 'DIANA'}:
                choice = menuclus1(out, attrib, exitno)
            elif attrib == 'MONA':
                choice = menuclus2(out, attrib, exitno)
            elif attrib in {'PAM', 'FANNY'}:
                choice = menuclus3(out, attrib, exitno, labclus)
            elif attrib == 'CLARA':
                choice = menuclus4(out, attrib, exitno, labclus)


def menucov(out, attrib, exitno, labmcd, classic, classicplots):
    choice = input("Covariance plots: \n1. All\n2. Index plot of the distances\n3. Quantile-Quantile plot of the distances\n4. Distance-distance plot\n5. Tolerance ellipse (for bivariate data)\n6. Exit\nChoose an option: ")
    choice = int(choice)
    if choice == 1:
        plotn = 'all'
    elif choice == 2:
        plotn = 'Idist'
    elif choice == 3:
        plotn = 'qqmcd'
    elif choice == 4:
        plotn = 'dd'
    elif choice == 5:
        plotn = 'ellipse'
    elif choice == 6:
        return exitno
    whichplot(out, plotn, attrib, exitno, 3, 3, 3, labmcd, 3, [], classic, classicplots)
    return choice

def menureg(out, attrib, exitno, labsd, labresd, classic, classicplots):
    choice = input("Regression Plots: \n1. All\n2. Regression outlier map\n3. Exit\nChoose an option: ")
    choice = int(choice)
    if choice == 1:
        plotn = 'all'
    elif choice == 2:
        plotn = 'regdiag'
    elif choice == 3:
        return exitno
    whichplot(out, plotn, attrib, exitno, labsd, 3, labresd, 3, 3, [], classic, classicplots)
    return choice

def menuscore(out, attrib, exitno, labsd, labod, classic, classicplots):
    choice = input("Score Plots: \n1. All\n2. Scree plot\n3. Score outlier map\n4. Exit\nChoose an option: ")
    choice = int(choice)
    if choice == 1:
        plotn = 'all'
    elif choice == 2:
        plotn = 'scree'
    elif choice == 3:
        plotn = 'pcadiag'
    elif choice == 4:
        return exitno
    whichplot(out, plotn, attrib, exitno, labsd, labod, 3, 3, 3, [], classic, classicplots)
    return choice

def menupls(out, attrib, exitno, labsd, labod, labresd, classic, classicplots):
    choice = input("PLS Plots: \n1. All\n2. Score outlier map\n3. Regression outlier map\n4. 3D outlier map (Highly memory consuming)\n5. Exit\nChoose an option: ")
    choice = int(choice)
    if choice == 1:
        plotn = 'all'
    elif choice == 2:
        plotn = 'pcadiag'
    elif choice == 3:
        plotn = 'regdiag'
    elif choice == 4:
        plotn = '3ddiag'
    elif choice == 5:
        return exitno
    whichplot(out, plotn, attrib, exitno, labsd, labod, labresd, 3, 3, [], classic, classicplots)
    return choice

def menuscoreg(out, attrib, exitno, labsd, labod, labresd, classic, classicplots):
    choice = input("Score and Regression Plots: \n1. All\n2. Scree plot\n3. Score outlier map\n4. Regression outlier map\n5. 3D outlier map (Highly memory consuming)\n6. Exit\nChoose an option: ")
    choice = int(choice)
    if choice == 1:
        plotn = 'all'
    elif choice == 2:
        plotn = 'scree'
    elif choice == 3:
        plotn = 'pcadiag'
    elif choice == 4:
        plotn = 'regdiag'
    elif choice == 5:
        plotn = '3ddiag'
    elif choice == 6:
        return exitno
    whichplot(out, plotn, attrib, exitno, labsd, labod, labresd, 3, 3, [], classic, classicplots)
    return choice

def menuls(out, attrib, exitno, labsd, labresd, lablts, classic, classicplots):
    choice = input("Residual plots: \n1. All\n2. Standardized residuals versus fitted values\n3. Index plot of standardized residuals\n4. Normal QQplot of residuals\n5. Diagnostic plot of residuals versus robust distances\n6. Scatter plot with regression line\n7. Exit\nChoose an option: ")
    choice = int(choice)
    if choice == 1:
        plotn = 'all'
    elif choice == 2:
        plotn = 'resfit'
    elif choice == 3:
        plotn = 'resindex'
    elif choice == 4:
        plotn = 'qqlts'
    elif choice == 5:
        plotn = 'regdiag'
    elif choice == 6:
        plotn = 'scatter'
    elif choice == 7:
        return exitno
    whichplot(out, plotn, attrib, exitno, labsd, 3, labresd, 3, lablts, [], classic, classicplots)
    return choice

def menuda(out, attrib, exitno, classic, classicplots):
    choice = input("Discriminant analysis: \n1. All\n2. Tolerance ellipses (for bivariate data)\n3. Exit\nChoose an option: ")
    choice = int(choice)
    if choice == 1:
        plotn = 'all'
    elif choice == 2:
        plotn = 'da'
    elif choice == 3:
        return exitno
    whichplot(out, plotn, attrib, exitno, 3, 3, 3, 3, 3, [], classic, classicplots)
    return choice

def menusimca(out, attrib, exitno, classic, classicplots):
    choice = input("SIMCA analysis: \n1. All\n2. Scatter plot\n3. Exit\nChoose an option: ")
    choice = int(choice)
    if choice == 1:
        plotn = 'all'
    elif choice == 2:
        plotn = 'simca'
    elif choice == 3:
        return exitno
    whichplot(out, plotn, attrib, exitno, 3, 3, 3, 3, 3, [], classic, classicplots)
    return choice

def menuclus1(out, attrib, exitno):
    choice = input("Cluster analysis: \n1. All\n2. Banner plot\n3. Tree plot (n<30)\n4. Exit\nChoose an option: ")
    choice = int(choice)
    if choice == 1:
        plotn = 'all'
    elif choice == 2:
        plotn = 'banner'
    elif choice == 3:
        plotn = 'tree'
    elif choice == 4:
        return exitno
    whichplot(out, plotn, attrib, exitno, 3, 3, 3, 3, 3, [], 0, 0)
    return choice

def menuclus2(out, attrib, exitno):
    choice = input("Cluster analysis: \n1. All\n2. Banner\n3. Exit\nChoose an option: ")
    choice = int(choice)
    if choice == 1:
        plotn = 'all'
    elif choice == 2:
        plotn = 'banner'
    elif choice == 3:
        return exitno
    whichplot(out, plotn, attrib, exitno, 3, 3, 3, 3, 3, [], 0, 0)
    return choice

def menuclus3(out, attrib, exitno, labclus):
    choice = input("Cluster analysis: \n1. All\n2. Silhouette plot\n3. Clusplot\n4. Exit\nChoose an option: ")
    choice = int(choice)
    if choice == 1:
        plotn = 'all'
    elif choice == 2:
        plotn = 'silhouet'
    elif choice == 3:
        plotn = 'clus'
    elif choice == 4:
        return exitno
    whichplot(out, plotn, attrib, exitno, 3, 3, 3, 3, 3, labclus, 0, 0)
    return choice

def menuclus4(out, attrib, exitno, labclus):
    choice = input("Cluster analysis: \n1. All\n2. Clusplot\n3. Exit\nChoose an option: ")
    choice = int(choice)
    if choice == 1:
        plotn = 'all'
    elif choice == 2:
        plotn = 'clus'
    elif choice == 3:
        return exitno
    whichplot(out, plotn, attrib, exitno, 3, 3, 3, 3, 3, labclus, 0, 0)
    return choice


def whichplot(out, plotn, attrib, exitno, labsd, labod, labresd, labmcd, lablts, labclus, classic, classicplots):
    if plotn == 'all':
        # Call all relevant plotting functions
        if attrib in {'MCDCOV', 'COV'}:
            distplot(out['rd'], out['cutoff']['rd'], attrib, labmcd)
            chiqqplot(out['rd'], out['center'].shape[0], attrib)
            ddplot(out['md'], out['rd'], out['cutoff']['md'], attrib, labmcd)
            if out['center'].shape[0] == 2:
                ellipsplot(out['center'], out['cov'], out['X'], out['rd'], labmcd, attrib)
        # Add more cases for other attributes and plot types as needed
    elif plotn == 'Idist':
        distplot(out['rd'], out['cutoff']['rd'], attrib, labmcd)
    elif plotn == 'qqmcd':
        chiqqplot(out['rd'], out['center'].shape[0], attrib)
    elif plotn == 'dd':
        ddplot(out['md'], out['rd'], out['cutoff']['md'], attrib, labmcd)
    elif plotn == 'ellipse':
        if out['center'].shape[0] == 2:
            ellipsplot(out['center'], out['cov'], out['X'], out['rd'], labmcd, attrib)
        else:
            print('Tolerance ellipse plot is only available for bivariate data')
    # Add more cases for other plot types as needed
    
def distplot(rd, cutoff, attrib, labmcd):
    plt.figure()
    plt.plot(rd, 'o')
    plt.axhline(y=cutoff, color='r', linestyle='-')
    plt.title(f'{attrib} - Index plot of the distances')
    plt.xlabel('Index')
    plt.ylabel('Distance')
    plt.show()

def chiqqplot(rd, p, attrib):
    plt.figure()
    sorted_rd = np.sort(rd)
    chi2_quantiles = np.sqrt(np.sort(np.random.chisquare(p, len(rd))))
    plt.plot(chi2_quantiles, sorted_rd, 'o')
    plt.plot([0, max(chi2_quantiles)], [0, max(sorted_rd)], 'r--')
    plt.title(f'{attrib} - Q-Q plot of the distances')
    plt.xlabel('Chi-squared quantiles')
    plt.ylabel('Ordered distances')
    plt.show()

def ddplot(md, rd, cutoff, attrib, labmcd):
    plt.figure()
    plt.plot(md, rd, 'o')
    plt.axhline(y=cutoff, color='r', linestyle='-')
    plt.axvline(x=cutoff, color='r', linestyle='-')
    plt.title(f'{attrib} - Distance-distance plot')
    plt.xlabel('Mahalanobis distance')
    plt.ylabel('Robust distance')
    plt.show()

def ellipsplot(center, cov, data, rd, labmcd, attrib):
    from matplotlib.patches import Ellipse
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=rd, cmap='viridis')
    plt.colorbar(label='Robust distance')
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigvals)
    ell = Ellipse(xy=center, width=width, height=height, angle=angle, edgecolor='r', fc='None', lw=2)
    plt.gca().add_patch(ell)
    plt.title(f'{attrib} - Tolerance ellipse')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

