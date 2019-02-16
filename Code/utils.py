# import utils
# %matplotlib inline
## Launch the graph in a session.
# sess = utils.tf.Session()


import gpflow
import numpy as np
import matplotlib
import math
import scipy.stats as ss
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm

fontsize_label = 24
fontsize_legend = 20
fontsize_ticks = 20
fontsize_title = 24
linewidth = 2.3
markeredgewidth = 3
markersize = 15
matplotlib.rcParams['figure.figsize'] = (16, 8)
anchor = (1.15,0.5)


def lengthScale_SE(r, l):
    """
	Calculate covariance value of SE kernel for passed distance r and specific length scale l

	Parameters:
    r:   	(float) Distance r
    l: 		(float) Specific length scale for kernel
    
    Return:
	Value of kernel for given r and l

	"""
    return np.exp((-r ** 2) / (2 * l ** 2))


def lengthScale_Matern(r, l, p):
    """
    Calculate covariance value of matern kernel with different p for passed distance r and specific length scale l

    Parameters:
    r:   	(float) Distance r
    l: 		(float) Specific length scale for kernel
    p: 		(int) Value to specify matern kernel we draw from 
    
    Return:
    Value of kernel for given r and l

    """
    if p == 3.0:
        return (1. + (np.sqrt(p) * r) / l) * np.exp(-(np.sqrt(p) * r) / l)
    elif p == 5.0:
        return (1. + (np.sqrt(p) * r) / l + (np.sqrt(p) * r ** 2) / (3 * l ** 2)) * np.exp(-np.sqrt(p) * r / l)


def drawRandomFunction(k, X, num_Functions):
    """
    Draws a number of functions from a given kernel for input space X

    Parameters:
    X:   			(np.array) All possible inputs
    k: 				Kernel that is currently used in context (GPFlow kernel!)
    num_Functions:	(int) number of functions to draw

    Return:
    Ŕandom multivariate functions drawn from kernel
    """
    # Calculat covariance matrix for input space

    K = k.compute_K_symm(X)
    return np.random.multivariate_normal(np.zeros(X.shape[0]), K, num_Functions).T


def plotRandomFunction(X, Y, K, labels):
    """
    Plot a random function drawn from a kernel

    Parameters:
    X:   	(np.array) All possible inputs
    Y: 		(list) List of function values of given kernel for input space
    K: 		(list) All kernels the values where drawn from. Has to have same number of entries as len(Y)
    labels: (str) labels of plot

    Return:
    -
    """

    plt.figure()
    ax = plt.gca()

    for y, k, l in zip(Y, K, labels):
        if l is not None:
            ax.plot(X, y, lw=linewidth, label=l)
        else:
            ax.plot(X, y, lw=linewidth)
            # Title
        if not any(x in k.__class__.__name__ for x in ["White", "Linear", "Constant", "Polynomial"]):
            ax.set_title(
                k.__class__.__name__ + '(l={0}, $\sigma_f^2$={1})'.format(str(k.lengthscales.value),
                                                                          str(k.variance.value)),
                fontsize=fontsize_legend)
        else:
            ax.set_title(k.__class__.__name__, fontsize=fontsize_legend)

    # Format
    plt.xlabel('x', fontsize=fontsize_label)
    plt.ylabel('f(x)', fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid()
    plt.legend()
    ax.set_xlim([X[0], X[-1]])


def plotLengthScales(X, Y, scales, title):
    """
    Plots the covariance value of a given kernel depending on the scale distance between two points

    Parameters:
    X:   	(np.array) All possible inputs
    Y: 		(list) List of covariance values of given kernel for input space
    scales: (np.array) All scales to plot. Has to have same number of entries as len(Y)
    title: 	(str) Title of plot

    Return:
    -
    """

    plt.figure()
    ax = plt.gca()

    for l, y in zip(scales, Y):
        ax.plot(X, y, lw=linewidth, label='l={0}'.format(str(l)))

        # Format plot
    ax.set_title(title, fontsize=fontsize_legend)
    plt.legend(prop={'size': fontsize_legend}, loc=0)
    plt.xlabel('r', fontsize=fontsize_label)
    plt.ylabel('k(r)', fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid()
    ax.set_xlim([X[0], X[-1]])


def plot2D(mean, var, X, *argv):
    """
    Plot a 2D estimate based on a Gaussian Process estimate of some function f
    
    Mean and Variance of the prediction as well as a not known number of arrays is passed to this plotting function. 
    Each array contains an object that is plotted.The array has to have the following structure: 
    [xaxis, yaxis, "label", "marker"] where "marker" is an optional parameter
    
    For colors to work properly the following convention should be minded: 
    1. Array is the ground trouth if known
    2. Array is the estimated mean
    3. Array is the estimated variance
    """

    fig = plt.figure()
    for arg in argv:
        if len(arg) == 4:
            plt.plot(arg[0], arg[1], arg[3], mew=markeredgewidth, label=arg[2], lw=linewidth)
        else:
            plt.plot(arg[0], arg[1], mew=markeredgewidth, label=arg[2], lw=linewidth)
    # Plot variance
    if mean is not None and var is not None:
        plt.fill_between(X[:, 0], mean[:, 0] - 2 * np.sqrt(var[:, 0]), mean[:, 0] + 2 * np.sqrt(var[:, 0]),
                         color='C1', alpha=0.2, label='$\sigma^2$')
    # Format plot
    plt.legend(prop={'size': fontsize_legend}, loc=0)
    #     plt.legend(prop={'size': fontsize_legend}, loc=4)
    plt.xlabel('x', fontsize=fontsize_label)
    plt.ylabel('f(x)', fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid()


def plot_Acq(mean, var, X, Y_true, X_sample, data, mark_max=True, *argv):
    '''
    Function to plot ground truth, observed data as well as mean, variance (posterior) and
    the acquisition function for given arguments. The function will plot multiple acquisition functions in
    multiple subplot if receiving more than one array.

    Parameters:
    mean:     Mean of the posterior
    var:      Variance of the posterior
    X:        All possible inputs
    X_Sample: X-Coordinate of the observed data
    Data:     Value of the observed data
    mark_max: Whether or not to mark the best value according to the acq. function
    argv:     Array of acq. functions and their values as well as labels

    Return:
    -
    '''

    fig = plt.figure()
    ax = fig.add_subplot(len(argv) + 1, 1, 1)
    ax.plot(X, Y_true, label='$f_{true}$', lw=linewidth)
    ax.plot(X, mean, label='$\mathbb{E[}\hat{f(x)}\mathbb{]}$', lw=linewidth)
    # Plot variance
    if mean is not None and var is not None:
        ax.fill_between(X[:, 0], mean[:, 0] - 2 * np.sqrt(var[:, 0]), mean[:, 0] + 2 * np.sqrt(var[:, 0]), color='C1',
                        alpha=0.2, label='Var($\hat{f(x)}$)')
    ax.plot(X_sample, data, 'kx', label="Data", mew=markeredgewidth, lw=linewidth)

    #     ax.set_xlim(left=np.min(X), right=np.max(X))
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid()
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.ylabel('f(x)', fontsize=fontsize_label)
    plt.legend(prop={'size': fontsize_legend}, loc=4)

    for arg, index in zip(argv, range(len(argv))):

        ax = fig.add_subplot(len(argv) + 1, 1, index + 2, sharex=ax)
        ax.plot(arg[0], arg[1], mew=markeredgewidth, label=arg[2], lw=linewidth)  # , marker='o')
        #         ax.set_xlim(left=np.min(X), right=np.max(X))
        if mark_max:
            ax.plot(X[np.argmax(arg[1])], arg[1][np.argmax(arg[1])], 'ro', mew=markeredgewidth, markersize=7)
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        plt.legend(prop={'size': fontsize_legend}, loc=4)
        plt.xlabel('x', fontsize=fontsize_label)
        plt.ylabel('f(x)', fontsize=fontsize_label)
        plt.grid()


def plot_convergence(X_sample, Y_sample, num_init=2):
    plt.figure()

    # Exclude initial samples
    x = X_sample[num_init:].ravel()
    y = Y_sample[num_init:].ravel()
    r = range(1, len(x)+1)
    # Calculate distance between first selected point and the next one
    x_dist = [np.abs(a-b) for a, b in zip(x, x[1:])]
#     y_max_watermark = np.maximum.accumulate(y)

    plt.subplot(1, 2, 1)
    plt.plot(r[1:], x_dist, 'o-', lw=linewidth)
    plt.xlabel('Iteration', fontsize=fontsize_label)
    plt.ylabel('Distance', fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.title('Distance between consecutive x\'s', fontsize=fontsize_title)


def compare_acq(means, variances, labels, X, Y_true, samples_x, samples_y, markers, size_init,show_var, mark_max, *argv):

    col = []
    # Create colors according to number of functions compared
    for i in range(len(means)):
        col.append('C'+str(i))
    fig = plt.figure()
    ax = fig.add_subplot(len(argv)+1, 1, 1)
    for mean, var, label, index in zip(means, variances,labels,range(len(means))):
        ax.plot(X, mean, label='$\mu_{'+ label + '}$', lw=linewidth)
        # Plot variance
        if var is not None and show_var:
            ax.fill_between(X[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]),
                            alpha=0.2, label='$\sigma_{' + label + '}^2$', color=col[index])

    ax.plot(X, Y_true, '--', label='$f_{true}$', lw=linewidth)
    for x,y, marker in zip(samples_x,samples_y, markers):
        # Mark initial data points
        ax.plot(x[:size_init], y[:size_init], 'kv', mew=markeredgewidth, lw=linewidth)
        # Mark all others
        ax.plot(x[size_init:], y[size_init:], 'k' + marker, mew=markeredgewidth, lw=linewidth)#, label='Data')

    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.grid()
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.ylabel('f(x)', fontsize=fontsize_label)
    plt.legend(prop={'size': fontsize_legend}, loc=7, bbox_to_anchor=anchor)

    ax = fig.add_subplot(len(argv)+1, 1, 2, sharex=ax)
    for arg, index in zip(argv, range(len(argv))):
        ax.plot(arg[0], minmax(arg[1]), mew=markeredgewidth, label=arg[2], lw=linewidth)
        if mark_max:
            ax.plot(X[np.argmax(arg[1])], minmax(arg[1])[np.argmax(arg[1])], col[index] + markers[index], mew=markeredgewidth,
                    markersize=7)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(prop={'size': fontsize_legend}, loc=7, bbox_to_anchor=anchor)
    plt.xlabel('x', fontsize=fontsize_label)
    plt.ylabel('f(x)', fontsize=fontsize_label)
    plt.grid()


def removefromArray(base,indices):
    '''
    Function to remove indices from an np.array

    Parameters:
    base:    (np.array) Array from which to remove from
    indices:  (np.array) Array that hold indecies of values to remove

    Return
    (np.array) Array without values at indices

    '''

    arr = np.copy(base)
    arr = arr.tolist()
    for ind in indices:
        try:
            del arr[ind]
        except:
            pass

    return np.asarray(arr)


def findIndices(X,targets):
    '''
    Find indices of targets in given array

    Parameters:
    X:    (np.array) Array in which to search
    target:  (np.array) Array that hold values to find

    Return:
    indices (list) List of indices
    '''
    X = X.tolist()
    indices = []
    for target in targets:
        try:
            ind=X.index(target.tolist())
            #print(ind)
            indices.append(ind)
        except:
            pass

    return indices


def cond_var(x, A, k):
    '''
    Calculate conditional variance according to equation 2 in Krause, Singh, Guestrin - 2008 - Near-Optimal
    Sensor Placements in Gaussian Processes Theory, Efficient Algorithms and Empirical Studies. Same equation as
    eq. 2.24 in http://www.gaussianprocess.org/gpml/chapters/RW.pdf. Cares for negative variances due to numerical
    issues.

    Parameters:
    x:       Potential sampling point
    A:       Subset of already placed sensors/measurement points or all remaining points
    k:       Kernel that is currently used in context (GPFlow kernel!)

    Return:
    cond_var: Conditional variance according to eq. 2 in mentioned paper
    '''
    #print("cond_var(x, A, k)")
    #print("k.compute_K_symm(x)="+str(k.compute_K_symm(x)))
    #print("k.compute_K(x,A)="+str(k.compute_K(x,A)))
    #print("k.compute_K_symm(A)="+str(k.compute_K_symm(A)))
    #print("k.compute_K(A,x)="+str(k.compute_K(A,x)))
    #print("np.linalg.inv(k.compute_K_symm(A)))="+str(np.linalg.inv(k.compute_K_symm(A))))
    #print("----------------------------------------------------------------")
    cov_conditional = k.compute_K_symm(x) - np.dot(np.dot(k.compute_K(x,A) ,np.linalg.inv(k.compute_K_symm(A))),
                                                   k.compute_K(A,x))

    # Check for negative variances resulting from numerical issues
    cov_conditional = np.array([0.000001]) if cov_conditional <= 0.0 else cov_conditional

    return cov_conditional


def cond_entropy(var):
    '''
    Calculate conditional entropy according to eq. 5 from Krause, Singh, Guestrin - 2008 - Near-Optimal
    Sensor Placements in Gaussian Processes Theory, Efficient Algorithms and Empirical Studies


    :param var: variance to calculate conditional entropy from
    :return: conditional variance
    '''

    return 0.5 * np.log(var) + 0.5 * np.log((2 * np.pi) + 1)


def minmax(X):

    lst = []
    min = np.amin(X)
    max = np.amax(X)
    for x in X:
        lst.append((x - min) / (max - min) )

    return np.asarray(lst)


def sampleFeature(lst, ind):
    '''
    Sample from a feature vector by index. For each array(feature) in lst select the indices specified by the 
    coloumn in ind.
    Example:
    len(lst)=3 -> ind.shape = Nx3
    Now select N values from lst[x] where x is the x.th array
    
    Parameters:
    lst: List that holds the feature vectors of type np.array
    ind: Array with all index pairs of shape: N x len(lst) 
    
    Return:
    ret: List of samples from each feature vector
    
    '''
    ret = []
    for i in range(len(lst)):
        ret.append(lst[i][ind[:,i]].reshape(len(ind),1))        

    return ret


def samplefromFunc(f, ind):
    '''
    Function that samples from an N-dimensional input function by given indices. 
    
    Parameters:
    f:   Multidimensional array of function values
    ind: Array with all index pairs of shape: N x f.ndim
    
    Return:
    ret: Array of samples from passed function f
    '''
    
    lst = []
    for i in range(len(ind)):
        # Use tuple to index array since using a list of array is deprecated
        lst.append(f[tuple(np.split(ind[i][:],f.ndim))])
    
    return np.asarray(lst)



def plot3D(x1_mesh, x2_mesh, z, X1_sample, X2_sample, offset, bound, markersize):

    fig = plt.figure(figsize=(16, 12))
    ax = plt.axes(projection='3d')

    #if X1_sample and X2_sample not None:
    #    ax.scatter3D(X1_sample, X2_sample, offset, marker='o', edgecolors='k', color='r', label="Data", s=150)
    ax.plot_surface(x1_mesh, x2_mesh, z, cmap=cm.viridis, linewidth=0.5, antialiased=True, alpha=0.8)
    contour = ax.contourf(x1_mesh, x2_mesh, z, zdir='z', offset=offset, cmap=cm.viridis, antialiased=True)
    fig.colorbar(contour)
    ax.scatter3D(X1_sample, X2_sample, offset, marker='o', edgecolors='k', color='r', label="Data", s=markersize, zorder=10)

    for t in ax.zaxis.get_major_ticks():
        t.label.set_fontsize(fontsize_ticks)
    ax.set_title("$f(x)$", fontsize=fontsize_title)
    ax.set_xlabel("\n$x_1$", fontsize=fontsize_label)
    ax.set_ylabel("\n$x_2$", fontsize=fontsize_label)
    ax.set_zlabel('\n\n$f(x_1,x_2)$', fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlim(left=-bound, right=bound)
    plt.ylim(bottom=-bound, top=bound)
    ax.set_zlim3d(offset, np.max(z))


def plotAcq3D(mean, var, x1_mesh, x2_mesh, X_sample, bound, mark_max, *argv):
    '''
    Function to plot ground truth, observed data as well as mean, variance (posterior) and
    the acquisition function for given arguments. The function will plot multiple acquisition functions in
    multiple subplot if receiving more than one array.

    Parameters:
    mean:     Mean of the posterior
    var:      Variance of the posterior
    X:        All possible inputs
    X_sample: X-Coordinate of the observed data
    argv:     Array of acq. functions and their values as well as labels

    Return:
    -
    '''
    offset = -3.
    
    # Plot posterior mean and variance
    fig = plt.figure(figsize=(24, 10))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    cbar = ax.plot_surface(x1_mesh, x2_mesh, mean, cmap=cm.viridis, linewidth=0.5, antialiased=True, alpha=0.8)
    cbar = ax.contourf(x1_mesh, x2_mesh, mean, zdir='z', offset=offset, cmap=cm.viridis, antialiased=True)
    ax.scatter3D(X_sample[:,0], X_sample[:,1], offset, marker='o',edgecolors='k', color='r', label="Data", s=150)
    fig.colorbar(cbar)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(fontsize_ticks)
    ax.set_title("$\mu(x_1,x_2)$", fontsize=fontsize_title)
    ax.set_xlabel("\n$x_1$", fontsize=fontsize_label)
    ax.set_ylabel("\n$x_2$", fontsize=fontsize_label)
    ax.set_zlabel('\n\n$\mu(x_1,x_2)$', fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlim(left=-bound, right=bound)
    plt.ylim(bottom=-bound, top=bound)
    ax.set_zlim3d(offset,np.max(mean))

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    cbar = ax.contourf(x1_mesh, x2_mesh, var, zdir='z', offset=offset, cmap=cm.viridis, antialiased=True)
    ax.scatter3D(X_sample[:,0], X_sample[:,1], offset, marker='o',edgecolors='k', color='r', label="Data", s=150)
    fig.colorbar(cbar)

    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(fontsize_ticks)
    ax.set_title("$\sigma^2(x_1,x_2)$", fontsize=fontsize_title)
    ax.set_xlabel("\n$x_1$", fontsize=fontsize_label)
    ax.set_ylabel("\n$x_2$", fontsize=fontsize_label)
    ax.set_zlabel('\n\n$\sigma^2(x_1,x_2)$', fontsize=fontsize_label)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlim(left=-bound, right=bound)
    plt.ylim(bottom=-bound, top=bound)
    ax.set_zlim3d(offset,np.max(mean))
    plt.show()
    
    # Plot Acq Function
    for arg in argv:
        fig = plt.figure(figsize=(16, 12))
        ax = plt.axes(projection='3d')
        cbar = ax.contourf(x1_mesh, x2_mesh, arg[0], zdir='z', offset=offset, cmap=cm.viridis, antialiased=True)
        fig.colorbar(cbar)
        ax.scatter3D(X_sample[:-1,0], X_sample[:-1,1], offset, marker='o',edgecolors='k', color='r', label="Data", s=150)
        if mark_max:
            #ind_max = np.unravel_index(indices=np.argmax(ei), dims=(num,num))
            #ax.scatter3D(x1[ind_max[0]], x2[ind_max[1]], offset, marker='^',edgecolors='k', color='m', label="Data", s=150)
            ax.scatter3D(x1_mesh.reshape(-1)[np.argmax(arg[0])], x2_mesh.reshape(-1)[np.argmax(arg[0])], offset, marker='^',edgecolors='k', 
                         color='m', label="Data", zorder=10, s=150)

        ax.set_title(arg[1], fontsize=fontsize_title)
        ax.set_xlabel("\n$x_1$", fontsize=fontsize_label)
        ax.set_ylabel("\n$x_2$", fontsize=fontsize_label)
        ax.set_zlabel('\n\n'+arg[1], fontsize=fontsize_label)
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        plt.xlim(left=-bound, right=bound)
        plt.ylim(bottom=-bound, top=bound)
        ax.set_zlim3d(offset, np.max(mean))
    