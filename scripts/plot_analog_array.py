import matplotlib
from pylab import *
import numpy as np
import itertools
import mpl_toolkits.mplot3d.axes3d as p3

import numpy as np
from scipy import *
from matplotlib import pyplot as gplt
from scipy import fftpack
import scipy.sparse as sp
import scikits.statsmodels.api as sm

import sys

sys.path.append('/users/federico/projects/work/trunk/code/python/spkInt/scripts/')

import analyze_balanced

matplotlib.rcParams.update({'font.size': 18})
signal = np.loadtxt('analog_array.txt')
mut_m = analyze_balanced.mutual_information(signal,2000,32) 

n = len(signal)
df = 1./(n*0.001)
psd = abs(0.001*fftpack.fft(signal[:n/2]))**2
f=df*np.arange(n/ 2) 

def f(Y,x):
    total = 0
    for i in range(20):
        total += Y.real[i]*np.cos(i*x) + Y.imag[i]*np.sin(i*x)
    return total


Y=fft(signal)
n=len(Y)
print n

xs = linspace(0, 2*pi,1000)
plot(xs, [f(Y, x) for x in xs], '.')

ion()
#mutual information
figure()
plot(mut_m[:,0],mut_m[:,1],'ro-')
xlabel(r'$\tau [s]$', fontsize=22)
ylabel(r'$I(\tau)$', fontsize=22)

nn = len(signal)
dimm = 4
mm = nn-(dimm-1)*480
yy = analyze_balanced.psr_embedded_dim(signal,4,480,mm)

dele_base = 48
ion()
for i in range(1):
    dele = dele_base +  i
    nn = len(signal)
    dimm = 4
    mm = nn-(dimm-1)*dele
    figure()
    yy = analyze_balanced.psr_embedded_dim(signal,dimm,dele,mm)
    plot(yy[dele::dele],yy[0:len(yy)-dele:dele],color='blue', linewidth='0.2')
    title('del'+str(dele))

#sparse matrix
matrix_space = sp.csr_matrix(((len(yy)-48), (len(yy)-48)))

data = np.repeat(1,len(yy[48::48]))

matrix_total = []
for i in range(np.shape(yy)[1]):
    matrix = sp.csr_matrix((data,(yy[48::48,i]+128,yy[0:len(yy)-48:48,i]+128)))
    matrix_total.append(matrix)

matrix_space_dots = matrix_total[0].todense()+matrix_total[1].todense()+matrix_total[2].todense()+matrix_total[3].todense()
ion()
figure()
imshow(np.rot90(matrix_space_dots))
colorbar()

#try this
vector_field = np.loadtxt('1_vectfield3.dat')
figure()
quiver(vector_field[:,0],vector_field[:,1],vector_field[:,2],vector_field[:,3],headwidth='2',headlength='4')

xlabel(r'$x_{i}$',fontsize=28)
ylabel(r'$x_{i+\tau}$',fontsize=28)
xlim([-130,130])
ylim([-130,130])

axis('equal')
#we divide them in squares of 8*8

point_now =  [yy[48+48*0::48,i][0],yy[48+48*1::48,i][0]]
point_after = [yy[48+48*2::48,i][0],yy[48+48*3::48,i][0]]

yy = gramm(yy)
plot(yy[480::480]*10000,yy[0:len(yy)-480:480]*10000,color='blue')

xlabel(r'$x_{i}$',fontsize=22)
ylabel(r'$x_{i+\tau}$',fontsize=22)

marker = itertools.cycle(('o', 's', '8', 'D')) 
color = itertools.cycle(('r', 'g', 'm', 'y')) 
linestyle = itertools.cycle(('-','--','-.',':'))

figure()
for dim in range(4):
    subplot(2,2,dim)
    plot(yy[:,dim],color=color.next(),marker=marker.next())
    xticks([0,2000,4000,6000,8000,10000,12000,14000,16000,18000], [0, 20, 40, 60, 80, 100,120,140,160,180])
    xlim([0, 16000])
    xlabel(r'$\tau$ [s]', fontsize=22)
    ylabel(r'$a$', fontsize=24)


figure()
for dim in range(4):
    hold(True)
    plot(yy[:,dim])
    xticks([0,2000,4000,6000,8000,10000,12000,14000,16000,18000], [0, 20, 40, 60, 80, 100,120,140,160,180])
    xlim([0, 16000])



def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as NP
    from scipy import linalg as LA
    mn = NP.mean(data, axis=0)
    # mean center the data
    data -= mn
    # calculate the covariance matrix
    C = NP.cov(data.T)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = LA.eig(C)
    # sorted them by eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:,:dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    data_rescaled = NP.dot(evecs.T, data.T).T
    # reconstruct original data array
    data_original_regen = NP.dot(evecs, dims_rescaled_data).T + mn
    return data_rescaled, data_original_regen


def plot_pca(data):
    clr1 =  '#2026B2'
    fig = figure()
    ax1 = fig.add_subplot(111)
    data_resc, data_orig = PCA(data, dims_rescaled_data=3)
    #ax1.plot(data_resc[:,0], data_resc[:,1], '.', mfc=clr1, mec=clr1)
    #scatter((data_resc[:, 0]), (data_resc[:, 1]),c=('r','g','b'))
    fig = figure()
    ax = p3.Axes3D(fig)
    ax.scatter(data_resc[:,0], data_resc[:,1], data_resc[:,2], c=('b','g','r'))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.add_axes(ax)
    show()
    #import numpy as np
    #from mpl_toolkits.mplot3d import Axes3D
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #n = 100
    #for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    #    xs = data_resc[:,0]
    #    ys = data_resc[:,1]
    #    zs = data_resc[:,2]
    #    ax.scatter(xs, ys, zs, c=c, marker=m)

    #ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')

    #ax.azim = 200
    #ax.elev = -45

    #plt.show()



from math import sqrt
 
def gramm(X,inplace = False):
    # Returns the Gramm-Schmidt orthogonalization of matrix X
    if not inplace:
       V = [row[:] for row in X]  # make a copy.
    else:
       V = X
    k = len(X[0])          # number of columns. 
    n = len(X)             # number of rows.
 
    for j in range(k):
       for i in range(j):
          # D = < Vi, Vj>
          D = sum([V[p][i]*V[p][j] for p in range(n)])
 
          for p in range(n): 
            # Note that the Vi's already have length one!
            # Vj = Vj - <Vi,Vj> Vi/< Vi,Vi >
            V[p][j] -= (D * V[p][i])
 
       # Normalize column V[j]
       invnorm = 1.0 / sqrt(sum([(V[p][j])**2 for p in range(n)]))
       for p in range(n):
           V[p][j] *= invnorm
    return V





