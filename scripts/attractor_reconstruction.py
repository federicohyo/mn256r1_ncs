import numpy
from pylab import *
import matplotlib
import pylab

def generate(data_length, odes, state, parameters):
    data = numpy.zeros([state.shape[0], data_length])
 
    for i in xrange(5000):
        state = rk4(odes, state, parameters)
 
    for i in xrange(data_length):
        state = rk4(odes, state, parameters)
        data[:, i] = state
 
    return data


def rk4(odes, state, parameters, dt=0.01):
    k1 = dt * odes(state, parameters)
    k2 = dt * odes(state + 0.5 * k1, parameters)
    k3 = dt * odes(state + 0.5 * k2, parameters)
    k4 = dt * odes(state + k3, parameters)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def lorenz_odes((x, y, z), (sigma, beta, rho)):
    return numpy.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
 
 
def lorenz_generate(data_length):
    return generate(data_length, lorenz_odes, \
        numpy.array([-8.0, 8.0, 27.0]), numpy.array([10.0, 8/3.0, 28.0])) #or as rosenstein paper: 16.0, 45.92, 4.0

def rossler_odes((x, y, z), (a, b, c)):
    return numpy.array([-y - z, x + a * y, b + z * (x - c)])
 
 
def rossler_generate(data_length):
    return generate(data_length, rossler_odes, \
        numpy.array([10.0, 0.0, 0.0]), numpy.array([0.15, 0.2, 10.0]))



data = lorenz_generate(2**13)
pylab.plot(data[0])

numpy.savetxt('lorenz.dat',data)
from mpl_toolkits.mplot3d.axes3d import Axes3D
 
figure = pylab.figure()
axes = Axes3D(figure)
axes.plot3D(data[0], data[1], data[2])
figure.add_axes(axes)
pylab.show()



#time delay embedding
# create time series
data = lorenz_generate(2**14)[0]
#data = preprocess(data, quantize_cols=[0], quantize_bins=1000)
 
# find usable time delay via mutual information
tau_max = 100
mis = []
 
for tau in range(1, tau_max):
    unlagged = data[:-tau]
    lagged = numpy.roll(data, -tau)[:-tau]
    joint = numpy.hstack((unlagged, lagged))
    mis.append(mutual_information(joint, normalized=True))
 
    if len(mis) > 1 and mis[-2] < mis[-1]: # return first local minima
        tau -= 1
        print tau, mis
        break
 
# plot time delay embedding
figure = pylab.figure()
axes = Axes3D(figure)
data_lag0 = data[:-2].flatten()
data_lag1 = numpy.roll(data, -tau)[:-2].flatten()
data_lag2 = numpy.roll(data, -2 * tau)[:-2].flatten()
axes.plot3D(data_lag0, data_lag1, data_lag2)
figure.add_axes(axes)
 
pylab.show()

