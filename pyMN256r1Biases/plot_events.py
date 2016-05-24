
import numpy as np
from pylab import *
import matplotlib



def raster_plot(cicles):
    a = []
    for i in range(cicles):
        a.append(m.get())

    a = np.array(a)
    plot(a[:,1],a[:,0],'*')

    show()
