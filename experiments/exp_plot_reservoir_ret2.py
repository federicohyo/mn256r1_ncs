'''
 Copyright (C) 2014 - Federico Corradi
 Copyright (C) 2014 - Juan Pablo Carbajal
 
 This progrm is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.
'''
############### author ##########
# federico corradi
# federico@ini.phys.ethz.ch
# Juan Pablo Carbajal 
# ajuanpi+dev@gmail.com
#
# ===============================
#!/usr/env/python
from __future__ import division
import numpy as np
from pylab import *
import sys
import matplotlib
sys.path.append('../api/reservoir/')
import reservoir as L
import time
import glob        #command line parser
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import savemat, loadmat


ion()
close('all')
directory = 'lsm_ret/'
fig_dir = 'figures/'            #make sure they exists
extension = "txt"
flag = {"reload": False}
plot_omega_grid = True

def _ismember(a, b):
    '''
    as matlab: ismember
    '''
    tf = np.array([i in b for i in a])
    u = np.unique(a[tf])
    index = np.array([(np.where(b == i))[0][-1] if t else 0 for i,t in zip(a,tf)])
    return tf, index

print "loading files... "
### folder parser
filenames_dat = []          #without extensions and dirs
inputs_dat = []
outputs_dat = []
omegas_dat = []
gestures = []
trials = [] 
omegas = []
for f in sorted(glob.iglob(directory+"*."+extension)):    
    f_raw = f
    f = f.replace(".txt","") #remove extension
    f = f.split("/")         #get directories
    if( f[1] != 'phi' and f[1] != 'theta' and f[1] != 'timev'): 
        ndirs  = np.shape(f)
        ndirs = int(''.join(map(str,ndirs)))
        filenames_dat.append(f[ndirs-1])
        if( f[ndirs-1].split("_")[0] == 'inputs'):
            inputs_dat.append(f_raw)
        if( f[ndirs-1].split("_")[0] == 'outputs'):    
            outputs_dat.append(f_raw)
        if( f[ndirs-1].split("_")[0] == 'omegas'):    
            omegas_dat.append(f_raw)    
            this_omegas = np.loadtxt(f_raw)
            omegas.append(this_omegas)

        this_gesture = int(f[ndirs-1].split("_")[2])
        gestures.append(this_gesture)
        this_trial = int(f[ndirs-1].split("_")[4])
        trials.append(this_trial)

omegas = np.array(omegas)    
omegas_shape = np.shape(omegas)
trials = np.array(trials)
gestures = np.array(gestures)    
ngestures = len(np.unique(gestures))    
ntrials =  len(np.unique(trials))   
print "found ", ngestures, " gestures"
print "found ", ntrials,   " trials per gesture"

## choose where to train and test    
timev = np.loadtxt(directory+"timev.txt")  
theta = np.loadtxt(directory+"theta.txt")   
phi = np.loadtxt(directory+"phi.txt")
[T,P] = np.meshgrid(theta,phi)
wX = (np.sin(P)*np.cos(T)).ravel() 
wY = (np.sin(P)*np.sin(T)).ravel() 
wZ = np.cos(P).ravel()

fig1 = figure(1)
ax = Axes3D(fig1)
for i in range(len(wX)):
    if(i%2):
        ax.scatter(wX[i],wY[i],wZ[i],c='r',marker='^')
    else:
        ax.scatter(wX[i],wY[i],wZ[i],c='b',marker='o')
title('teaching on triangles and training on dots')    
fig1.canvas.flush_events()

#this will be the teachings
wx_teaching = wX[::2]
wy_teaching = wY[::2]
wz_teaching = wZ[::2]

wx = []
wy = []
wz = []
index_teaching = []
index_testing = []
for i in range(omegas_shape[0]):
    tf,idx = _ismember(wx_teaching, [omegas[i,0]])
    if(np.sum(tf)>0):
        index_teaching.append(i)
    else:
        index_testing.append(i)
    wx.append(omegas[i,0])
    wy.append(omegas[i,1])
    wz.append(omegas[i,2])
wx = np.array(wx)
wy = np.array(wy)
wz = np.array(wz)
index_teaching = np.array(index_teaching)
index_testing  = np.array(index_testing)
n_teach        = len(index_teaching)
n_test         = len(index_testing)

#order wx as ... just to check
#fig = figure()
#ax = Axes3D(fig)
#for i in range(len(index_teaching)):
#    ax.scatter(wx[index_teaching[i]],wy[index_teaching[i]],wz[index_teaching[i]],c='r',marker='^')            
#for i in range(len(index_testing)):    
#    ax.scatter(wx[index_testing[i]],wy[index_testing[i]],wz[index_testing[i]],c='g',marker='o') 

####################
# Parameters for converting spikes to analog signals
dt_spk2sig = 35 # milliseconds
membrane   = lambda t,ts: \
             np.atleast_2d(np.exp((-(t-ts)**2)/(2*dt_spk2sig**2)))
# membrane converts output spikes to analog singals

if flag["reload"]:

    # Time vector for ALL analog signals
    T  = np.max(timev)
    nT = np.round(2 * T / dt_spk2sig)

    t_analog = np.linspace(0,T,nT) # milliseconds

    print "### PRE-LOADING TRAINING DATA"
    # Pre-load training data
    X_train = []
    W_train = []
    for i in xrange(n_teach):
        outputs = np.loadtxt(outputs_dat[index_teaching[i]])
        omegas  = np.loadtxt(omegas_dat[index_teaching[i]])
        X = L.ts2sig(t_analog, membrane, \
                     outputs[:,0], outputs[:,1], n_neu = 256)
        X_train.append(X)
        W_train.append(omegas)

    W_train = np.array (W_train)

    print "### PRE-LOADING TEST DATA"
    # Pre-load training data
    X_test = []
    W_test = []
    for i in xrange(n_test):
        outputs = np.loadtxt(outputs_dat[index_testing[i]])
        omegas  = np.loadtxt(omegas_dat[index_testing[i]])
        X = L.ts2sig(t_analog, membrane, \
                     outputs[:,0], outputs[:,1], n_neu = 256)

        X_test.append(X)
        W_test.append(omegas)

    W_test = np.array (W_test)


    print "### CUTING DATA"
    activation = np.array (map(lambda x: np.mean(x**2,axis=1),X_train)).T
    activation = np.where(activation**8 > 1)[0]

    idx        = [activation.min(), activation.max()+1]
    idx_spk    = [0,0]
    idx_spk[0] = np.argmin (np.abs(t_analog[idx[0]]-timev))
    idx_spk[1] = np.argmin (np.abs(t_analog[idx[1]]-timev))
    
    # Store analog signals
    X_train = np.array(map(lambda x: x[idx[0]:idx[1],:],X_train))
    X_train = np.transpose (X_train, axes=[1,2,0])

    X_test = np.array(map(lambda x: x[idx[0]:idx[1],:],X_test))
    X_test = np.transpose (X_test, axes=[1,2,0])

    nT = idx[1]-idx[0]
    T  = t_analog[idx[1]]-t_analog[idx[0]]

    t_analog = np.linspace(0,T,nT)[:,None] # milliseconds

    savemat ("cut_data.mat",{"X_train":X_train,\
                             "X_test":X_test,\
                             "t_analog":t_analog,\
                             "idx":idx,\
                             "idx_spk":idx_spk,\
                             "W_train":W_train,\
                             "W_test":W_test})
else:
    tmp = loadmat ("cut_data.mat")
    X_train  = tmp["X_train"]
    W_train  = tmp["W_train"]
    X_test  = tmp["X_test"]
    W_test  = tmp["W_test"]
    t_analog = tmp["t_analog"]
    idx      = tmp["idx"][0]
    del tmp
    

## Reservoir
res = L.Reservoir()


# Frequency scaling of teaching signal
base_freq = 1.2*np.pi/1e3; # MHz
delt_freq = base_freq*0.1; # MHz
# Teaching signal
T_sig = lambda t,w: np.mean( \
             np.sin((base_freq+delt_freq*w)*t),\
                axis=1)[:,None]

print "#### TEACHING"
## Update readout weights
for i in xrange(n_teach):
    #build teaching signal
    print "Offline training on data point ",\
             i, " of ", n_teach
    res.train(X_train[:,:,i], \
              teach_sig=T_sig(t_analog,W_train[i,:]))

print "#### EVALUATING TEACHING"
fig2       = figure(2)

n_subplots = np.ceil(np.sqrt(n_teach))
ax2 = []
for i in xrange(n_teach):
    ax2.append(fig2.add_subplot(n_subplots,n_subplots,i+1))
    ax2[i].axis('off')
fig2.canvas.draw()
fig2.canvas.flush_events()

rmse_teachings = []
for i in xrange(n_teach):
    zh   = res.predict(X_train[:,:,i])
    targ = T_sig(t_analog,W_train[i,:])
    rmse = res.root_mean_square(targ, \
                                zh["output"], norm=True)

    print "### RMSE training",i,": ", rmse
    rmse_teachings.append(rmse)

    ax2[i].plot(t_analog,zh["output"])
    ax2[i].plot(t_analog,targ)
    fig2.canvas.draw()
    fig2.canvas.flush_events()

print "#### TESTING"
fig3 = figure(3)

n_subplots = np.ceil(np.sqrt(n_test))
ax3 = []
for i in xrange(n_test):
    ax3.append(fig3.add_subplot(n_subplots,n_subplots,i+1))
    ax3[i].axis('off')
fig3.canvas.draw()
fig3.canvas.flush_events()

rmse_testings = []
for i in range(n_test):
    print "Offline training on data point ",\
             i, " of ", n_test

    zh   = res.predict(X_test[:,:,i])
    targ = T_sig(t_analog,W_test[i,:])
    rmse = res.root_mean_square(targ, \
                                zh["output"], norm=True)

    print "### RMSE test",i,": ", rmse

    rmse_testings.append(rmse)

    ax3[i].plot(t_analog,zh["output"])
    ax3[i].plot(t_analog,targ)
    fig3.canvas.draw()
    fig3.canvas.flush_events()


#order wx as ... just to check cmap=plt.cm.get_cmap('jet'),
fig = figure(4)
ax = Axes3D(fig)
color_teachings = (rmse_teachings/np.max(rmse_teachings))*30
color_testings = (rmse_testings/np.max(rmse_testings))*30
scatter1 = ax.scatter(wx[index_teaching],wy[index_teaching],wz[index_teaching],c=color_teachings*10,marker='^',s=50, label='teaching')            
scatter2 = ax.scatter(wx[index_testing],wy[index_testing],wz[index_testing],c=color_testings,marker='o',s=50,label='testing') 
    
scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='b', marker = '^')
scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='g', marker = 'o')

ax.legend([scatter1_proxy, scatter2_proxy], ['teaching', 'testing'], numpoints = 1)



