import matplotlib
from pylab import *
import numpy as np

ion()

###########################
# ATTRACTOR SYNAPTIC MATRIX
############################

matrix_learning_rec = np.loadtxt("con_matrix_learning_rec.txt")
matrix_programmable_rec = np.loadtxt("con_matrix_programmable_rec.txt")
matrix_programmable_exc_inh = np.loadtxt("con_matrix_programmable_rec.txt")

chip_matrix = np.zeros([256,512])

chip_matrix[:, 256:512] = matrix_learning_rec.reshape([256,256])
chip_matrix[:, 0:256] = matrix_programmable_rec.reshape([256,256])
chip_matrix[:, 0:256] = chip_matrix[:, 0:256] + matrix_programmable_exc_inh.reshape([256,256])*2

figure()
imshow(chip_matrix)


###############################
# RASTER PLOTS
###############################

### ECX
pop_1_ids = np.loadtxt("att_a_ids.txt")
pop_1_times = np.loadtxt("att_a_times.txt")
pop_2_ids = np.loadtxt("att_b_ids.txt")
pop_2_times = np.loadtxt("att_b_times.txt")
pop_3_ids = np.loadtxt("att_c_ids.txt")
pop_3_times = np.loadtxt("att_c_times.txt")

## INH
pop_i_1_ids = np.loadtxt("inh_a_ids.txt")
pop_i_1_times = np.loadtxt("inh_a_times.txt")
pop_i_2_ids = np.loadtxt("inh_b_ids.txt")
pop_i_2_times = np.loadtxt("inh_b_times.txt")
pop_i_3_ids = np.loadtxt("inh_c_ids.txt")
pop_i_3_times = np.loadtxt("inh_c_times.txt")

figure()
plot(pop_3_times, pop_3_ids, 'o', markersize=0.5, color='r')
plot(pop_2_times, pop_2_ids, 'o',markersize=0.5, color='g')
plot(pop_1_times, pop_1_ids, 'o',markersize=0.5, color='b')
plot(pop_i_3_times, pop_i_3_ids, 'o', markersize=0.5, color='b')
#plot(pop_i_2_times, pop_i_2_ids, 'o',markersize=0.5, color='r')
#plot(pop_i_1_times, pop_i_1_ids, 'o',markersize=0.5, color='g')
xlim([172000,184000])
xticks([172000,174000,176000,178000, 180000, 182000, 184000],[0,2000,4000,6000,8000,10000,12000])
xlabel('Time (ms)')
ylabel('Neurons')

#MEAN RATE PLOT
bins = 250
mf_att3, counts = np.histogram(pop_3_times-np.min(pop_3_times), bins)
mf_att1, counts = np.histogram(pop_1_times-np.min(pop_1_times), bins)
mf_att2, counts = np.histogram(pop_2_times-np.min(pop_2_times), bins)



