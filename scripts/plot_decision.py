import numpy as np
import pylab
import matplotlib

matplotlib.rcParams.update({'font.size': 18})

figure() #interactive plot
a = np.loadtxt('decision_times_b_this_freq_a50.0_this_freq_b50.0.txt')
a = a-200 #stimulus duration
hist(a,12,normed=True, color='g', label='pop B')
xlabel('reaction time [ms]', fontsize=20)
ylabel('norm. counts', fontsize=20)
legend(loc='best')

figure() #interactive plot
a = np.loadtxt('decision_times_a_this_freq_a50.0_this_freq_b50.0.txt')
a = a-200
hist(a,12,normed=True, color='b',linestyle='dashed', linewidth='2', label='pop A')
xlabel('reaction time [ms]', fontsize=20)
ylabel('norm. counts', fontsize=20)
legend(loc='best')
show()
