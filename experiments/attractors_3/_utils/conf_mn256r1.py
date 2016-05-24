############### author ##########
# federico corradi
# federico@ini.phys.ethz.ch
# script for generating connectivity matrixes for zurich multi neuron chip
# ===========================================================================

### ========================= import packages ===============================
import random
import numpy as np

### ========================= define what is needed to program the chip ====

available_neus = range(256)
matrix_learning_rec = np.zeros([256,256])
matrix_learning_pot = np.zeros([256,256])
matrix_programmable_rec = np.zeros([256,256])
matrix_programmable_w = np.zeros([256,256])
matrix_programmable_exc_inh = np.zeros([256,256])
popsne = np.array([])
popsni = np.array([])

### ========================= functions ===================================

def populate_neurons(n_neu, order=True):
    global available_neus 

    #how many neurons we need to place?
    if(n_neu > len(available_neus)):
        print 'game over.. not so many neurons'
        raise Exception
    else:
        neu_ret = []
        if(order == False):
            np.random.shuffle(available_neus)
        elif(order == True):
            available_neus.sort()
        for i in range(n_neu):
            neu_ret.append(available_neus.pop(0))
    return neu_ret


def connect_populations_programmable_inh(pop_pre,pop_post,connectivity,w):
    '''
        Connect two populations via programmable synapses with specified connectivity and w
    '''    
    if(np.shape(w)[0] == 2):
        w_min = w[0]        
        w_max = w[1]
        random_w = True
    elif(np.shape(w)[0] == 1 ):
        random_w = False
    else:
        print 'w should be np.shape(2), [w_min,w_max]'

    global matrix_programmable_exc_inh
    global matrix_programmable_w
    global matrix_programmable_rec
    #loop trought the pop and connect with probability connectivity
    for pre in pop_pre:
        for post in pop_post:
            coin = np.random.rand()
            if(coin < connectivity):
                #we connect this pre with this post
                matrix_programmable_exc_inh[post,pre] = 0  
                matrix_programmable_rec[post,pre] = 1    
                if(random_w):
                    matrix_programmable_w[post,pre] = np.random.randint(w_max)+w_min
                else:
                    matrix_programmable_w[post,pre] = w[0]




def connect_populations_learning(pop_pre,pop_post,connectivity,pot):
    '''
        Connect two populations via learning synapses with specified connectivity and pot
    '''    
    global matrix_learning_rec
    global matrix_learning_pot
    #loop trought the pop and connect with probability connectivity
    for pre in pop_pre:
        for post in pop_post:
            coin = np.random.rand()
            if(coin < connectivity):
                #we connect this pre with this post
                matrix_learning_rec[post,pre] = 1 
                coin = np.random.rand()
            if(coin < pot):  
                matrix_learning_pot[post,pre] = 1


def connect_populations_programmable(pop_pre,pop_post,connectivity,w):
    '''
        Connect two populations via programmable synapses with specified connectivity and w
    '''    
    if(np.shape(w)[0] == 2):
        w_min = w[0]        
        w_max = w[1]
        random_w = True
    elif(np.shape(w)[0] == 1 ):
        random_w = False
    else:
        print 'w should be np.shape(2), [w_min,w_max]'

    global matrix_programmable_rec
    global matrix_programmable_w
    global matrix_programmable_exc_inh
    #loop trought the pop and connect with probability connectivity
    for pre in pop_pre:
        for post in pop_post:
            coin = np.random.rand()
            if(coin < connectivity):
                #we connect this pre with this post
                matrix_programmable_exc_inh[post,pre] = 1
                matrix_programmable_rec[post,pre] = 1   
                if(random_w):
                    matrix_programmable_w[post,pre] = np.random.randint(w_max)+w_min
                else:
                    matrix_programmable_w[post,pre] = w[0]


def load_configuration(directory):
    '''
        load configuration from folder 
    '''
    global matrix_programmable_rec
    global matrix_programmable_w
    global matrix_programmable_exc_inh
    global available_neus 
    global popsne
    global popsni
    
    popsne = np.loadtxt(directory+'popse.txt')
    popsni = np.loadtxt(directory+'popsi.txt')
    available_neus = np.loadtxt(directory+'conf_available_neus.txt')
    matrix_learning_rec = np.loadtxt(directory+'conf_matrix_learning_rec.txt')
    matrix_learning_pot = np.loadtxt(directory+'conf_matrix_learning_pot.txt')
    matrix_programmable_rec = np.loadtxt(directory+'conf_matrix_programmable_rec.txt')
    matrix_programmable_w = np.loadtxt(directory+'conf_matrix_programmable_w.txt')
    matrix_programmable_exc_inh = np.loadtxt(directory+'conf_matrix_matrix_programmable_exc_inh.txt')


