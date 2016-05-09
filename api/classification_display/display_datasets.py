import Tkinter
from PIL import Image, ImageTk
import numpy as np
import time
from scipy.misc import imresize
import missmatch as mm
from threading import Thread

class mainWindow(mm.base_classifier.BaseClassifier):
    times = 1
    timestart = time.clock()

    def __init__(self, setup, net,
                 train_fraction=0.8,
                 show_shape=(360, 360),
                 teacher_shape=(30, 175),
                 pattern_shape=(128, 128), 
                 speed_fac=1,
                 roll_fac=5,
                 **dataset_args):
        self.teacher = np.zeros([teacher_shape[0],teacher_shape[1]])
        self.root = Tkinter.Tk()
        self.frame = Tkinter.Frame(self.root,
                                   width=show_shape[0]+teacher_shape[0],
                                   height=show_shape[1])
        self.frame.pack()
        self.canvas = Tkinter.Canvas(self.frame,
                                     width=show_shape[0],
                                     height=show_shape[1])
        self.canvas_teach = Tkinter.Canvas(self.frame,
                                     width=teacher_shape[0],
                                     height=teacher_shape[1] * 2)
        self.canvas.place(x=-2,y=-2)
        self.canvas_teach.place(x=show_shape[0],y=0)
        self.shakerato = False
        self.smokypixels = True
        self.max_rand = 255
        self.this_p = 0
        #to_read = [0,1,2,3,4,5,6,7,8,9]
        #self.data = imresize(self.train[:, 0].reshape(pattern_shape), show_shape)
        self.pattern_shape = pattern_shape
        self.show_shape = show_shape
        self.teacher_shape = teacher_shape
        self.speed_fac = speed_fac
        self.roll_fac = roll_fac
        self.loop = False
        self.root.geometry('390x360+932+765')
        #self.root.mainloop()
        self.setup = setup
        super(mainWindow, self).__init__()
        self.retina_state = False    
        self.net = net
        

    def change_img(self, pattern=None, label=None):

        self.time_0 = time.time()
        
        if pattern is None or label is None:
            self.this_p =  np.random.randint(len(self.labels))        
            new_img = self.train[:, self.this_p].reshape(self.pattern_shape)
            new_lab = self.labels[self.this_p]
        else:
            new_img = pattern.reshape(self.pattern_shape)
            new_label = label
        self.data = imresize(new_img, np.r_[self.show_shape] + 2 * self.roll_fac)
        self.label = new_label
        self.refresh_image()
                
    def flip_teacher(self):
        #self.canvas_teach.configure(background=self.teacher[0,0])
        self.teacher = np.random.randint(0, 2, (self.teacher_shape)) * 255
    
    def roll_image(self):
    
        #self.data = np.roll(self.data,
                            #np.random.randint(-self.roll_fac, self.roll_fac+1),
                            #np.random.choice([0, 1]))
        dx, dy = np.random.normal(0, self.roll_fac, 2)
        self.canvas.coords(self.photo_im, dx, dy)
        self.root.update()

    def refresh_image(self):
        self.im = Image.fromstring('L',
                                   self.data.shape,
                                   self.data.astype('b').tostring())
        self.photo = ImageTk.PhotoImage(image=self.im)
        try:
            self.canvas.delete(self.photo_im)
        except:
            pass
        self.photo_im = self.canvas.create_image(0, 0, image=self.photo, anchor=Tkinter.NW)
        self.root.update()

    def refresh_teacher(self):
        self.im_teach = Image.fromstring('L',
                                   self.teacher_shape,
                                   self.teacher.astype('b').tostring())
        self.photo_teach = ImageTk.PhotoImage(image=self.im_teach)
        try:
            self.canvas.delete(self.photo_teach_im)
        except:
            pass
        self.photo_teach_im = self.canvas_teach.create_image(0, 0, image=self.photo_teach, anchor=Tkinter.NW)
        position =  self.teacher_shape[1] * (self.label > 0)
        self.canvas_teach.place(x = self.show_shape[0], y = position)
        self.root.update()

    def start(self):
        if not self.loop:
            return
        #self.flip_teacher()
        #if(self.shakerato == True):
        time_now = time.time() - self.time_0
        #print "%lf\r"%time_now
        
        #roll
        if time_now > self.delay_roll and\
           time_now < (self.delay_roll + self.roll_duration):
            if(not self.retina_state): 
                self.setup.mapper._program_detail_mapping(2**6)   #retina detail mapping on (interface 6) 
                self.retina_state = True       
            self.roll_image()
        if time_now > self.delay_roll+self.roll_duration and self.retina_state:
                self.setup.mapper._program_detail_mapping(2**7)
                self.retina_state = False
             
        # flicker
        if time_now > self.delay_teacher and time_now < (self.delay_teacher + self.teacher_duration) and self.teach_is_on:
            self.refresh_teacher()
            self.flip_teacher()
        
        # stop the loop
        if time_now > self.img_duration:
            self.loop = np.invert(self.loop)

        #self.root.update_idletasks()
        self.refresh_image()

    def _update(self, x, y):
        duration = 4000
        t = Thread(target=self._record_chip_activity, args=(duration,))
        t.daemon = True
        t.start()
        
        self.change_img(x, y)
        self.loop = np.invert(self.loop)
        while self.loop:
            self.start()
            
        t.join()    
        
            
    def _train_setup(self):
        self.delay_teacher = 0.2 * self.speed_fac
        self.teacher_duration = 0.7 * self.speed_fac
        self.delay_roll = 0.4 * self.speed_fac
        self.roll_duration = 0.5 * self.speed_fac
        self.img_duration = 1 * self.speed_fac
        self.teach_is_on = True

    def _test_setup(self):
        self.delay_teacher = 0.2 * self.speed_fac
        self.teacher_duration = 0.7 * self.speed_fac
        self.delay_roll = 0.4 * self.speed_fac
        self.roll_duration = 0.5 * self.speed_fac
        self.img_duration = 1 * self.speed_fac
        self.teach_is_on = False

    def _record_chip_activity(self, duration):
         self.out = self.setup.stimulate({}, send_reset_event=False, duration=duration) 

    def _predict(self, x):
        duration = 4000
        results = []
        if len(x.shape) > 1:
            for p in x.T:
                t = Thread(target=self._record_chip_activity, args=(duration,))
                t.daemon = True
                t.start()
                self.change_img(p, 0) # foo label
                self.loop = np.invert(self.loop)
                while self.loop:
                    self.start()
                    
                t.join()
                spikes_raw = self.out[0].raw_data()
                perceptron_neus = self.net.perceptrons_pop.soma.addr['neu']
                perceptron_neuron_classes = []
                step_class = len(perceptron_neus)/self.net.n_class
                for i in range(self.net.n_class):
                    current_class = perceptron_neus[i*step_class:(i+1)*step_class]
                    perceptron_neuron_classes.append(current_class)
                   
                self.out[0].t_start = self.out[0].t_stop - self.delay_roll*1000. 
                rates = self.out[0].mean_rates() 
                win_class = -1  #just init
                max_rates = -10 #init 
                for i in range(self.net.n_class):
                    this_r = rates[perceptron_neuron_classes[i]]
                    score = np.mean(this_r)
                    print ' class', i, 'score', score
                    if(score > max_rates):
                        win_class = i
                        max_rates = score
                
                if(win_class == 0):
                    print 'win class -1'    
                    results.append(-1)    
                else:
                    print 'win class 1'
                    results.append(1)
                    
            return results



if __name__ == '__main__':
    my_data_folder = '/local/d0/home/fabios/python/missmatch/data/'
    datasets = mm.seek_data(my_data_folder)
    #print datasets
    x, y = mm.load_data(my_data_folder + "mnist_train_1000.csv") # choose MNIST
    #x, y = mm.load_data(my_data_folder + "caltech101_car_sideVSmotorbikes.csv")
    # the target class is the one with label > 0
    # here we do 5 VS 3
    label_true = 5
    label_false = 3
    y_true = np.where(y == label_true, 1, 0)
    y_false = np.where(y == label_false, -1, 0)
    y_select = np.nonzero(y_true + y_false)[0]
    y = (y_true + y_false)[y_select]
    x = x[:, y_select]
    xi, yi, xt, yt = mm.sample_train_test(x, y, train_fraction=0.8,
                                          normalization=None)
    # pattern_shape is the shape of the original images
    win = mainWindow(pattern_shape=(28, 28),
                     show_shape=(500, 500),
                     train_fraction=0.8,
                     roll_fac=3,
                     normalization=None, teach_is_on=True, speed_fac=5)

    #ens = mm.Ensemble()
    #label_true = 5
    #ens.labels = np.unique(yi)
    #ens._set_ECOC('one-vs-rest', {'n':1})
    #ind_pos, ind_neg = ens._which_inds(label_true, yi)
    #win.train(xi, yi, 1)
    #win.root.mainloop()

