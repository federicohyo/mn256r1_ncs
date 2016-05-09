import Tkinter
from PIL import Image, ImageTk
import numpy
import time
import mnist
from scipy.misc import imresize


class mainWindow():
    times=1
    timestart=time.clock()
    to_read = [1,8]#[0,1,2,3,4,5,6,7,8,9]
    train , labels = mnist.read(to_read)
    data = imresize(train[0], (400,500))

    def __init__(self):
        self.root = Tkinter.Tk()
        self.frame = Tkinter.Frame(self.root, width=500, height=400)
        self.frame.pack()
        self.canvas = Tkinter.Canvas(self.frame, width=500,height=400)
        self.canvas.place(x=-2,y=-2)
        self.root.after(1,self.start) # INCREASE THE 0 TO SLOW IT DOWN
        self.root.mainloop()
        self.shakerato = False
        self.smokypixels = True
        self.max_rand = 255
        to_read = [1,8]#[0,1,2,3,4,5,6,7,8,9]
        self.train , self.labels = mnist.read(to_read)	
        self.data = imresize(self.train[0], (400,500))

    def change_img(self):
        this_p = numpy.random.randint(len(self.labels))
        self.train[this_p][self.labels[this_p]*(28/10):(self.labels[this_p]+1)*(28/10),0:3] = 255
        self.data = imresize(numpy.fliplr(self.train[this_p]), (400,500))
                
    def start(self):
        self.im=Image.fromstring('L', (self.data.shape[1],\
        self.data.shape[0]), self.data.astype('b').tostring())
        self.photo = ImageTk.PhotoImage(image=self.im)
        self.canvas.create_image(0,0,image=self.photo,anchor=Tkinter.NW)
        self.root.update()
        self.times+=1
        #### HERE REGULATES TIME BETWEEN IMAGES
        if self.times%25==0:
                #print "%.02f FPS"%(self.times/(time.clock()-self.timestart))
                self.change_img()
        self.root.after(25,self.start)
        #if(self.shakerato == True):
       	#self.data=numpy.roll(self.data,numpy.random.choice(([-1,1])),numpy.random.choice(([-1,1])))
        #elif(self.smokypixels == True):
        ### THIS REGULATES PIXELS UPDATES
        tmp_data = self.data
        x,y = numpy.where(tmp_data<=150)
        tmp_data[x,y] = 130 
        x,y = numpy.where(tmp_data>133) #133
        for i in range(len(x)*2):
            this_p = numpy.random.randint(len(x))
            self.data[x[this_p],y[this_p]] = numpy.random.randint(20)


        #else:
        #default is roll
        #self.data=numpy.roll(self.data,-1,-1)

if __name__ == '__main__':
    x=mainWindow()
