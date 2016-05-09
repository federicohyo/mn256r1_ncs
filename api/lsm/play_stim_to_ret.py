#!/usr/bin/env python
"""
An animated image
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
interval_speed = 25

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(nsteps):
    im = plt.imshow(np.reshape(M[:,i],[nx_d,ny_d]), interpolation='nearest')
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=interval_speed, blit=True,
    repeat_delay=0)

#ani.save('dynamic_images.mp4')


plt.show()

