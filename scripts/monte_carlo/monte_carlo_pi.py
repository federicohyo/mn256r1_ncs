from random import random
from math import pow, sqrt

DARTS=1000
hits = 0
throws = 0
for i in range (1, DARTS):
	throws += 1
	x = random()
	y = random()
	dist = sqrt(pow(x, 2) + pow(y, 2))
	if dist <= 1.0:
		hits = hits + 1.0

# hits / throws = 1/4 Pi
pi = 4 * (hits / throws)

print "pi = %s" %(pi)


DARTS=len(analog_array)/2
hits = 0
throws = 0
for i in range (1, DARTS):
	counts = 0
	throws += 1
	x = analog_array[counts]
	y = analog_array[counts+1]
	dist = sqrt(pow(x, 2) + pow(y, 2))
	if dist <= 1.0:
		hits = hits + 1.0
	counts = counts+1

# hits / throws = 1/4 Pi
pi = 4 * (hits / throws)

print "pi = %s" %(pi)
