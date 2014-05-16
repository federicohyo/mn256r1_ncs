mn256r1_ncs
===========

Specific software libraries to control neuromorphic VLSI multi neuron chip developed at the Neuromorphic Cognitive System group, UZH ETHZ Zurich.


1) This software meant to be used with the latest MN256R1 VLSI multi-neuron chip and it requires pyNCS software package. pyNCS is available at [ https://github.com/inincs/pyNCS ] and documented here [ https://github.com/inincs/pyNCS/wiki ]


INSTALLATION
============

Install aerfx2 driver for usb communication
1) sh install_drivers.sh

Compile C libraries for sending and receiving events via usb
2) sh compile_lib.sh


TEST YOUR SETUP
===============

You can now test your setup by using the [ testsetup/test_all.py ]
It will run some routing test as 1) setting biases 2) sending and receiving spikes 3) programming synaptic matrixes.
If the test end succesfully you can start using the mn256r1 without any worries.
