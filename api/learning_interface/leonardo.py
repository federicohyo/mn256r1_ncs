#!/usr/bin/python

import sys

if __name__=='__main__':

    if len(sys.argv)!=3:
        print("Usage is: 'python "+__file__+" hostname channel'")
        sys.exit(0)

    import pyNCS
    from LeonardoViewer import Viewer

    from expSetup_monster_learning import *

    print("Starting AerViewer on " + sys.argv[1])
    Viewer(int(sys.argv[2]),host=sys.argv[1],markersize=1).show()
