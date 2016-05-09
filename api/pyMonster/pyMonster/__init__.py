# -*- coding: utf-8 -*-
'''
pyMonster

pyMonster is the pyNCS API used to configure MN256 chip throught the Monster daugther board

Toplevel functions
------------------

Modules
-------

Example
-------

'''

try:
    from .MonsterAPI import Configurator, Mappings
    from .client_monster import Client
    from .server_monster import Server
except ImportError as e:
    import sys
    print >>sys.stderr, '''\
Could not import submodules (exact error was: %s).

There are many reasons for this error the most common one is that you have
either not built the packages or have built (using `python setup.py build`) or
installed them (using `python setup.py install`) and then proceeded to test
the API **without changing the current directory**.

Try installing and then changing to another directory before importing
pyMonster.
''' % e

__all__ = [
    'Configurator', 'Client', 'Server'
    ]

