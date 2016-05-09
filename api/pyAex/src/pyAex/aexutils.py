from collections import defaultdict
from contextlib import contextmanager
from pyNCS.pyST import events
from pyNCS.pyST import channelEvents
import numpy as np
import errno
import time
import sys
import os
from os import system
import aex_globals
from getpass import getuser

def is_file_empty(filename):
    return os.stat(filename)[6] == 0


def default_user(user=None):
    if user == None:
        return getuser()
    else:
        return user


def __import_alt__(mod, alt_mod):
    '''
    import mod, if that module throws an ImportError, import alt_mod
    '''
    try:
        return __import__(mod)
    except ImportError:
        return __import__(alt_mod)


def dlist_to_dict(mapping):
    #sort list
    mapping_dict = defaultdict(list)
    func = lambda srctgt: mapping_dict[srctgt[0]].append(srctgt[1])
    map(func, mapping)
    return mapping_dict


@contextmanager
def empty_context():
    yield


def set_MAPHOST(MAPHOST):
    aex_globals.MAPHOST = str(MAPHOST)


def get_MAPHOST():
    return aex_globals.MAPHOST


def set_MAPVERS(MAPVERS):
    aex_globals.MAPVERS = float(MAPVERS)


def get_MAPVERS():
    return aex_globals.MAPVERS

## {{{ http://code.activestate.com/recipes/576862/ (r1)
"""
doc_inherit decorator

Usage:

class Foo(object):
    def foo(self):
        "Frobber"
        pass

class Bar(Foo):
    @doc_inherit
    def foo(self):
        pass

Now, Bar.foo.__doc__ == Bar().foo.__doc__ == Foo.foo.__doc__ == "Frobber"
"""

from functools import wraps


class DocInherit(object):
    """
    Docstring inheriting method descriptor

    The class itself is also used as a decorator
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            func.__doc__ = ''
        else:
            func.__doc__ = source.__doc__
        return func

doc_inherit = DocInherit
## end of http://code.activestate.com/recipes/576862/ }}}
