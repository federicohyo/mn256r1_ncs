# Module implementing the mapper portion of the configuration API defined by
# the NCS tools

from aexutils import *
import aex_globals


def _get_user_host_vers(user, host, vers):
    user = default_user(user)
    if host == None:
        host = aex_globals.get_MAPHOST()
    if vers == None:
        vers = aex_globals.get_MAPVERS()
    else:
        vers = float(vers)
    return user, host, vers


def clearAllMappings(user='root', host=None, vers=1):
    user, host, vers = _get_user_host_vers(user, host, vers)

    if vers < 3:
        system(
            'ssh -C ' + user + '@' + host + ' "mappinglib/clearallmappings"')
    else:
        system(
            'ssh -C ' + user + '@' + host + ' "mappinglib/clearMappingTable"')
    return None


def getMapping(user='root', host=None, vers=1):
    user, host, vers = _get_user_host_vers(user, host, vers)

    filename = '/tmp/map' + getuser()
    if vers < 3:
        system('ssh -C ' + user + '@' + host +
             ' "mappinglib/getallmappings >' + filename + '"')
    elif vers == 3:
        system('ssh -C ' + user + '@' + host +
             ' "mappinglib/printMappingTable >' + filename + '"')
    system('scp ' + user + '@' + host + ':' + filename + ' ' + filename)
    if vers < 3:
        try:
            return np.loadtxt(filename, dtype='float')
        except IOError, e:
            print e
            return np.zeros([2, 0])
    elif vers == 3:
        try:
            #numpy bug: if filename is empty, then an indexerror is returned.
            return np.loadtxt(filename, dtype='float', converters={0: float.fromhex, 1: float.fromhex, 2: float})
        except (IOError,IndexError), e:
            print e
            return np.zeros([3, 0])


def setMappings(mappings, user='root', host=None, vers=1):
    user, host, vers = _get_user_host_vers(user, host, vers)

    tdir = '/tmp/'
    mappings = np.array(mappings, dtype='uint32')

    if vers <= 1:
        assert mappings.shape[
            1] == 2, "mappings must be a Nx2 array (shape (N,2))"
        fmt = '%u %u'
    elif vers <= 2:
        assert mappings.shape[
            1] == 2, "mappings must be a Nx2 array (shape (N,2))"
        fmt = '%x %x'
    elif vers <= 3:
        assert mappings.shape[
            1] == 3, "mappings must be a Nx3 array (shape (N,3))"
        fmt = '%x %x %u'
    else:
        raise ValueError("Mapper version {0} is unknown".format(vers))

    np.savetxt(
        tdir + '/tmp_mapping_' + getuser(), mappings, delimiter=' ', fmt=fmt)
    system('scp ' + tdir + '/tmp_mapping_' + getuser() + ' ' + user + '@' + host +
         ':/tmp/tmp_mapping')
    if vers < 3:
        system('ssh -C ' + user + '@' + host +
             ' "mappinglib/setmapping /tmp/tmp_mapping"')
    else:
        system('ssh -C ' + user + '@' + host +
             ' "mappinglib/setMapping /tmp/tmp_mapping"')
    return None


def addMappings(mappings, user='root', host=None, vers=1):
    user, host, vers = _get_user_host_vers(user, host, vers)

    tdir = '/tmp/'
    mappings = np.array(mappings, dtype='uint32')

    if vers <= 1:
        assert mappings.shape[
            1] == 2, "mappings must be a Nx2 array (shape (N,2))"
        fmt = '%u %u'
    elif vers <= 2:
        assert mappings.shape[
            1] == 2, "mappings must be a Nx2 array (shape (N,2))"
        fmt = '%x %x'
    elif vers <= 3:
        assert mappings.shape[
            1] == 3, "mappings must be a Nx3 array (shape (N,3))"
        fmt = '%x %x %u'
    else:
        raise ValueError("Mapper version {0} is unknown".format(vers))

    np.savetxt(
        tdir + '/tmp_mapping_' + getuser(), mappings, delimiter=' ', fmt=fmt)
    system('scp ' + tdir + '/tmp_mapping_' + getuser() + ' ' + user + '@' + host +
         ':/tmp/tmp_mapping')
    system('ssh -C ' + user + '@' + host +
         ' "mappinglib/addMapping /tmp/tmp_mapping"')
    return None


def removeMappings(mappings, user='root', host=None, vers=1):
    user, host, vers = _get_user_host_vers(user, host, vers)

    tdir = '/tmp/'
    mappings = np.array(mappings, dtype='uint32')

    if vers < 3:
        raise NotImplementedError(
            'removal of mappings is only supported in version 3')
    elif vers == 3:
        assert mappings.shape[1] == 3 or mappings.shape[
            1] == 2, "mappings must be a Nx2 or Nx3 array"
        fmt = '%x %x %u'
    else:
        raise ValueError("Mapper version {0} is unknown".format(vers))

    np.savetxt(
        tdir + '/tmp_mapping_' + getuser(), mappings, delimiter=' ', fmt=fmt)
    system('scp ' + tdir + '/tmp_mapping_' + getuser() + ' ' + user + '@' + host +
         ':/tmp/tmp_mapping')
    system('ssh -C ' + user + '@' + host +
         ' "mappinglib/removeMapping /tmp/tmp_mapping"')
    return None
