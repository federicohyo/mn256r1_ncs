from pyNCS.api.ConfAPI import MappingsBase
import pyAex.mapclient as mapclient
from pyAex.aexutils import doc_inherit, getuser, get_MAPHOST, get_MAPVERS
from pyAex.aexutils import set_MAPHOST, set_MAPVERS


class Mappings(MappingsBase):

    @doc_inherit
    def __init__(self, host=None, user='root', version=None, debug=False, *args, **kwargs):
        self._host = host
        self._vers = eval(version)
        self._user = str(user)
        self.debug = debug
        super(self.__class__, self).__init__()

    @property
    def host(self):
        if self._host == None:
            return get_MAPHOST()
        else:
            return self._host

    @property
    def user(self):
        if self._user == None:
            return getuser()
        else:
            return self._user

    @property
    def vers(self):
        if self._vers == None:
            return get_MAPVERS()
        else:
            return self._vers

    @doc_inherit
    def add_mappings(self, mappings):
        mapclient.setMappings(mappings, self.user, self.host, self.vers)
        if self.debug:
            print "Successfully written {0} connections".format(len(mappings))

    @doc_inherit
    def get_mappings(self):
        cur_table = mapclient.getMapping(self.user, self.host, self.vers)
        if self.vers < 3:
            cur_table = cur_table.reshape(-1, 2)
        elif self.vers == 3:
            cur_table = cur_table.reshape(-1, 3)
        return cur_table

    @doc_inherit
    def clear_mappings(self):
        mapclient.clearAllMappings(self.user, self.host, self.vers)
        return None
