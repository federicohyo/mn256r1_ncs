# Differential Mapping module.
# Operates faster than the standad module because it makes the
# changes differentially - not by sending the mapping table
# each time.

from pyNCS.api.ConfAPI import MappingsBase
import pyAex.mapclient as mapclient
from pyAex.aexutils import doc_inherit, getuser, get_MAPHOST, get_MAPVERS
from pyAex.aexutils import set_MAPHOST, set_MAPVERS
import copy

class Mappings(MappingsBase):

    @doc_inherit
    def __init__(self, host=None, user='root', version=None, *args, **kwargs):
        self._host = host
        self._vers = eval(version)
        self._user = user
        self._table = []
        self._cleared = False
        super(self.__class__, self).__init__()

    @property
    def table(self):
        return self._table
    
    @table.setter
    def table(self, new_table):
        self._table = copy.copy(new_table)

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

    @property
    def cleared(self):
        return self._cleared

    @doc_inherit
    def set_mappings(self, mappings):
        if not self.cleared: self.clear_mappings()
        rem_map, add_map = self._diff(self._table, mappings)
        if len(rem_map) > 0:
            self.del_mappings(rem_map)
        if len(add_map) > 0:
            self.add_mappings(add_map)
        self.table = copy.copy(mappings)

    def _diff(self, table_pre, table_post):
        """
        Returns elements that are added to table_post and removed from table_post.
        Does NOT recover original order!!! Instead sorts according to the same order as logical addresses
        """

        A = set(tuple(i) for i in table_pre)
        B = set(tuple(i) for i in table_post)
        removed = list(A.difference(B))
        added = list(B.difference(A))

        return removed, added
        #Importing from unordered list: must sort back to initial order
        # zip(*a[::-1]) because we want to sort with respect to rightmost
        # column (this is how pyST builds it)
        #AinterB=AinterB[np.lexsort(zip(*AinterB[:,::-1]))]

    def del_mappings(self, mappings):
        if not self.cleared: self.clear_mappings()
        mapclient.removeMappings(mappings, self.user, self.host, self.vers)

    @doc_inherit
    def add_mappings(self, mappings):
        if not self.cleared: self.clear_mappings()
        if self.vers < 3:
            mapclient.setMappings(mappings, self.user, self.host, self.vers)
        else:
            mapclient.addMappings(mappings, self.user, self.host, self.vers)

    @doc_inherit
    def get_mappings(self):
        cur_table = mapclient.getMapping(self.user, self.host, self.vers)
        if self.vers < 3:
            cur_table = cur_table.reshape(-1, 2)
        elif self.vers == 3:
            cur_table = cur_table.reshape(-1, 3)
        self.table = cur_table
        return self.table

    @doc_inherit
    def clear_mappings(self):
        mapclient.clearAllMappings(self.user, self.host, self.vers)
        self._cleared = True
        self.get_mappings()
        return None
