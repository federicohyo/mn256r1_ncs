# mapper gui
# author: Marc Osswald - marc@ini.phys.ethz.ch
# Mar 2014

from MappingTable import MappingTable

class BasicMapper():

    mappingTable = MappingTable()

    def __init__(self, n=255):
        for i in range(0,n):
            self.mappingTable.add(i,i)

    #default mapping function
    def map(self, src):
        return self.mappingTable.get(src)