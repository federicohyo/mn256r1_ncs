# mapping table
# author: Marc Osswald - marc@ini.phys.ethz.ch
# Mar 2014

class MappingTable():

    mappingDictionary = {}

    def add(self, src, dst):
        self.mappingDictionary[src] = dst

    def get(self, src):
        return self.mappingDictionary[src]

    def clear(self):
        self.mappingDictionary.clear()

    def remove(self, src):
        del self.mappingDictionary[src]
