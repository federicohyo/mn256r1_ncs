/* mapper.h – Header file */

void die(const char *m);
void programMemoryRange(unsigned int address_start,unsigned int address_stop, int value,unsigned int last4bit);
void programMemory(unsigned int address,int value,unsigned int last4bit);
void programBulkSpec(unsigned int interfaceNumber,unsigned int bulkMask);
void programDetailMapping(unsigned int detailMappingMask);
void programAddressRange( unsigned int interfaceNumber , unsigned int addressRange);
void programOffset(unsigned int interfaceNumber,unsigned int offset);

