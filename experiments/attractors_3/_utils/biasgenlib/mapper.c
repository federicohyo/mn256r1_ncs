/* flags.c – Source file */

#include "mapper.h"

// g++ -shared -c -fPIC mapper.c -o mapper.o
// g++ -shared -Wl,-soname,library.so -o library.so mapper.o
// gcc -o mapper.so -shared -fPIC mapper.c


#include "mapper.h"
// g++ -shared -c -fPIC mapper.c -o mapper.o
// g++ -shared -Wl,-soname,library.so -o library.so mapper.o
// gcc -o mapper.so -shared -fPIC mapper.c


#include "mapper.h"

//#define _BSD_SOURCE
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>
//#include <stdint.h>

//#include <iostream.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <sys/types.h>
//#include <sys/time.h>
#include <sched.h>
#include <time.h>
#include "sched.h"

/* for PATH_MAX */
#include <linux/limits.h>

#include <assert.h>


#define u8  uint8_t
#define u32 uint32_t
#define u64 uint64_t


struct aer {
    u32 timestamp;
    u32 address;
};


void die(const char *m) {
    fprintf(stderr, "%s", m);
    perror("errno:");
    exit(1);
}

#define SIZE_JITTER_BUF 100000
/* MAIN FUNCTION C */

void programMemoryRange(unsigned int address_start,unsigned int address_stop, int value,unsigned int last4bit)
{



  int aexfd = open("/dev/aerfx2_0", O_RDWR | O_NONBLOCK);
  printf("clean memory range\n");
  int i;

  if(aexfd < 0)
    die("can not open /dev/aerfx2_0\n");

	  for(i=address_start; i<address_stop; i++){

	  unsigned int writeBuf[2];
	  writeBuf[0]=writeBuf[1]=0;
	  writeBuf[1] |= 1<<30;
	  writeBuf[1] |= last4bit;
	  writeBuf[1] |= i<<4;
	  writeBuf[0] = value;

	  unsigned int n_written = 8;
	  int res;
	  while(((res = write(aexfd,writeBuf,8))<0 && errno==EAGAIN) || (n_written-=res));

     }

 close(aexfd);
}

void programMemory(unsigned int address,int value,unsigned int last4bit)
{



  int aexfd = open("/dev/aerfx2_0", O_RDWR | O_NONBLOCK);
  if(aexfd < 0)
    die("can not open /dev/aerfx2_0\n");

  unsigned int writeBuf[2];
  writeBuf[0]=writeBuf[1]=0;
  writeBuf[1] |= 1<<30;
  writeBuf[1] |= last4bit;
  writeBuf[1] |= address<<4;
  writeBuf[0] = value;
//  int i;

	printf("program memory \n");
//	 for(i=0; i<2; i++){
//		printf("%d\n", writeBuf[i]);
//		}


  unsigned int n_written = 8;
  int res;
  while(((res = write(aexfd,writeBuf,8))<0 && errno==EAGAIN) || (n_written-=res));

 close(aexfd);
}

void programBulkSpec(unsigned int interfaceNumber,unsigned int bulkMask)
{



  int aexfd = open("/dev/aerfx2_0", O_RDWR | O_NONBLOCK);
  if(aexfd < 0)
    die("can not open /dev/aerfx2_0\n");

  unsigned int writeBuf[2];
  writeBuf[0]=writeBuf[1]=0;
  writeBuf[1] |= 1<<30;
  writeBuf[1] |= 1<<29;
  writeBuf[1] |= interfaceNumber;
  writeBuf[0] = bulkMask;
  unsigned int n_written = 8;
  int res;
//int i;
printf("program bulk \n");
// for(i=0; i<2; i++){
//	printf("%d\n", writeBuf[i]);
//	}

  while(((res = write(aexfd,writeBuf,8))<0 && errno==EAGAIN) || (n_written-=res));
close(aexfd);

}

void programDetailMapping(unsigned int detailMappingMask)
{


  int aexfd = open("/dev/aerfx2_0", O_RDWR | O_NONBLOCK);
  if(aexfd < 0)
    die("can not open /dev/aerfx2_0\n");

  unsigned int writeBuf[2];
  writeBuf[0]=writeBuf[1]=0;
  writeBuf[1] |= 1<<30;
  writeBuf[1] |= 1<<29;
  writeBuf[1] |= 1<<28;
  writeBuf[0] = detailMappingMask;
//  int i;
//printf("program detail mapping \n");
 //for(i=0; i<2; i++){
//	printf("%d\n", writeBuf[i]);
	//}

  unsigned int n_written = 8;
  int res;
  while(((res = write(aexfd,writeBuf,8))<0 && errno==EAGAIN) || (n_written-=res));
close(aexfd);

}

void programAddressRange( unsigned int interfaceNumber , unsigned int addressRange)
{

  int aexfd = open("/dev/aerfx2_0", O_RDWR | O_NONBLOCK);
  if(aexfd < 0)
    die("can not open /dev/aerfx2_0\n");

  unsigned int writeBuf[2];
  writeBuf[0]=writeBuf[1]=0;
  writeBuf[1] |= 1<<30;
  writeBuf[1] |= 1<<31;
  writeBuf[1] |= interfaceNumber;
  writeBuf[0] = addressRange;
//int i;
printf("program address range \n");
// for(i=0; i<2; i++){
//	printf("%d\n", writeBuf[i]);
//	}

  unsigned int n_written = 8;
  int res;
  while(((res = write(aexfd,writeBuf,8))<0 && errno==EAGAIN) || (n_written-=res));

close(aexfd);

}


void programOffset(unsigned int interfaceNumber,unsigned int offset)
{


  int aexfd = open("/dev/aerfx2_0", O_RDWR | O_NONBLOCK);
  if(aexfd < 0)
    die("can not open /dev/aerfx2_0\n");

  unsigned int writeBuf[2];
  writeBuf[0]=writeBuf[1]=0;
  writeBuf[1] |= 1<<30;
  writeBuf[1] |= 1<<28;
  writeBuf[1] |= interfaceNumber;
  writeBuf[0] = offset;
//int i;
//printf("program offset \n");
// for(i=0; i<2; i++){
//	printf("%d\n", writeBuf[i]);
//	}


  unsigned int n_written = 8;
  int res;
  while(((res = write(aexfd,writeBuf,8))<0 && errno==EAGAIN) || (n_written-=res));

 close(aexfd);

}


  
