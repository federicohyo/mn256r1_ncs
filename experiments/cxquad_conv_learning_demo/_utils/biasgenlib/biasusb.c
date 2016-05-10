/* flags.c – Source file */

#include "biasusb.h"
// g++ -shared -c -fPIC biasusb.c -o biasusb.o
// g++ -shared -Wl,-soname,library.so -o library.so biasusb.o
// gcc -o biasusb.so -shared -fPIC biasusb.c


#include "biasusb.h"

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



void programBias(unsigned int programbits,unsigned int biasbranch)
{


  int aexfd = open("/dev/aerfx2_0", O_RDWR | O_NONBLOCK);
  if(aexfd < 0)
    die("can not open /dev/aerfx2_0\n");

  unsigned int writeBuf[2];
  writeBuf[0] = writeBuf[1] = 0;
  writeBuf[0] |= 1<<31;
  writeBuf[0] |= programbits<<7;
  writeBuf[0] |= biasbranch;

	printf("eccolo: %u\n", writeBuf[0]);

  unsigned int n_written = 8;
  int res;
  while(((res = write(aexfd,writeBuf,8))<0 && errno==EAGAIN) || (n_written-=res));

 close(aexfd);

}


void send_32(unsigned int bits)
{

  int aexfd = open("/dev/aerfx2_0", O_RDWR | O_NONBLOCK);
  if(aexfd < 0)
    die("can not open /dev/aerfx2_0\n");

  int res;
  unsigned int n_written = 8;
  unsigned int writeBuf[2];
  writeBuf[0] = writeBuf[1] = 0;
  writeBuf[1] = bits;


  while(((res = write(aexfd,writeBuf,8))<0 && errno==EAGAIN) || (n_written-=res));
  if(res<0)
    die("error writing\n");
 
  //printf("wrote %d\n",res);

close(aexfd);
}

void send_32_1(unsigned int bits)
{

  int aexfd = open("/dev/aerfx2_0", O_RDWR | O_NONBLOCK);
  if(aexfd < 0)
    die("can not open /dev/aerfx2_0\n");

  int res;
  unsigned int n_written = 8;
  unsigned int writeBuf[2];
  writeBuf[0] = writeBuf[1] = 0;
  writeBuf[0] = bits;


  while(((res = write(aexfd,writeBuf,8))<0 && errno==EAGAIN) || (n_written-=res));
  if(res<0)
    die("error writing\n");
 
  //printf("wrote %d\n",res);


close(aexfd);
}
