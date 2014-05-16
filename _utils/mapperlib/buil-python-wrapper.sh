swig -python pyflags.i
gcc -fPIC -c mapper.c pyflags_wrap.c -I/usr/include/python2.7
ld -shared mapper.o pyflags_wrap.o -o _mapper_wrap.so
