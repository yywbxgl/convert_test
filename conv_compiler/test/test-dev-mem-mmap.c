#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv)
{
	size_t size;
	off_t phy_addr;
	unsigned tmp;
	unsigned char *p;
	int offset;
	unsigned char v;
	int fd;

	fd = open("/dev/mem", O_RDWR);

	sscanf(argv[1], "%x", &tmp);
	phy_addr = tmp;
	sscanf(argv[2], "%u", &tmp);
	size = tmp;
	if((p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, phy_addr)) == MAP_FAILED) {
		perror("mmap");
	}
	close(fd);
	
	if(strcmp(argv[3], "clear") == 0) {
		memset((void*)p, 0, size);
	} else if(strcmp(argv[3], "increase") == 0) {
		size_t i;
		v = 0;
		for(i=0;i<size;i++) {
			p[i] = v++;
		}
	} else if(strcmp(argv[3], "decrease") == 0) {
		size_t i;
		v = 0;
		for(i=0;i<size;i++) {
			p[i] = v--;
		}
	} else if(strcmp(argv[3], "read") == 0) {
		int i;
		for(i=4;i<argc;i++) {
			sscanf(argv[i], "%i", &offset);
			printf("%hhu\n", p[offset]);
		}
	} else if(strcmp(argv[3], "write") == 0) {
		int i;
		for(i=4;i<argc;i++) {
			sscanf(argv[i], "%i,%hhu", &offset, &v);
			p[offset] = v;
		}
	} else {
		printf("ERROR\n");
	}

	munmap(p, size);
	return 0;
}
