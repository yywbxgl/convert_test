#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "cal_bank.h"
#include "data_format.h"
#include "npy_array.h"

#define DRAM_PHY_ADDR 0x40000000
#define DRAM_SIZE 0x40000000

char *featuremap_file; /*输入featuremap的numpy文件*/
char *weight_file; /*卷积核的numpy文件*/
int conv_stride; /*卷积的步长*/
int conv_dilation; /*卷积的dilation*/
int conv_pad; /*卷积核的H,W*/
void *addr_dram; /*DRAM的地址*/
void *addr_datain; /*输入的featuremap地址*/
void *addr_dataout; /*输出的featuremap地址*/
void *addr_weght; /*卷积核的地址*/
int datain_shape[3]; /*输入的featuremap形状,(C,H,W)*/
int dataout_shape[3]; /*输出的featuremap形状,(C,H,W)*/
int weight_shape[4]; /*卷积核的形状,(K,C,H,W)*/

static int parse_arg(int argc, char **argv)
{
	int ret;

	while((ret = getopt(argc, argv, "d:w:D:s:p:")) >= 0) {
		switch((char)ret) {
			case 'D':
				featuremap_file = optarg;
				break;
			case 'W':
				weight_file = optarg;
				break;
			case 'd':
				conv_dilation = atoi(optarg);
				break;
			case 's':
				conv_stride = atoi(optarg);
				break;
			case 'p':
				conv_pad = atoi(optarg);
				break;
			default:
				return -1;
				break;
		}
	}
	return 0;
}

static int addr_map(void)
{
	int fd;

	fd = open("/dev/mem", O_RDWR);
	if(fd < 0) {
		perror("open");
		return -1;
	}
	if((addr_dram = mmap(NULL, DRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DRAM_PHY_ADDR)) == (void*)-1) {
		perror("mmap");
		return -1;
	}
	close(fd);
	return 0;
}

static int put_data_to_dram(void)
{
	tensor_int8_t t;
	int len;
	char *p = (char*)addr_dram;
	
	/*数据导入到DRAM*/
	numpy_read(featuremap_file, &t);
	memcpy(datain_shape, t.shape, sizeof(int)*3);
	addr_datain = p;
	len = featuremap_npy_to_nvdla(&t, p);
	tensor_free(&t);
	p += len;
	numpy_read(weight_file, &t);
	memcpy(weight_shape, t.shape, sizeof(int)*4);
	addr_weght = p;
	len = weight_npy_to_nvdla(&t, p);
	tensor_free(&t);
	p += len;
	addr_dataout = p;

	/*计算输出的featuremap形状*/
	dataout_shape[0] = weight_shape[0];
	dataout_shape[1] = (datain_shape[1]+conv_pad+conv_pad-(1+(weight_shape[2]-1)*conv_dilation))/conv_stride+1;
	dataout_shape[2] = (datain_shape[2]+conv_pad+conv_pad-(1+(weight_shape[3]-1)*conv_dilation))/conv_stride+1;

	return 0;
}

int main(int argc, char **argv)
{
	parse_arg(argc, argv);
	addr_map();
}
