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
#include "include/conv_arg.h"

conv_arg_t conv_arg; /*所处理卷积的参数*/

static int parse_arg(int argc, char **argv)
{
	int ret;

	while((ret = getopt(argc, argv, "d:w:D:s:p:")) >= 0) {
		switch((char)ret) {
			case 'D':
				conv_arg.featuremap_file = optarg;
				break;
			case 'W':
				conv_arg.weight_file = optarg;
				break;
			case 'd':
				conv_arg.conv_dilation = atoi(optarg);
				break;
			case 's':
				conv_arg.conv_stride = atoi(optarg);
				break;
			case 'p':
				conv_arg.conv_pad = atoi(optarg);
				break;
			default:
				return -1;
				break;
		}
	}
	return 0;
}

static int addr_map(conv_arg_t *pc)
{
	int fd;

	fd = open("/dev/mem", O_RDWR);
	if(fd < 0) {
		perror("open");
		return -1;
	}
	if((pc->addr_dram = mmap(NULL, DRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DRAM_PHY_ADDR)) == MAP_FAILED) {
		perror("mmap");
		return -1;
	}
	close(fd);
	return 0;
}

static int put_data_to_dram(conv_arg_t *pc)
{
	tensor_int8_t t;
	int len;
	char *p = (char*)pc->addr_dram;
	
	/*数据导入到DRAM*/
	numpy_read(pc->featuremap_file, &t);
	memcpy(pc->datain_shape, t.shape, sizeof(int)*3);
	pc->addr_datain = p;
	len = featuremap_npy_to_nvdla(&t, p);
	tensor_free(&t);
	p += len;
	numpy_read(pc->weight_file, &t);
	memcpy(pc->weight_shape, t.shape, sizeof(int)*4);
	pc->addr_weght = p;
	len = weight_npy_to_nvdla(&t, p);
	tensor_free(&t);
	p += len;
	pc->addr_dataout = p;

	/*计算输出的featuremap形状*/
	pc->dataout_shape[0] = pc->weight_shape[0];
	pc->dataout_shape[1] = (pc->datain_shape[1]+pc->conv_pad+pc->conv_pad-(1+(pc->weight_shape[2]-1)*pc->conv_dilation))/pc->conv_stride+1;
	pc->dataout_shape[2] = (pc->datain_shape[2]+pc->conv_pad+pc->conv_pad-(1+(pc->weight_shape[3]-1)*pc->conv_dilation))/pc->conv_stride+1;

	return 0;
}

int main(int argc, char **argv)
{
	int ret;
	conv_block_t *blk;

	parse_arg(argc, argv);
	addr_map(&conv_arg);
	put_data_to_dram(&conv_arg);
	ret = split_conv(&conv_arg, &blk);

	return 0;
}
