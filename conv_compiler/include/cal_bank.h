#ifndef __CAL_BANK_H__
#define __CAL_BANK_H__
#include "conv_arg.h"
#define BANKS_TOTAL 16
#define BANK_SIZE (32 * 1024)
int banks_featuremap_cal(const int *shape);
int banks_weight_cal(const int *shape);
int max_height_in_banks(int banks, int channel, int width);

typedef struct {
	int datain_height_start;
	int datain_height;
	int dataout_height_start;
	int dataout_height;
} conv_block_t;
int split_conv(const conv_arg_t *pc, conv_block_t **pblk);
int write_conf(const conv_arg_t *pc, const conv_block_t *blk, const char* name_conf_file);

#endif
