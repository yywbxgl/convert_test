#include <stdio.h>
#include <stdlib.h>
#include "cal_bank.h"
int split_conv(conv_arg_t *pc, conv_block_t **pblk)
{
	int banks_weight;
	int max_height;
	int dataout_height_start;
	int conv_height;
	int datain_height_next;
	int dataout_height_next;
	conv_block_t *p, *blk;
	int blk_len = 0;

	banks_weight = banks_weight_cal(pc->weight_shape);
	if(banks_weight > BANKS_TOTAL - 1) {
		/*暂时不考虑这么大的卷积核*/
		return -1;
	}
	max_height = max_height_in_banks(BANKS_TOTAL-banks_weight, pc->datain_shape[0], pc->datain_shape[2]);
	
	conv_height = 1+(pc->weight_shape[2]-1)*pc->conv_dilation;
	dataout_height_start = 0;
	blk = NULL;
	blk_len = 0;
	while(1) {
		blk = realloc(blk, sizeof(*blk)*(++blk_len));
		p = &blk[blk_len-1];
		p->dataout_height_start = dataout_height_start;
		p->datain_height_start = pc->conv_stride*dataout_height_start - pc->conv_pad;
		if(p->datain_height_start <= 0) {
			p->datain_height_start = 0;
		}
		p->datain_height = max_height;
		if(p->datain_height_start + p->datain_height >= pc->datain_shape[1]) {
			p->datain_height = pc->datain_shape[1] - p->datain_height_start; 
			p->dataout_height = pc->dataout_shape[1] - p->dataout_height_start;
			break;
		}
		datain_height_next = p->datain_height_start + p->datain_height;
		dataout_height_next = 1 + (datain_height_next - conv_height) / pc->conv_stride;
		p->dataout_height = dataout_height_next - p->dataout_height_start;
		datain_height_next = pc->conv_stride*(dataout_height_next-1) + conv_height - pc->conv_pad;
		p->datain_height = datain_height_next - p->datain_height_start;
		dataout_height_start = dataout_height_next;
	}

	*pblk = blk;
	return blk_len;
}


