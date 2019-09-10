#include <stdio.h>
#include <stdlib.h>
#include "cal_bank.h"
int split_conv(const conv_arg_t *pc, conv_block_t **pblk)
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
		fprintf(stderr, "%s:%d", __FILE__, __LINE__);
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

int write_conf(const conv_arg_t *pc, const conv_block_t *blk, const char* name_conf_file)
{
	FILE *f;
	int src_line_stride, src_surface_stride, dst_line_stride, dst_surface_stride;
	int shape[3];
	
	src_line_stride = 32 * pc->datain_shape[2];
	src_surface_stride = src_line_stride * pc->datain_shape[1];
	dst_line_stride = 32 * pc->dataout_shape[2];
	dst_surface_stride = dst_line_stride * pc->dataout_shape[1];

	if((f = fopen(name_conf_file, "wb")) == NULL) {
		fprintf(stderr, "%s:%d", __FILE__, __LINE__);
		return 1;
	}

	fprintf(f, "{\n");
	fprintf(f, "\t\"group\" : 0,\n");
	fprintf(f, "\t\"src_reuse\" : 0,\n");
	fprintf(f, "\t\"src_addr\" : %u,\n", (unsigned)(pc->addr_datain-pc->addr_dram)+DRAM_PHY_ADDR+src_line_stride*blk->datain_height_start);
	fprintf(f, "\t\"src_line_stride\" : %d,\n", src_line_stride);
	fprintf(f, "\t\"src_surface_stride\" : %d,\n", src_surface_stride);
	fprintf(f, "\t\"src_line_pack\" : 1,\n");
	if(blk->datain_height == pc->datain_shape[1]) {
		fprintf(f, "\t\"src_surface_pack\" : %d,\n", 1);
	} else {
		fprintf(f, "\t\"src_surface_pack\" : %d,\n", 0);
	}
	fprintf(f, "\t\"src_c\" : %d,\n", pc->datain_shape[0]);
	fprintf(f, "\t\"src_h\" : %d,\n", pc->datain_shape[1]);
	fprintf(f, "\t\"src_w\" : %d,\n", pc->datain_shape[2]);
	shape[0] = pc->datain_shape[0];
	shape[0] = blk->datain_height;
	shape[2] = pc->datain_shape[2];
	fprintf(f, "\t\"feature_bank\" : %d,\n", banks_featuremap_cal(shape));
	fprintf(f, "\n\n");


	fprintf(f, "\t\"weight_reuse\" : 0,\n");
	fprintf(f, "\t\"weight_addr\" : %u,\n", (unsigned)(pc->addr_weight-pc->addr_dram)+DRAM_PHY_ADDR);
	fprintf(f, "\t\"weight_k\" : %d,\n", pc->weight_shape[0]);
	fprintf(f, "\t\"weight_c\" : %d,\n", pc->weight_shape[1]);
	fprintf(f, "\t\"weight_h\" : %d,\n", pc->weight_shape[2]);
	fprintf(f, "\t\"weight_w\" : %d,\n", pc->weight_shape[3]);
	if(blk->datain_height_start == 0) {
		fprintf(f, "\t\"pad_top\" : %d,\n", pc->conv_pad);
	} else {
		fprintf(f, "\t\"pad_top\" : %d,\n", 0);
	}
	if(blk->datain_height_start + blk->datain_height == pc->datain_shape[1]) {
		fprintf(f, "\t\"pad_bottom\" : %d,\n", pc->conv_pad);
	} else {
		fprintf(f, "\t\"pad_bottom\" : %d,\n", 0);
	}
	fprintf(f, "\t\"pad_left\" : %d,\n", pc->conv_pad);
	fprintf(f, "\t\"pad_right\" : %d,\n", pc->conv_pad);
	fprintf(f, "\t\"h_stride\" : %d,\n", pc->conv_stride);
	fprintf(f, "\t\"w_stride\" : %d,\n", pc->conv_stride);
	fprintf(f, "\t\"h_dilation\" : %d,\n", pc->conv_dilation);
	fprintf(f, "\t\"w_dilation\" : %d,\n", pc->conv_dilation);
	fprintf(f, "\t\"weight_bank\" : %d,\n", banks_weight_cal(pc->weight_shape));
	fprintf(f, "\n\n");



	fprintf(f, "\t\"dst_addr\" : %u,\n", (unsigned)(pc->addr_dataout-pc->addr_dram)+DRAM_PHY_ADDR+dst_line_stride*blk->dataout_height_start);
	fprintf(f, "\t\"dst_line_stride\" : %d,\n", dst_line_stride);
	fprintf(f, "\t\"dst_surface_stride\" : %d,\n", dst_surface_stride);
	fprintf(f, "\t\"dst_line_pack\" : 1,\n");
	if(blk->dataout_height == pc->dataout_shape[1]) {
		fprintf(f, "\t\"dst_surface_pack\" : %d,\n", 1);
	} else {
		fprintf(f, "\t\"dst_surface_pack\" : %d,\n", 0);
	}
	fprintf(f, "\t\"dst_c\" : %d,\n", pc->dataout_shape[0]);
	fprintf(f, "\t\"dst_h\" : %d,\n", pc->dataout_shape[1]);
	fprintf(f, "\t\"dst_w\" : %d,\n", pc->dataout_shape[2]);
	fprintf(f, "\n\n");


	fprintf(f, "\t\"sdp_cvt_offset\" : %d,\n", pc->cvt_offset);
	fprintf(f, "\t\"sdp_cvt_scale\" : %d,\n", pc->cvt_offset);
	fprintf(f, "\t\"sdp_cvt_shift\" : %d\n", pc->cvt_offset);
	fprintf(f, "}\n");
	
	fclose(f);
	return 0;
}
