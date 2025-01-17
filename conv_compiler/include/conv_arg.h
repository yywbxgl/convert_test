#ifndef __CONV_ARG_H__
#define __CONV_ARG_H__
extern unsigned DRAM_PHY_ADDR, DRAM_SIZE;
typedef struct {
	char *featuremap_file; /*输入featuremap的numpy文件*/
	char *weight_file; /*卷积核的numpy文件*/
	int conv_stride; /*卷积的步长*/
	int conv_dilation; /*卷积的dilation*/
	int conv_pad; /*卷积核的H,W*/
	void *addr_dram; /*DRAM的地址*/
	void *addr_datain; /*输入的featuremap地址*/
	void *addr_dataout; /*输出的featuremap地址*/
	void *addr_weight; /*卷积核的地址*/
	int datain_shape[3]; /*输入的featuremap形状,(C,H,W)*/
	int dataout_shape[3]; /*输出的featuremap形状,(C,H,W)*/
	int weight_shape[4]; /*卷积核的形状,(K,C,H,W)*/
	int cvt_offset; /*sdp_out_cnt的offset*/
	int cvt_scale; /*sdp_out_cnt的scale*/
	int cvt_shift; /*sdp_out_cnt的rshift*/
} conv_arg_t;
extern conv_arg_t conv_arg;
#endif
