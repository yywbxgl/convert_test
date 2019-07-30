#!/usr/bin/python3
#-*- coding: utf-8 -*-
import sys
import numpy as np
from functools import reduce

def random_parameters():
	ret = {}
	#1,3,5,7,9,11
	ret["kernel_size"] = np.random.randint(0, 6)*2 + 1
	#1~300
	ret["num_output"] = np.random.randint(1, 301)
	ret["stride"] = np.random.randint(1, ret["kernel_size"]+1)
	ret["pad"] = ret["kernel_size"]//2 if np.random.randint(0, 2) == 0 else 0
	C = np.random.randint(1,17)
	n = np.random.randint(1, (250-ret["kernel_size"])//ret["stride"]+1)
	H = (n-1)*ret["stride"]+ret["kernel_size"] - ret["pad"]*2
	ret["shape"] = (C, H, H)
	print(ret)
	return ret
	
def random_result(arg, root_dir):
	prototxt = root_dir + 'conv.prototxt'
	weight_npy = root_dir + 'weight'
	bias_npy = root_dir + 'bias'
	featuremap_npy = root_dir + 'featuremap'
	with open(prototxt, 'w') as f:
		f.write('name: "convolution"\n')
		f.write('layer{\n')
		f.write('  name:"data"\n')
		f.write('  type:"Input"\n')
		f.write('  top:"data"\n')
		f.write('  input_param:{shape: {dim:1 dim:%d dim:%d dim:%d}}\n' % arg["shape"])
		f.write('}\n')
		f.write('layer {\n')
		f.write('  name: "conv"\n')
		f.write('  type: "Convolution"\n')
		f.write('  bottom: "data"\n')
		f.write('  top: "conv"\n')
		f.write('  convolution_param {\n')
		f.write('    num_output: %d\n' % (arg["num_output"],))
		f.write('    kernel_size: %d\n' % (arg["kernel_size"],))
		f.write('    stride: %d\n' % (arg["stride"],))
		f.write('    pad:%d\n' % (arg["pad"],))
		f.write('  }\n')
		f.write('}\n')
	featuremap = np.random.random(arg["shape"]) * 200.0
	np.save(featuremap_npy, featuremap.astype(np.float32))
	weight = np.random.random((arg["num_output"], arg["shape"][0], arg["kernel_size"], arg["kernel_size"]))
	np.save(weight_npy, weight.astype(np.float32))
	bias = (np.random.random((arg["num_output"],)) - 0.5) * 10.0
	np.save(bias_npy, bias.astype(np.float32))
	print(featuremap_npy+'.npy')
	print(weight_npy+'.npy')
	print(bias_npy+'.npy')
	print(prototxt)
	sys.stdout.write(open(prototxt, 'r').read())
	

#如果加命令行参数，那么就是生成文件的目录
if __name__ == '__main__':
	root_dir = './'
	if len(sys.argv) > 1:
		root_dir = sys.argv[1]

	if len(root_dir) == 0:
		root_dir = './'
	elif root_dir[-1] != '/':
		root_dir += '/'
	
	param = random_parameters()
	random_result(param, root_dir)

