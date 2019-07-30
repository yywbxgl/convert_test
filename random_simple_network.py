#!/usr/bin/python3
#-*- coding: utf-8 -*-
import sys
import numpy as np
from functools import reduce
import itertools

def random_graph():
	ret = []
	#input feature map 
	C = np.random.randint(1,17)
	H = np.random.randint(5,300)
	ret.append({"type":"Input","shape":(C,H,H)})

	#convolution
	choice_list = list(filter(lambda x: True if (x[2]==0 or x[2]==x[0]//2) and x[0]>=x[1] and H+x[2]*2-x[0]>=0 and (H+x[2]*2-x[0])%x[1]==0 else False,
		itertools.product(range(1,12,2), range(1, 12), range(0,6))))
	if len(choice_list)==0:
		return None
	#1~300
	num_output = np.random.randint(1, 301)
	kernel_size, stride, pad = choice_list[np.random.randint(0,len(choice_list))]
	ret.append({"type":"Convolution",
		"num_output":num_output,
		"kernel_size":kernel_size,
		"stride":stride,
		"pad":pad})
	X = (H+pad*2-kernel_size)//stride+1
	shape = (num_output, X, X)

	#relu
	if np.random.randint(0, 2) == 0:
		ret.append({"type":"ReLU"})

	#MaxPolling
	if np.random.randint(0, 2) == 0:
		choice_list = list(filter(lambda x: True if x[0]>=x[1] and X-x[0]>=0 and (X-x[0])%x[1]==0 else False,
			itertools.product(range(2,8), range(2,8))))
		if len(choice_list)==0:
			return None
		kernel_size, stride = choice_list[np.random.randint(0,len(choice_list))]
		ret.append({"type":"Pooling",
			"poll":"MAX",
			"kernel_size":kernel_size,
			"stride":stride})
		Y = (X-kernel_size)//stride+1
		shape = (shape[0], Y, Y)
	
	#full connection
	#2~20
	num_output = np.random.randint(2, 21)
	ret.append({"type":"InnerProduct",
		"num_output":num_output,
		"num_input":reduce(lambda a,b:a*b,shape,1)})
	
	print(ret)
	return ret

def random_result(arg):
	global root_dir
	prototxt = root_dir + 'deploy.prototxt'
	with open(prototxt, 'w') as f:
		print(prototxt)
		s = 'name: "MyNet"\n'
		for i in arg:
			if i["type"] == "Input":
				s += 'layer{\n'
				s += '  name:"data"\n'
				s += '  type:"%s"\n' % (i["type"],)
				s += '  top:"data"\n'
				s += '  input_param:{shape: {dim:1 dim:%d dim:%d dim:%d}}\n' % i["shape"]
				s += '}\n'
				layer_name = "data"
			elif i["type"] == "Convolution":
				s += 'layer{\n'
				s += '  name:"conv1"\n'
				s += '  type:"%s"\n' % (i["type"],)
				s += '  bottom: "%s"\n' % (layer_name,)
				s += '  top: "conv1"\n'
				s += '  convolution_param {\n'
				s += '    num_output: %d\n' % (i["num_output"],)
				s += '    kernel_size: %d\n' % (i["kernel_size"],)
				s += '    stride: %d\n' % (i["stride"],)
				s += '    pad:%d\n' % (i["pad"],)
				s += '  }\n'
				s += '}\n'
				layer_name = "conv1"
			elif i["type"] == "ReLU":
				s += 'layer {\n'
				s += '  name: "relu1"\n'
				s += '  type:"%s"\n' % (i["type"],)
				s += '  bottom: "%s"\n' % (layer_name,)
				#in_space
				#s += '  top: "relu1"\n'
				s += '  top: "%s"\n' % (layer_name,)
				s += '}\n'
				#layer_name = "relu1"
			elif i["type"] == "Pooling":
				s += 'layer {\n'
				s += '  name: "pool1"\n'
				s += '  type:"%s"\n' % (i["type"],)
				s += '  bottom: "%s"\n' % (layer_name,)
				s += '  top: "pool1"\n'
				s += '  pooling_param {\n'
				s += '    pool: MAX\n'
				s += '    kernel_size: %d\n' % (i["kernel_size"],)
				s += '    stride: %d\n' % (i["stride"],)
				s += '  }\n'
				s += '}\n'
				layer_name = "pool1"
			elif i["type"] == "InnerProduct":
				s += 'layer {\n'
				s += '  name: "ip1"\n'
				s += '  type:"%s"\n' % (i["type"],)
				s += '  bottom: "%s"\n' % (layer_name,)
				s += '  top: "ip1"\n'
				s += '  inner_product_param {\n'
				s += '    num_output: %d\n' % (i["num_output"],)
				s += '  }\n'
				s += '}\n'

		f.write(s)
	sys.stdout.write(open(prototxt, 'r').read())
	# to do, numpy

if __name__ == '__main__':
	root_dir = './'
	if len(sys.argv) > 1:
		root_dir = sys.argv[1]

	if len(root_dir) == 0:
		root_dir = './'
	elif root_dir[-1] != '/':
		root_dir += '/'
	
	while True:
		graph = random_graph()
		if graph != None:
			break
	random_result(graph)
