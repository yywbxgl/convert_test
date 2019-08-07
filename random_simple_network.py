#!/usr/bin/python3
#-*- coding: utf-8 -*-
import sys
import numpy as np
from functools import reduce
import itertools

def _random_graph(t):
	"""
	随机产生一个简单的神经网络结构,结构放在一个list里
	这个函数import不要使用，应使用random_graph函数
	"""
	ret = []
	#input feature map 
	C = np.random.randint(1,17)
	H = np.random.randint(5,300)
	shape = (C,H,H)
	ret.append({"type":"Input",
		"name":"data",
		"shape":shape})

	#convolution
	choice_list = list(filter(lambda x: True if (x[2]==0 or x[2]==x[0]//2) and x[0]>=x[1] and H+x[2]*2-x[0]>=0 and (H+x[2]*2-x[0])%x[1]==0 else False,
		itertools.product(range(1,12,2), range(1, 12), range(0,6))))
	if len(choice_list)==0:
		return None
	#1~300
	num_output = np.random.randint(1, 301)
	kernel_size, stride, pad = choice_list[np.random.randint(0,len(choice_list))]
	ret.append({"type":"Convolution",
		"name":"conv1",
		"num_output":num_output,
		"kernel_size":kernel_size,
		"stride":stride,
		"pad":pad})
	if t=='Convolution':
		return ret
	X = (shape[1]+pad*2-kernel_size)//stride+1
	shape = (num_output, X, X)

	#relu
	if np.random.randint(0, 2) == 0:
		ret.append({"type":"ReLU",
				"name":"relu1"})

	#MaxPolling
	if np.random.randint(0, 2) == 0:
		choice_list = list(filter(lambda x: True if x[0]>=x[1] and X-x[0]>=0 and (X-x[0])%x[1]==0 else False,
			itertools.product(range(2,8), range(2,8))))
		if len(choice_list)==0:
			return None
		kernel_size, stride = choice_list[np.random.randint(0,len(choice_list))]
		ret.append({"type":"Pooling",
			"name":"pool1",
			"poll":"MAX",
			"kernel_size":kernel_size,
			"stride":stride})
		X = (shape[1]-kernel_size)//stride+1
		shape = (shape[0], X, X)
	
	#full connection
	#2~20
	num_output = np.random.randint(2, 21)
	ret.append({"type":"InnerProduct",
		"name":"ip1",
		"num_output":num_output,
		"num_input":reduce(lambda a,b:a*b,shape,1)})
	
	#print(ret)
	return ret

def random_graph(t='Convolution'):
	"""
	参数如果不传，或者传'Convolution'，则是生成单个卷积
	如果传'Network'，则是生成简单网络
	将来再考虑扩展
	"""
	if t!='Convolution' and t!='Network':
		return None
	while True:
		graph = _random_graph(t)
		if graph != None:
			return graph

def random_result(arg, root_dir):
	"""
	根据神经网络结构随机产生prototxt/featuremap/weight/bias，并按照结构里的名字为文件命名
	"""
	prototxt = root_dir + 'deploy.prototxt'
	with open(prototxt, 'w') as f:
		print(prototxt)
		s = 'name: "MyNet"\n'
		for i in arg:
			if i["type"] == "Input":
				s += 'layer{\n'
				s += '  name:"%s"\n' % (i["name"],)
				s += '  type:"%s"\n' % (i["type"],)
				s += '  top:"data"\n'
				s += '  input_param:{shape: {dim:1 dim:%d dim:%d dim:%d}}\n' % i["shape"]
				s += '}\n'
				layer_name = "data"
			elif i["type"] == "Convolution":
				s += 'layer{\n'
				s += '  name:"%s"\n' % (i["name"],)
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
				s += '  name:"%s"\n' % (i["name"],)
				s += '  type:"%s"\n' % (i["type"],)
				s += '  bottom: "%s"\n' % (layer_name,)
				#in_space
				#s += '  top: "relu1"\n'
				s += '  top: "%s"\n' % (layer_name,)
				s += '}\n'
				#layer_name = "relu1"
			elif i["type"] == "Pooling":
				s += 'layer {\n'
				s += '  name:"%s"\n' % (i["name"],)
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
				s += '  name:"%s"\n' % (i["name"],)
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
	for i in arg:
		if i["type"] == "Input":
			arr = np.random.random(i["shape"]) * 200.0
			npy = root_dir + i["name"]
			np.save(npy, arr.astype(np.float32))
			print(npy + '.npy', arr.shape)
			shape = i["shape"]
		elif i["type"] == "Convolution":
			arr = (np.random.random((i["num_output"], shape[0], i["kernel_size"], i["kernel_size"])) - 0.5)/20.0
			npy = root_dir + i["name"] + '-weight'
			np.save(npy, arr.astype(np.float32))
			print(npy + '.npy', arr.shape)
			arr = np.random.random((i["num_output"],)) - 0.5
			npy = root_dir + i["name"] + '-bias'
			np.save(npy, arr.astype(np.float32))
			print(npy + '.npy', arr.shape)
			X = (shape[1]+i["pad"]*2-i["kernel_size"])//i["stride"]+1
			shape = (i["num_output"], X, X)
		elif i["type"] == "ReLU":
			pass
		elif i["type"] == "Pooling":
			X = (shape[1]-i["kernel_size"])//i["stride"]+1
			shape = (shape[0], X, X)
		elif i["type"] == "InnerProduct":
			arr = (np.random.random((i["num_output"], reduce(lambda a,b:a*b,shape,1))) - 0.5)/20.0
			npy = root_dir + i["name"] + '-weight'
			np.save(npy, arr.astype(np.float32))
			print(npy + '.npy', arr.shape)
			arr = np.random.random((i["num_output"],)) - 0.5
			npy = root_dir + i["name"] + '-bias'
			np.save(npy, arr.astype(np.float32))
			print(npy + '.npy', arr.shape)
			shape = (i["num_output"],)


class test_inference(object):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		#搜索目录file_dir下后缀名为postfix的文件列表
		import os
		f_name = lambda file_dir, postfix : list(map((lambda x : x[1]), (lambda s : filter(lambda x : os.path.splitext(x[0])[1] == '.' + postfix, map(lambda x:(x, os.path.join(s[0],x)), s[2])))(tuple(os.walk(file_dir))[0])))
		prototxt = f_name(root_dir, 'prototxt')[0]
		self.graph = self.make_graph_by_prototxt(prototxt)
	def make_graph_by_prototxt(self, prototxt):
		"""
		从prototxt文件中导出网络结构
		"""
		s = open(prototxt, 'r').read().replace(':',' : ').replace('{',' { ').replace('}',' } ').split()
		graph = []
		level = 0
		for i in s:
			if i == '{':
				level += 1
				if level == 1:
					node = {}
			elif i == '}':
				level -= 1
				if level == 0:
					graph.append(node)
			elif i[0]>='0' and i[0]<='9' and level>=1:
				n = int(i)
				if key == 'dim':
					node['shape'] = node['shape'] + (n,) if 'shape' in node else (n,)
				else:
					node[key] = n
			elif i != ':' and i[0] != '"' and level>=1:
				key = i
			elif i[0] == '"' and level>=1:
				node[key] = i.replace('"', '')
		return graph
	def input(self, data, node):
		return np.load(self.root_dir+node["name"]+'.npy')
	def conv(self, data, node):
		weight = np.load(self.root_dir+node["name"]+'-weight'+'.npy')
		bias = np.load(self.root_dir+node["name"]+'-bias'+'.npy')
		if node["pad"] != 0:
			X = node["pad"]*2 + data.shape[1]
			d = np.zeros((data.shape[0], X, X)).astype(np.float32)
			d[:, node["pad"]:node["pad"]+data.shape[1], node["pad"]:node["pad"]+data.shape[2]] = data
		else:
			d = data
		stride = node["stride"]
		kernel_size = node["kernel_size"]
		X = (d.shape[1]-kernel_size)//stride+1
		data = np.zeros((node["num_output"], X, X)).astype(np.float32)
		for c,h,w in itertools.product(*map(lambda x:range(x),data.shape)):
			data[c,h,w] = np.sum(np.multiply(
						d[:, h*stride:h*stride+kernel_size, w*stride:w*stride+kernel_size],
						weight[c]
						)) + bias[c]
		return data
	def relu(self, data, node):
		return np.where(data>0.0,data,0.0)
	def pool(self, data, node):
		stride = node["stride"]
		kernel_size = node["kernel_size"]
		X = (data.shape[1]-kernel_size)//stride+1
		d = np.zeros((data.shape[0], X, X)).astype(np.float32)
		for c,h,w in itertools.product(*map(lambda x:range(x),d.shape)):
			d[c,h,w] = np.max(data[c, h*stride:h*stride+kernel_size, w*stride:w*stride+kernel_size])
		return d
	def fc(self, data, node):
		return np.matmul(data.reshape(reduce(lambda a,b:a*b, data.shape, 1)), np.load(self.root_dir+node["name"]+'-weight'+'.npy').transpose(1,0)) + np.load(self.root_dir+node["name"]+'-bias'+'.npy')
	def inference(self):
		op_table = {
			"Input" : self.input,
			"Convolution" : self.conv,
			"ReLU" : self.relu,
			"Pooling" : self.pool,
			"InnerProduct" : self.fc
		}
		return reduce(lambda data, node : op_table[node["type"]](data, node), self.graph, None)
		
def inference(root_dir):
	"""
	对简单网络做推理
	"""
	x = test_inference(root_dir)
	print('Inferencing...Please wait')
	data = x.inference()
	npy = root_dir + "inference-output-result"
	np.save(npy, data.astype(np.float32))
	print(npy + '.npy', data.shape)
	print(data)
		
if __name__ == '__main__':
	root_dir = './'
	if len(sys.argv) > 1:
		root_dir = sys.argv[1]

	if len(root_dir) == 0:
		root_dir = './'
	elif root_dir[-1] != '/':
		root_dir += '/'

	# 随机生成简单网络，如果random_graph不带参数，则只生成一个卷积
	graph = random_graph('Network')
	# 将prototxt和相应的npy生成在root_dir所存的目录
	random_result(graph, root_dir)
	# numpy下的推理，比较慢，但可以拿来对比
	inference(root_dir)
	
