#!/usr/bin/python3
#-*- coding: utf-8 -*-
"""
本程序用于对mobilenet的仿真
"""
import sys
import numpy as np
from functools import reduce
import itertools

def _random_graph(conf, t):
	"""
	测试过程中，一般以下超参是被指定的，而不是随机
	所以具体测试时，手动修改指定超参
	"""
	ret = []
	#input feature map 
	#组合算子
	compose = lambda *lst : lst[0] if len(lst)==1 else lambda *s : lst[0](lst[1](*s)) if len(lst)==2 else lst[0](compose(*lst[1:])(*s))
	arr = lambda name : (lambda a : a if isinstance(a, tuple) else list(range(*a)))(conf[name])
	#rand = lambda name : (lambda v:v[np.random.randint(len(v))])(arr(name))
	random_select = lambda lst:lst[np.random.randint(len(lst))]
	rand = compose(random_select, arr)
	C = rand("C")
	H = rand("H")
	shape = (C,H,H)
	ret.append({"type":"Input",
		"name":"data",
		"shape":shape})

	#convolution
	choice_list = list(filter(lambda x:(lambda conv_size, stride, pad : pad in (0, conv_size//2) and conv_size>=stride and H+pad*2-conv_size>=0 and (H+pad*2-conv_size)%stride==0)(1+x[3]*(x[0]-1), x[1], x[2]),
		itertools.product(arr("kernel_size"), arr("stride"), arr("pad"), arr("dilation"))))
	if len(choice_list)==0:
		return None
	num_output = rand("K")
	kernel_size, stride, pad, dilation = random_select(choice_list)
	ret.append({"type":"Convolution",
		"name":"conv1",
		"num_output":num_output,
		"kernel_size":kernel_size,
		"stride":stride,
		"pad":pad,
		"dilation":dilation})
	if t=='Convolution':
		return ret
	X = (shape[1]+pad*2-kernel_size)//stride+1
	shape = (num_output, X, X)

	return ret

def _check(graph):
	bank_size = 32 * 1024
	for i in graph:
		if i["type"] == "Input":
			shape = i["shape"]
			#feature_size = ((shape[0]+31)//32*32) * shape[1] * ((shape[2]+1)//2*2)
			c_groups = (shape[0]+31)//32
			entries = (c_groups//2)*shape[2] + (c_groups%2)*((shape[2]+1)//2)
			feature_size = (entries*shape[1]+511)//512*512*64
			feature_banks = (feature_size + (bank_size-1)) // bank_size
		elif i["type"] == "Convolution":
			weight_size = i["num_output"] * shape[0] * i["kernel_size"] * i["kernel_size"]
			weight_size = (weight_size + 511) // 512 * 512
			weight_banks = (weight_size + (bank_size-1)) // bank_size
			#if weight_banks + feature_banks > 16:
			#	return False
			X = (shape[1]+i["pad"]*2-i["kernel_size"])//i["stride"]+1
			shape = (i["num_output"], X, X)
		elif i["type"] == "ReLU":
			pass
		elif i["type"] == "Pooling":
			X = (shape[1]-i["kernel_size"])//i["stride"]+1
			shape = (shape[0], X, X)
		elif i["type"] == "InnerProduct":
			if reduce(lambda a,b:a*b,shape,1) > 15000:
				return False
			shape = (i["num_output"],)
	global reg
	reg['BANK:FEATURE_BANK'] = feature_banks - 1
	reg['BANK:WEIGHT_BANK'] = weight_banks - 1
	return True

def random_graph(conf, t='Convolution'):
	"""
	参数如果不传，或者传'Convolution'，则是生成单个卷积
	如果传'Network'，则是生成简单网络
	将来再考虑扩展
	"""
	if t!='Convolution' and t!='Network':
		return None
	while True:
		graph = _random_graph(conf, t)
		if graph != None and _check(graph) == True:
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
				s += '    dilation:%d\n' % (i["dilation"],)
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

	global reg
	for i in arg:
		if i["type"] == "Input":
			reg['FEATURE:IN:C'], reg['FEATURE:IN:H'], reg['FEATURE:IN:W'] = map(lambda x:x-1,i["shape"])
			arr = np.random.randint(-128,128,size = i["shape"])
			npy = root_dir + i["name"]
			np.save(npy, arr.astype(np.int8))
			print(npy + '.npy', arr.shape)
			shape = i["shape"]
		elif i["type"] == "Convolution":
			reg['CONV:WEIGHT:SHAPE:K'], reg['CONV:WEIGHT:SHAPE:C'], reg['CONV:WEIGHT:SHAPE:H'], reg['CONV:WEIGHT:SHAPE:W'] = i["num_output"]-1, shape[0]-1, i["kernel_size"]-1, i["kernel_size"]-1
			reg['CONV:STRIDE:X_STRIDE'] = i["stride"] - 1
			reg['CONV:STRIDE:Y_STRIDE'] = i["stride"] - 1
			reg['CONV:DILATION:X_DILATION'] = i["dilation"] - 1
			reg['CONV:DILATION:Y_DILATION'] = i["dilation"] - 1
			reg['CONV:PAD:PAD_LEFT'] = i["pad"]
			reg['CONV:PAD:PAD_RIGHT'] = i["pad"]
			reg['CONV:PAD:PAD_TOP'] = i["pad"]
			reg['CONV:PAD:PAD_BOTTOM'] = i["pad"]
			reg['CONV:PAD:PAD_VALUE'] = 0
			arr = np.random.randint(-128,128,size=(i["num_output"], shape[0], i["kernel_size"], i["kernel_size"])) 
			npy = root_dir + i["name"] + '-weight'
			np.save(npy, arr.astype(np.int8))
			print(npy + '.npy', arr.shape)


class test_inference(object):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		#搜索目录file_dir下后缀名为postfix的文件列表
		import os
		f_name = lambda file_dir, postfix : list(map((lambda x : x[1]), (lambda s : filter(lambda x : os.path.splitext(x[0])[1] == '.' + postfix, map(lambda x:(x, os.path.join(s[0],x)), s[2])))(tuple(os.walk(file_dir))[0])))
		graph_file = f_name(root_dir, 'gpaph')[0]
		self.graph = self.make_graph_by_prototxt(graph_file)
	def make_graph_by_prototxt(self, graph_file):
		"""
		从graph文件中导出网络结构
		"""
		with open(txt, 'r') as f:
			graph = []
			while True:
				s = f.readline()
				if s:
					pos = s.find(':')
					if pos < 0:
						continue
					key = s[:pos].strip()
					value = s[pos+1:].strip()
					if value[0] == '"':
						value = value[1:value[1:].find('"')+1]
					else:
						value = int(value)
					if key == 'type':
						graph.append({})
					graph[-1][key] = value
				else:
					break
			return graph
	def input(self, data, node):
		return np.load(self.root_dir+node["name"]+'.npy').astype(np.int32)
	def conv(self, data, node):
		weight = np.load(self.root_dir+node["name"]+'-weight'+'.npy').astype(np.int32)
		if node["pad"] != 0:
			X = node["pad"]*2 + data.shape[1]
			d = np.zeros((data.shape[0], X, X)).astype(np.float32)
			d[:, node["pad"]:node["pad"]+data.shape[1], node["pad"]:node["pad"]+data.shape[2]] = data
		else:
			d = data
		stride = node["stride"]
		kernel_size = node["kernel_size"]
		dilation = node["dilation"]
		conv_size = 1+dilation*(kernel_size-1)
		X = (d.shape[1]-conv_size)//stride+1
		if not ("group" in node):
			node["group"] = 1
		if len(weight.shape) == 4:
		 	weight = weight.reshape(1,*weight.shape)
		a = []
		c_in = d.shape[0]//node["group"]
		c_out = node["num_output"]//node["group"]
		for i in range(node["group"]):
			data_group = np.zeros((c_out, X, X)).astype(np.int32)
			for c,h,w in itertools.product(*map(lambda x:range(x),data_group.shape)):
				data_group[c,h,w] = np.vdot(
							d[c_in*i:c_in*(i+1), h*stride:h*stride+conv_size:dilation, w*stride:w*stride+conv_size:dilation],
							weight[i,c]
							)
			a = a + [data_group]
		data = np.concatenate(a, axis=0)
		return data
	def bias_add(self, data, node):
		bias = np.load(self.root_dir+node["name"]+'-bias'+'.npy').astype(np.int32)
		for i in range(data.shape[0]):
			data[i] = data[i] + bias[i]
		return data
	def BN(self, data, node):
		# y = (x+m)*n
		args = np.load(self.root_dir+node["name"]+'-BN'+'.npy').astype(np.int64)
		m = args[0]
		n = args[1]
		data = data.astype(np.int64)
		for i in range(data.shape[0]):
			data[i] = (data[i] + m[i]) * n[i]
		return data
	def relu(self, data, node):
		return np.where(data>0,data,0).astype(np.int32)
	def truncate(self, data, node):
		return rshift_round_numpy(data, node["rshift"]).astype(np.int32)
	def sdp_out_cvt(self, data, node):
		return rshift_round_numpy((data.astype(np.int64)-node["offset"])*node["mul"], node["rshift"]).astype(np.int8)
	def pool(self, data, node):
		stride = node["stride"]
		kernel_size = node["kernel_size"]
		X = (data.shape[1]-kernel_size)//stride+1
		d = np.zeros((data.shape[0], X, X)).astype(np.int8)
		if node["function"] == "max":
			f = np.max
		elif node["function"] == "min":
			f = np.min
		elif node["function"] == "avg":
			f = lambda a:int(np.round(np.mean(a)))
		for c,h,w in itertools.product(*map(lambda x:range(x),d.shape)):
			d[c,h,w] = f(data[c, h*stride:h*stride+kernel_size, w*stride:w*stride+kernel_size])
		return d
	def inference_graph(self, graph, start_node, data_in = None):
		op_table = {
			"Input" : self.input,
			"Convolution" : self.conv,
			"BiasAdd" : self.bias_add,
			"BN" : self.BN,
			"ReLU" : self.relu,
			"Truncate" : self.truncate,
			"Cvt" : sdp_out_cvt,
			"Pooling" : self.pool
		}
		data_out = reduce(lambda data, node : op_table[node["type"]](data, node), graph[start_node:], data_in)
		return data_out
	def inference(self, data_in = None):
		return self.inference_graph(self.graph, 0, data_in)

"""
以下两个函数是用来右移的，NVDLA的右移有用到round(四舍五入)，
也就是如果移除的最高位为1,那么结果得加上1
"""
def rshift_round(num, rshift):
	return rshift_round_numpy((lambda n:np.array([n]).astype(np.uint64))(num), rshift)[0]
		
def rshift_round_numpy(a, rshift):
	if rshift==0:
		return a.astype(np.int64)
	a = a.astype(np.uint64)
	ret = (a>>rshift) + ((a>>(rshift-1))&1)
	ret = np.where(a>>63, ret|(((1<<rshift)-1)<<(64-rshift)), ret)
	return ret.astype(np.int64)

def inference(root_dir):
	"""
	对简单网络做推理
	"""
	x = test_inference(root_dir)
	print('Inferencing...Please wait')
	r = x.inference()
	npy = root_dir + "inference-output-result-origin"
	np.save(npy, r['data_origin'].astype(np.int32))
	npy = root_dir + "inference-output-result"
	np.save(npy, r['data'].astype(np.int8))

	mk_list = lambda shape:[None]*shape[0] if len(shape)==1 else [mk_list(shape[1:])] if shape[0]==1 else mk_list((shape[0]-1,)+shape[1:]) + mk_list((1,)+shape[1:])
	lst = mk_list(r['data'].shape)
	for i,j,k in itertools.product(*map(lambda x:range(x),r['data'].shape)):
		lst[i][j][k] = "%#04x" % (r['data'].astype(np.uint8)[i,j,k],)
	print(lst, file=open(root_dir + 'result.txt', 'w'))
	print(npy + '.npy', r['data'].shape)

	global reg
	#reg["FEATURE:OUT:C"], reg["FEATURE:OUT:H"], reg["FEATURE:OUT:W"] = r['data'].shape
	reg["SDP:CVT:OFFSET"] = np.array([r['offset']]).astype(np.int32).astype(np.uint32)[0]
	reg["SDP:CVT:SCALE"] = np.array([r['mul']]).astype(np.int32).astype(np.uint32)[0]
	reg["SDP:CVT:SHIFT"] = r['rshift']
	reg['FEATURE:OUT:STRIDE:LINE_STRIDE'] = r['data'].shape[2] * 32
	reg['FEATURE:OUT:STRIDE:SURF_STRIDE'] = r['data'].shape[2] * r['data'].shape[1] * 32

def write_ini(reg, ini):
	with open(ini, 'w') as f:
		section = ""
		for i in sorted(reg.keys(), key=lambda k:(len(k.split(":")),k)):
			s = i[:i.find(":")]
			if s != section:
				section = s
				f.write('\n[%s]\n' % (section,))
			if type(reg[i]) == type('') or type(reg[i]) == type(True):
				f.write('%s = %s\n' % (i, str(reg[i])))
			else:
				f.write('%s = %#x\n' % (i, reg[i]))

def write_json(reg, json_file):
	with open(json_file, "w") as f:
		f.write('{\n')
		for i, k in enumerate(sorted(reg.keys(), key=lambda k:(len(k.split(":")),k))):
			#print(i, k, reg[k])
			f.write('\t"%s" : %d' % (k, np.array([reg[k]]).astype(np.int32)[0]) + (',\n' if i<len(reg)-1 else '\n'))
		f.write('}\n')
	
if __name__ == '__main__':
	root_dir = './'
	conf = { "C" : [1,4],
		"H" : [5,20],
		"K" : [1,17],
		"kernel_size" : [1,12,2],
		"stride" : [1,12],
		"pad" : [0,6],
		"dilation" : [1,6]
		}
	for arg in sys.argv[1:]:
		if arg == '-h':
			print("For example:\n\t" + sys.argv[0] + " C=1:4 H=5:20 K=1:17 kernel_size=1:12:2 stride=1:12 pad=0:6 dilation=1,2,3,4,5 ./")
			exit(0)
		i = arg.find('=')
		if i<0:
			root_dir = arg
			if root_dir[-1] != '/':
				root_dir += '/'
			continue
		v_str = arg[i+1:]
		if v_str.find(':') >= 0:
			v = list(map(int,v_str.split(':')))
			if len(v) == 2:
				v += [1 if v[1]>v[0] else -1]
		else:
			v = tuple(map(int,v_str.split(',')))
		conf[arg[:i]] = v

	reg = {}
	reg['CDMA:CVT_EN'] = 0

	# 随机生成简单网络，如果random_graph不带参数t，则只生成一个卷积
	graph = random_graph(conf)
	# 将prototxt和相应的npy生成在root_dir所存的目录
	random_result(graph, root_dir)
	# numpy下的推理，比较慢，但可以拿来对比
	inference(root_dir)
	write_ini(reg, root_dir+'reg.ini')
	write_json(reg, root_dir+'reg.json')
