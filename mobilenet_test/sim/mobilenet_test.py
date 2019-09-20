#!/usr/bin/python3
#-*- coding: utf-8 -*-
"""
本程序用于对mobilenet的仿真
"""
import sys
import numpy as np
from functools import reduce
import itertools
import os
f_name = lambda file_dir, postfix : list(map((lambda x : x[1]), (lambda s : filter(lambda x : os.path.splitext(x[0])[1] == '.' + postfix, map(lambda x:(x, os.path.join(s[0],x)), s[2])))(tuple(os.walk(file_dir))[0])))

class test_inference(object):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		#搜索目录file_dir下后缀名为postfix的文件列表
		graph_file = f_name(root_dir, 'graph')[0]
		self.graph = self.make_graph_by_file(graph_file)
	def make_graph_by_file(self, graph_file):
		"""
		从graph文件中导入网络结构
		"""
		with open(graph_file, 'r') as f:
			graph = []
			for s in f.readlines():
				pos = s.find('#')
				if pos >= 0:
					s = s[:pos]
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
			# 检查是否有节点重名
			is_dup_name = (lambda arr:True in map(lambda a,b:a==b,arr[1:],arr[:-1]))\
					(sorted(map(lambda x:x['name'],graph)))
			if is_dup_name:
				print("ERROR!Duplication of names!")
				return None
			return graph
	def write_graph_file(self, graph_file):
		"""
		导出网络结构到graph文件
		"""
		shape = None
		with open(graph_file, 'w') as f:
			for node in self.graph:
				f.write('type : "%s"\nname : "%s"\n' % (node["type"], node["name"]))
				for key in node.keys():
					if key == 'type' or key == 'name' :
						continue
					if isinstance(node[key],str):
						f.write('%s : "%s"\n' % (key, str(node[key])))
					else:
						f.write('%s : %s\n' % (key, str(node[key])))
				shape = self.shape_inference(shape, node)
				f.write('# Output shape : ' + str(shape) + '\n\n')
	def shape_inference_input(self, shape, node):
		return (node['c'], node['h'], node['w'])
	def shape_inference_conv(self, shape, node):
		stride = node["stride"]
		kernel_size = node["kernel_size"]
		dilation = node["dilation"]
		conv_size = 1+dilation*(kernel_size-1)
		X = (node["pad"]*2+shape[1]-conv_size)//stride+1
		return (node["num_output"], X, X)
	def shape_inference_bias_add(self, shape, node):
		return shape
	def shape_inference_BN(self, shape, node):
		return shape
	def shape_inference_relu(self, shape, node):
		return shape
	def shape_inference_truncate(self, shape, node):
		return shape
	def shape_inference_sdp_out_cvt(self, shape, node):
		return shape
	def shape_inference_pool(self, shape, node):
		stride = node["stride"]
		kernel_size = node["kernel_size"]
		X = (shape[1]-kernel_size)//stride+1
		return (shape[0], X, X)
	def shape_inference(self, shape, node):
		f_table = {
			"Input" : self.shape_inference_input,
			"Convolution" : self.shape_inference_conv,
			"BiasAdd" : self.shape_inference_bias_add,
			"BN" : self.shape_inference_BN,
			"ReLU" : self.shape_inference_relu,
			"Truncate" : self.shape_inference_truncate,
			"Cvt" : self.shape_inference_sdp_out_cvt,
			"Pooling" : self.shape_inference_pool
		}
		return f_table[node["type"]](shape, node)
	def input(self, data, node):
		return np.load(self.root_dir+node["name"]+'.npy').astype(np.int32)
	def conv(self, data, node):
		weight = np.load(self.root_dir+node["name"]+'.npy').astype(np.int32)
		if node["pad"] != 0:
			X = node["pad"]*2 + data.shape[1]
			d = np.zeros((data.shape[0], X, X)).astype(np.int32)
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
		all_groups_result = []
		c_in = d.shape[0]//node["group"]
		c_out = node["num_output"]//node["group"]
		for i in range(node["group"]):
			data_group = np.zeros((c_out, X, X)).astype(np.int32)
			for c,h,w in itertools.product(*map(lambda x:range(x),data_group.shape)):
				data_group[c,h,w] = np.vdot(
							d[c_in*i:c_in*(i+1), h*stride:h*stride+conv_size:dilation, w*stride:w*stride+conv_size:dilation],
							weight[i,c]
							)
			all_groups_result.append(data_group)
		data = np.concatenate(all_groups_result, axis=0)
		return data
	def bias_add(self, data, node):
		bias = np.load(self.root_dir+node["name"]+'.npy').astype(np.int32)
		bias = bias<<node['lshift']
		for i in range(data.shape[0]):
			data[i] = data[i] + bias[i]
		return data
	def BN(self, data, node):
		# y = (x+b)*k
		args = np.load(self.root_dir+node["name"]+'.npy').astype(np.int64)
		b = args[0]<<node['lshift']
		k = args[1]
		data = data.astype(np.int64)
		for i in range(data.shape[0]):
			data[i] = (data[i] + b[i]) * k[i]
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
	def inference_graph(self, start_index, inf_graph_len, data_in = None):
		op_table = {
			"Input" : self.input,
			"Convolution" : self.conv,
			"BiasAdd" : self.bias_add,
			"BN" : self.BN,
			"ReLU" : self.relu,
			"Truncate" : self.truncate,
			"Cvt" : self.sdp_out_cvt,
			"Pooling" : self.pool
		}
		data_out = reduce(lambda data, node : op_table[node["type"]](data, node), self.graph[start_index:start_index+inf_graph_len], data_in)
		return data_out
	def inference(self, data_in = None):
		return self.inference_graph(0, len(self.graph), data_in)

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

def random_result(root_dir):
	"""
	根据神经网络结构随机产生featuremap/weight/bias，并按照结构里的名字为文件命名
	"""
	net = test_inference(root_dir)
	data = None
	for graph_index in range(len(net.graph)):
		node = net.graph[graph_index]
		print(node['type'], node['name'], '=> ',end = '')
		if node['type'] == 'Input':
			np.save(root_dir+node['name'], np.random.randint(-128, 128, size=(node['c'],node['h'],node['w'])).astype(np.int8))
		elif node['type'] == 'Convolution':
			if not ('group' in node):
				node['group'] = 1
			np.save(root_dir+node['name'], np.random.randint(-128, 128, size=(node['group'], node['num_output']//node['group'],data.shape[0]//node['group'],node['kernel_size'],node['kernel_size'])).astype(np.int8))
		elif node['type'] == 'BiasAdd':
			bias = np.random.randint(-(1<<15), 1<<15, size=(data.shape[0],)).astype(np.int16)
			np.save(root_dir+node['name'], bias)
			bias = bias.astype(np.int32)
			max_bias = np.max(bias)
			min_bias = np.min(bias)
			max_num = np.max(data)
			min_num = np.min(data)
			for i in range(32):
				if (max_bias<<i) > max_num or (min_bias<<i) < min_num :
					break
			if i != 0:
				i = i-1
			node['lshift'] = i
		elif node['type'] == 'BN':
			bn = np.random.randint(-(1<<15), 1<<15, size=(2, data.shape[0])).astype(np.int16)
			np.save(root_dir+node['name'], bn)
			b = bn[0].astype(np.int32)
			max_b = np.max(b)
			min_b = np.min(b)
			max_num = np.max(data)
			min_num = np.min(data)
			for i in range(32):
				if (max_b<<i) > max_num or (min_b<<i) < min_num :
					break
			if i != 0:
				i = i-1
			node['lshift'] = i
		elif node['type'] == 'ReLU':
			pass
		elif node['type'] == 'Truncate':
			max_num = np.max(data)
			min_num = np.min(data)
			for i in range(64):
				m = rshift_round(max_num, i)
				n = rshift_round(min_num, i)
				if m < (1<<31) and n >= -(1<<31):
					break
			node['rshift'] = i
		elif node['type'] == 'Cvt':
			max_num = np.max(data)
			min_num = np.min(data)
			to_int64 = lambda n:np.array([n]).astype(np.int64)[0]
			max_num = to_int64(max_num)
			min_num = to_int64(min_num)
			offset = (max_num + min_num)//2
			max_num = max_num - offset
			min_num = min_num - offset
			def get_v(max_num, min_num):
				for i in range(47):
					if max_num < (1<<i) and min_num >= (-1<<i):
						if i<8:
							break
						x = rshift_round_numpy(np.array([max_num, min_num]), i-7)
						x1 = np.max(x)
						x2 = np.min(x)
						if x1 > 127 or x2 < -128:
							i += 1
						break
				return (i,float(max_num)/(1<<i))
			r = max(map(lambda n:(n,)+get_v(max_num*n,min_num*n), range(1<<15)), key=lambda x:x[2])
			# 当所有的数据相等的时候，这里会有问题，此时offset后数据全为0
			# 取乘数为1,移位为0
			if max_num == min_num:
				r = (1, 7)
			mul = r[0]
			rshift = r[1]-7
			node['offset'] = offset
			node['mul'] = mul
			node['rshift'] = rshift
		elif node['type'] == 'Pooling':
			pass
		data = net.inference_graph(graph_index , 1, data)
		print(data.shape)
		np.save(root_dir+node['name']+'-result', data)

	np.save(root_dir + 'whole-net-inference-result', data)
	net.write_graph_file(f_name(root_dir, 'graph')[0])

if __name__ == '__main__':
	root_dir = './'
	if len(sys.argv)>1:
		root_dir = sys.argv[1]
	if root_dir[-1] != '/':
		root_dir += '/'
	if len(sys.argv)>2 and sys.argv[2][:3].lower()=='inf':
		np.save(root_dir + 'whole-net-inference-result', test_inference(root_dir).inference())
	else:
		random_result(root_dir)
