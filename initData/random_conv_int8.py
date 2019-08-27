#!/usr/bin/python3
#-*- coding: utf-8 -*-
"""
本程序用于蔡晋、王梁洁、杨俊对单个卷积的探索
产生一个卷积、SDP的C1
"""
import sys
import numpy as np
from functools import reduce
import itertools

def _random_graph(t):
	"""
	测试过程中，一般以下超参是被指定的，而不是随机
	所以具体测试时，手动修改指定超参
	"""
	ret = []
	#input feature map 
	C = np.random.randint(1,4)
	H = np.random.randint(5,20)
	shape = (C,H,H)
	ret.append({"type":"Input",
		"name":"data",
		"shape":shape})

	#convolution
	choice_list = list(filter(lambda x: True if (x[2]==0 or x[2]==x[0]//2) and x[0]>=x[1] and H+x[2]*2-x[0]>=0 and (H+x[2]*2-x[0])%x[1]==0 else False,
		itertools.product(range(1,12,2), range(1, 12), range(0,6))))
	if len(choice_list)==0:
		return None
	#1~16
	num_output = np.random.randint(1, 17)
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

	return ret

def _check(graph):
	for i in graph:
		if i["type"] == "Input":
			shape = i["shape"]
		elif i["type"] == "Convolution":
			bank_size = 32 * 1024
			weight_size = i["num_output"] * shape[0] * i["kernel_size"] * i["kernel_size"] * 2
			weight_size = (weight_size + 127) // 128 * 128
			weight_banks = (weight_size + (bank_size-1)) // bank_size
			feature_banks = 16 - weight_banks
			if weight_banks>15 or feature_banks*1024//i["kernel_size"]<1:
				return False
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
		if graph != None and _check(graph) == True:
		#if graph != None:
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

	global reg
	# to do, numpy
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
			reg['CONV:DILATION:X_DILATION'] = 1 - 1
			reg['CONV:DILATION:Y_DILATION'] = 1 - 1
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
		X = (d.shape[1]-kernel_size)//stride+1
		data = np.zeros((node["num_output"], X, X)).astype(np.int32)
		for c,h,w in itertools.product(*map(lambda x:range(x),data.shape)):
			data[c,h,w] = np.sum(np.multiply(
						d[:, h*stride:h*stride+kernel_size, w*stride:w*stride+kernel_size],
						weight[c]
						))
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
		ret = reduce(lambda data, node : op_table[node["type"]](data, node), self.graph, None)
		"""
		后面是经过SDP的C1，进行精度截断（实际上是一个近似的线性转换），需要三个步骤:平移、乘积、移位
		注意:尽可能的保留信息,也就是希望数据充满-128~127
		"""
		max_num = np.max(ret)
		min_num = np.min(ret)
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
		mul = r[0]
		rshift = r[1]-7
		return {"offset":offset, "mul":mul, "rshift":rshift, "data":rshift_round_numpy((ret.astype(np.int64)-offset)*mul, rshift), "data_origin":ret}

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

	#get_v = lambda a,i:a if i==() else get_v(a[i[0]],i[1:])
	mk_list = lambda shape:[None]*shape[0] if len(shape)==1 else [mk_list(shape[1:])] if shape[0]==1 else mk_list((shape[0]-1,)+shape[1:]) + mk_list((1,)+shape[1:])
	#lst = reduce(lambda r,n:[r]*n,reversed(r['data'].shape),None)
	lst = mk_list(r['data'].shape)
	for i,j,k in itertools.product(*map(lambda x:range(x),r['data'].shape)):
		#print(i,j,k,r['data'].astype(np.uint8)[i,j,k])
		lst[i][j][k] = "%#04x" % (r['data'].astype(np.uint8)[i,j,k],)
	print(lst, file=open(root_dir + 'result.txt', 'w'))
	#a = np.max(r['data'])
	#b = np.min(r['data'])
	#if max(a,b)>127 or min(a,b)<-128:
	#	print('ERROR!!!!!!!!!!!!!!')
	print(npy + '.npy', r['data'].shape)
	global reg
	#reg["FEATURE:OUT:C"], reg["FEATURE:OUT:H"], reg["FEATURE:OUT:W"] = r['data'].shape
	reg["SDP:CVT:OFFSET"] = np.array([r['offset']]).astype(np.int32).astype(np.uint32)[0]
	reg["SDP:CVT:SCALE"] = np.array([r['mul']]).astype(np.int32).astype(np.uint32)[0]
	reg["SDP:CVT:SHIFT"] = r['rshift']

def convert_weight_numpy(npy, dat):
	weight = np.load(npy).astype(np.uint8)
	K,C,H,W = weight.shape
	kernel_group = 32
	chanel_group = 64
	def next_pos(pos):
		k,c,h,w = pos
		while True:
			yield (k,c,h,w)
			if (c+1)%chanel_group!=0 and c<C-1:
				k,c,h,w = k,c+1,h,w
			elif (k+1)%kernel_group!=0 and k<K-1:
				k,c,h,w = k+1,c-c%chanel_group,h,w
			elif w<W-1:
				k,c,h,w = k-k%kernel_group,c-c%chanel_group,h,w+1
			elif h<H-1:
				k,c,h,w = k-k%kernel_group,c-c%chanel_group,h+1,0
			elif c<C-1:
				k,c,h,w = k-k%kernel_group,c+1,0,0
			elif k<K-1:
				k,c,h,w = k+1,0,0,0
			else:
				break
	with open(dat, 'w') as f:
		f.write('{\n')
		for i, pos in enumerate(next_pos((0,0,0,0))):
			if i%16==0:
				f.write('{offset:%#x, size:16, payload:' % (i,))
			else:
				f.write(' ')
			f.write('%#04x' % (weight[pos],))
			if i%16==15:
				f.write('},\n')
		f.write(' 0x00'*(15-i%16) + '},\n')
		i=(i+15)//16*16
		while i%128 != 0:
			f.write('{offset:%#x, size:16, payload:'%(i,) + '0x00 '*15 + '0x00},\n')
			i += 16
		f.write('}\n')

def convert_feature_numpy(npy, dat, is_wr_reg = True):
	feature = np.load(npy).astype(np.uint8)
	C,H,W = feature.shape
	def next_pos(pos):
		c,h,w = pos
		t = 1
		while True:
			yield (t, (c,h,w))
			if (c+1)%32!=0:
				c,h,w = c+1,h,w
				t = 1 if c<C else 0
			elif w<W-1:
				c,h,w = c-c%32,h,w+1
				t = 1
			elif h<H-1:
				c,h,w = c-c%32,h+1,0
				t = 2
			elif c<C-1:
				c,h,w = c+1,0,0
				t = 3
			else:
				break
	with open(dat, 'w') as f:
		f.write('{\n')
		addr = 0
		line_stride = W*32*2
		surface_stride = W*H*32*2
		if is_wr_reg:
			global reg
			reg['FEATURE:STRIDE:LINE_STRIDE'] = line_stride
			reg['FEATURE:STRIDE:SURF_STRIDE'] = surface_stride
		for t, pos in next_pos((0,0,0)):
			if addr%16==0:
				if t==2:
					addr = (addr+line_stride-1)//line_stride*line_stride
				elif t==3:
					addr = (addr+surface_stride-1)//surface_stride*surface_stride
				f.write('{offset:%#x, size:16, payload:' % (addr,))
			else:
				f.write(' ')
			f.write('%#04x' % (0 if t==0 else feature[pos],))
			if addr%16==15:
				f.write('},\n')
			addr += 1
		f.write('}\n')

def convert_numpy(root_dir):
	convert_weight_numpy(root_dir+'conv1-weight.npy', root_dir+'weight.dat')
	convert_feature_numpy(root_dir+'data.npy', root_dir+'input.dat')
	
def write_ini(reg, ini):
	with open(ini, 'w') as f:
		section = ""
		for i in sorted(reg.keys()):
			s = i[:i.find(":")]
			if s != section:
				section = s
				f.write('\n[%s]\n' % (section,))
			if type(reg[i]) == type('') or type(reg[i]) == type(True):
				f.write('%s = %s\n' % (i, str(reg[i])))
			else:
				f.write('%s = %#x\n' % (i, reg[i]))

#def write_json(reg, json_file):
#	print(reg, file=open(json_file, "w"))
	
if __name__ == '__main__':
	root_dir = './'
	if len(sys.argv) > 1:
		root_dir = sys.argv[1]

	if len(root_dir) == 0:
		root_dir = './'
	elif root_dir[-1] != '/':
		root_dir += '/'

	reg = {}
	reg['CDMA:CVT_EN'] = 0

	# 随机生成简单网络，如果random_graph不带参数，则只生成一个卷积
	graph = random_graph()
	# 将prototxt和相应的npy生成在root_dir所存的目录
	random_result(graph, root_dir)
	# numpy下的推理，比较慢，但可以拿来对比
	inference(root_dir)
	convert_numpy(root_dir)
	write_ini(reg, root_dir+'reg.ini')
	#write_json(reg, root_dir+'reg.json')
