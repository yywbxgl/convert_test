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
	W = rand("W")
	shape = (C,H,W)
	ret.append({"type":"Input",
		"name":"data",
		"shape":shape})

	#convolution
	choice_list = list(filter(lambda x:
		(lambda conv_H_size, conv_W_size, stride_H, stride_W, pad_H_max, pad_W_max, H_add_pad, W_add_pad : pad_H_max<conv_H_size and pad_W_max<conv_W_size and conv_H_size>=stride_H and conv_W_size>=stride_W and H_add_pad-conv_H_size>=0 and (H_add_pad-conv_H_size)%stride_H==0 and W_add_pad-conv_W_size>=0 and (W_add_pad-conv_W_size)%stride_W==0)
		((x[0]-1)*x[2]+1, (x[1]-1)*x[3]+1, x[4], x[5], max(x[6:8]), max(x[8:10]), H+sum(x[6:8]), W+sum(x[8:10])),
		itertools.product(arr("CONV_H"), arr("CONV_W"), arr("dilation_H"), arr("dilation_W"), arr("stride_H"), arr("stride_W"), arr("pad_top"), arr("pad_bottom"), arr("pad_left"), arr("pad_right"))))
	if len(choice_list)==0:
		return None
	num_output = rand("K")
	CONV_H, CONV_W, dilation_H, dilation_W, stride_H, stride_W, pad_top, pad_bottom, pad_left, pad_right = random_select(choice_list)
	ret.append({"type":"Convolution",
		"name":"conv1",
		"num_output":num_output,
		"CONV_H":CONV_H,
		"CONV_W":CONV_W,
		"dilation_H":dilation_H,
		"dilation_W":dilation_W,
		"stride_H":stride_H,
		"stride_W":stride_W,
		"pad_top":pad_top,
		"pad_bottom":pad_bottom,
		"pad_left":pad_left,
		"pad_right":pad_right})
	if t=='Convolution':
		return ret
	H_out = (shape[1]+pad_top+pad_bottom-((CONV_H-1)*dilation_H+1))//stride_H+1
	W_out = (shape[2]+pad_left+pad_right-((CONV_W-1)*dilation_W+1))//stride_W+1
	shape = (num_output, H_out, W_out)

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
			weight_size = i["num_output"] * shape[0] * i["CONV_H"] * i["CONV_W"]
			weight_size = (weight_size + 511) // 512 * 512
			weight_banks = (weight_size + (bank_size-1)) // bank_size
			#if weight_banks + feature_banks > 16:
			#	return False
			X = (shape[1]+i["pad"]*2-i["kernel_size"])//i["stride"]+1
			shape = (i["num_output"], X, X)
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
	根据神经网络结构随机产生featuremap/weight/bias，并按照结构里的名字为文件命名
	"""
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
			d = np.zeros((data.shape[0], X, X)).astype(np.int32)
			d[:, node["pad"]:node["pad"]+data.shape[1], node["pad"]:node["pad"]+data.shape[2]] = data
		else:
			d = data
		stride = node["stride"]
		kernel_size = node["kernel_size"]
		dilation = node["dilation"]
		conv_size = 1+dilation*(kernel_size-1)
		X = (d.shape[1]-conv_size)//stride+1
		data = np.zeros((node["num_output"], X, X)).astype(np.int32)
		for c,h,w in itertools.product(*map(lambda x:range(x),data.shape)):
			data[c,h,w] = np.vdot(
						d[:, h*stride:h*stride+conv_size:dilation, w*stride:w*stride+conv_size:dilation],
						weight[c]
						)
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
		# 当所有的数据相等的时候，这里会有问题，此时offset后数据全为0
		# 取乘数为1,移位为0
		if max_num == min_num:
			r = (1, 7)
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
		line_stride = W*32
		surface_stride = W*H*32
		if is_wr_reg:
			global reg
			reg['FEATURE:IN:STRIDE:LINE_STRIDE'] = line_stride
			reg['FEATURE:IN:STRIDE:SURF_STRIDE'] = surface_stride
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
		"W" : [5,20],
		"K" : [1,17],
		"CONV_H" : [1,12,2],
		"CONV_W" : [1,12,2],
		"stride_H" : [1,12],
		"stride_W" : [1,12],
		"pad_top" : [0,6],
		"pad_bottom" : [0,6],
		"pad_left" : [0,6],
		"pad_right" : [0,6],
		"dilation_H" : [1,6],
		"dilation_W" : [1,6]
		}
	for arg in sys.argv[1:]:
		if arg == '-h':
			print("For example:\n\t" + sys.argv[0] + " C=1:4 H=5:20 K=1:17 CONV_H=1:12:2 CONV_W=1:12:2 stride=1:12 pad_left=0:6 pad_right=0:6 pad_top=0:6 pad_bottom=0:6 dilation_H=1,2,3,4,5 dialtion_W=1:6 ./")
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
	convert_numpy(root_dir)
	write_ini(reg, root_dir+'reg.ini')
	write_json(reg, root_dir+'reg.json')
