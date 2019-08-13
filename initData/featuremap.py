#!/usr/bin/python
#-*- coding: utf-8 -*-
import numpy as np
import sys

csv_type = 'one_line' #'two_fields'

def trans_npy_to_csv(npy, csv_file):
	shape = npy.shape
	x = shape[0]*shape[3]
	"""
	#先把形状拼接好，按32字节对齐
	if x%32 != 0:
		x = 32-x%32
		append_np = np.array([0]*(x*shape[1]*shape[2]), np.uint8).reshape(x//shape[3],*shape[1:])
		npy = np.concatenate((npy,append_np), axis=0)
		shape = npy.shape
	get_value = lambda a,s : a[s]
	"""
	#以下应该节省一点内存
	if x%32 != 0:
		shape = ((32+x-x%32)//shape[3],)+shape[1:]
	shape_origin = npy.shape
	get_value = lambda a,s : a[s] if s[0]<shape_origin[0] else 0

	def next_pos(pos):
		if pos[3]<shape[3]-1:
			return pos[:3]+(pos[3]+1,)
		elif pos[0]%(32//shape[3]) < 32/shape[3]-1:
			return (pos[0]+1,pos[1],pos[2],0)
		elif pos[2]<shape[2]-1:
			return (pos[0]-pos[0]%(32//shape[3]),pos[1],pos[2]+1,0)
		elif pos[1]<shape[1]-1:
			return (pos[0]-pos[0]%(32//shape[3]),pos[1]+1,0,0)
		elif pos[0]<shape[0]-1:
			return (pos[0]+1,0,0,0)
		return None
	with open(csv_file, 'w') as f:
		print(csv_file)
		if csv_type != 'one_line':
			f.write('Index,Value\n')
		index = 0
		pos = (0,0,0,0)
		while True:
			if csv_type != 'one_line':
				f.write(str(index)+','+str(get_value(npy,pos))+'\n')
			elif index == 0:
				f.write("%#04x"%(get_value(npy,pos),))
			else:
				f.write(",%#04x"%(get_value(npy,pos),))
				
			pos = next_pos(pos)
			if pos == None:
				if csv_type == 'one_line':
					f.write('\n')
				break
			index += 1

def main(npy_file, csv_file, dtype=None, transpose=None) :
	try:
		feature_map = np.load(npy_file)
		if dtype != None:
			feature_map = feature_map.astype(dtype)
		shape = feature_map.shape
		if len(shape) != 3:
			print(npy_file + 'is not a numpy file of feature maps')
			return
		if transpose != None:
			feature_map = feature_map.transpose(transpose)
		feature_map.dtype = np.uint8
		shape += (feature_map.shape[-1]//shape[-1],)
		feature_map = feature_map.reshape(*shape)
		return trans_npy_to_csv(feature_map, csv_file)
	except:
		print(npy_file + ' is not a numpy file')

# csv转npy
def main2(csv_file, npy_file, dtype, shape, transpose=None) :
	try:
		a = np.array(list(map(lambda c:int(c,16),open(csv_file,'r').readline().split(','))),dtype=np.uint8)
		x = a.size
		a.dtype = dtype
		C_pad = a.size//(shape[1]*shape[2])
		#每个像素的字节数
		x = x//a.size
		
		shape_origin,shape = shape,(C_pad,shape[1],shape[2])
		b = np.zeros(a.size).astype(dtype).reshape(*shape)

		def next_pos(pos):
			if pos[0]%(32//x)<32/x-1:
				return (pos[0]+1,pos[1],pos[2])
			elif pos[2]<shape[2]-1:
				return (pos[0]-pos[0]%(32//x),pos[1],pos[2]+1)
			elif pos[1]<shape[1]-1:
				return (pos[0]-pos[0]%(32//x),pos[1]+1,0)
			elif pos[0]<shape[0]-1:
				return (pos[0]+1,0,0)
			return None

		pos = (0,0,0)
		for i in range(a.size):
			b[pos] = a[i]
			pos = next_pos(pos)
		
		b = b[:shape_origin[0]]
		if transpose != None:
			b = b.transpose(transpose)
		print(npy_file)
		np.save(npy_file,b)
	except:
		print(csv_file + ' is not correct')


# 运行类似如下
# ./featuremap.py img.npy img.csv uint8
# ./featuremap.py img.npy img.csv uint8 2,0,1
# ./featuremap.py img.csv img.npy uint8 3,229,229
# ./featuremap.py img.csv img.npy float16 3,229,229
# ./featuremap.py img.csv img.npy float16 3,229,229 1,2,0
if __name__ == '__main__':
	try:
		#用csv_type变量来区分用什么样的csv输出
		if len(sys.argv)<3:
			raise
		if len(sys.argv[1])>4 and sys.argv[1][-4:]=='.csv':
			cmd = 'main2(\''+sys.argv[1]+'\', \''+sys.argv[2]+'\', dtype=np.'+sys.argv[3]+', shape=('+sys.argv[4]+')'
			if len(sys.argv)==5:
				pass
			elif len(sys.argv)==6:
				cmd += ', transpose=('+sys.argv[5]+')'
			else:
				raise
			cmd += ')'
		else:
			if len(sys.argv)==3:
				cmd = 'main(\''+sys.argv[1]+'\', \''+sys.argv[2]+'\')'
			elif len(sys.argv)==4 or len(sys.argv)==5:
				cmd = 'main(\''+sys.argv[1]+'\', \''+sys.argv[2]+'\''
				for i in sys.argv[3:]:
					if i.find(',') < 0:
						cmd += ', dtype=np.'+i
					else:
						cmd += ', transpose=('+i+')'
				cmd += ')'
			else:
				raise
		print(cmd)
		eval(cmd)
	except:
		print(sys.argv[0] + ' npy csv [type] [transpose]')
		print('or')
		print(sys.argv[0] + ' csv npy type shape [transpose]')
