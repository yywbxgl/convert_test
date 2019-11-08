#!/bin/sh
mem0=0xc0000000
size=0x01000000
batch=1024
[ $# -ge 1 ] && mem0=`printf "%#x" $1`
[ $# -ge 2 ] && size=`printf "%#x" $2`
[ $# -ge 3 ] && batch=$3
mem0_dec=`printf "%d" $mem0`
size_dec=`printf "%d" $size`
mem1_dec=`expr $mem0_dec + $size_dec`
mem1=`printf "%#x" $mem1_dec`
size=$size_dec
unset mem0_dec mem1_dec size_dec 
echo increase
./test-dev-mem $mem0 $size increase
./test-dev-mem $mem1 $size increase
echo done
i=0
while :; do 
	echo i=$i
	let i=i+1
	j=0
	arg="$size write"
	while [ $j -lt $batch ]; do
		let addr=\(RANDOM\*32768+RANDOM\)%size
		let value=RANDOM%256
		arg=$arg" $addr,$value"
		let j=j+1
	done
	./test-dev-mem $mem0 $arg
	./test-dev-mem $mem1 $arg
	j=0
	arg=""
	while [ $j -lt $batch ]; do
		let addr=\(RANDOM\*32768+RANDOM\)%size
		arg=$arg" $addr"
		let j=j+1
	done
	x=`./test-dev-mem $mem0 $size read $arg`
	y=`./test-dev-mem $mem1 $size read $arg`
	if [ "$x" != "$y" ]; then
		echo ERROR
		{ echo $arg; echo $x; echo $y; }
		exit 1
	fi
done
