#!/usr/bin/env bash

function join { local IFS="$1"; shift; echo "$*"; }

N=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)

for i in ${N[@]};
do
  value[0]=$i
  blocks=$(($i / 1024))
  if [ $blocks -eq 0 ]
  then
    threads=$(($i/2))
    blocks=1
  else
    threads=1024
  fi
	for k in `seq 1 10`;
	do
		value[k]=$($1 -n $i -b $blocks -t $threads)
  done
  for k in `seq 11 20`;
  do
    value[k]=$($1 -n $i -c)
  done
	echo $blocks,$threads,$(join , "${value[@]}")
done
