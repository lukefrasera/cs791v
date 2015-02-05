#!/usr/bin/env bash

function join { local IFS="$1"; shift; echo "$*"; }

for i in `seq 10`;
do
	for j in `seq 1 10`;
	do
		for k in `seq 1 5`;
		do
			value[k]=$(./add-experimet -s -n 100000000 -b $(($i * 100)) -t $(($j * 100)))
		done
		echo $(($i * 100)),$(($j * 100)),$(join , "${value[@]}")
	done
done
