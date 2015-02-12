#!/usr/bin/env bash

function join { local IFS="$1"; shift; echo "$*"; }


for i in `seq 1 16`;
do
  for j in `seq 1 10`;
  do
    value[0]=$((2000/$i*2))
    value[1]=$(($i*2000*2))
    value[j+1]=$($1 -b $(($i*2)))
  done
  echo $(join , "${value[@]}")
done
