#!/usr/bin/env bash

function join { local IFS="$1"; shift; echo "$*"; }

N=(2048 4096 8192 262144 524288 1048576 10000000 100000000)
B=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768)
T=(1 2 4 8 16 32 64 128 256 512 1024)
V=(1 2)
Z=(1 4 32 128 1024 2048 4096 8192 10000)
echo N, Version, Threshold, Blocks, Threads, Samples
for i in ${N[@]}; # For Each N size
do
  for j in ${V[@]}; # For each Version
  do
    for k in ${Z[@]}; # for each threshold
    do
      for q in ${B[@]}; # Blocks
      do
        for r in ${T[@]}; # Threads
        do
          for s in `seq 1 5`;
          do
            value[0]=$i
            value[1]=$j
            value[2]=$k
            value[3]=$q
            value[4]=$r
            value[s+4]=$($1 -n $i -v $j -d $k -b $q -t $r)
            # echo $($1 -n $i -v $j -d $k -b $q -t $r)
          done
          echo $(join , "${value[@]}")
        done
      done
    done
  done
done