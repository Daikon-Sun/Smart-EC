#!/bin/bash

MEM=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)

if [ "$MEM" -lt "67108864" ]; then
  echo "Memory not enough!"
  echo "The program needs 64GB RAM!"
  exit
fi

if [ "$1" == "17" ]; then
  echo "The program cannot execute case17"
  exit
fi

python3 main.py "testcases/case$1.json" "$2"
