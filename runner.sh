#!/usr/bin/env bash

PY='/Users/redwan/PycharmProjects/GauessNewtonMethod/venv/bin/python'
eval()
{
  for i in 1 2 3
  do
    $PY GaussNewton.py --file test/$i.txt --noise 0.1 --num-samples $1
  done
}

for j in 90 100
do
  echo ```eval $j >> results/$j.txt```
done
