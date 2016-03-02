#!/bin/bash
COUNTER=0
while [ $COUNTER -lt 100 ]; do
  ./sse-bench >> dct.txt
  let COUNTER=COUNTER+1 
done
