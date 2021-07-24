#!/usr/bin/bash
thread_num=8
for((i=0;i<${thread_num};i++));do
{
	python3 make_entity_mask.py ${thread_num} ${i} train2017
}&
done
wait