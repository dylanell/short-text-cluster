#!/bin/bash
# Basic while loop
counter=1
while [ $counter -le $2 ]
do
	$1
	((counter++))
done
