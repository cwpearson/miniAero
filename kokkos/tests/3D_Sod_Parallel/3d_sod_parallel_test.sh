#!/bin/bash

set -x

echo "Running parallel 3D sod test using 4 processors."
#mpirun -np 4 $1 &> /dev/null
mpirun -np 4 $1 
diff=0
for i in `seq 0 3`;
do 
  ../tools/numeric_text_diff results.$i results.$i.gold > diff.$i.txt
  diff=$(($diff + $?))
done
if [ $diff -gt 0 ];
then
  ESTATUS=1
else
  ESTATUS=0
fi

rm -vf results.? gradients.? limiters.? setupmesh.?
exit $ESTATUS
