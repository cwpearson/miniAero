#!/bin/bash

set -x

echo "Running parallel flat plate test using 8 processors."
mpirun -np 8 --oversubscribe $1 &> /dev/null
diff=0
for i in `seq 0 7`;
do 
  ../tools/numeric_text_diff --relative-tolerance=1e-2 --floor=1e-3 results.$i results.$i.gold > diff.$i.txt
  diff=$(($diff + $?))
done
if [ $diff -gt 0 ];
then
  ESTATUS=1
else
  ESTATUS=0
fi

rm -vf  results.? gradients.? limiters.? setupmesh.?
exit $ESTATUS
