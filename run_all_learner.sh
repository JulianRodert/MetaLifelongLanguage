#!/bin/bash
# pass python train script as argument 1 to this script, for example
# run_all_learners.sh train_rel.py

for learner in sequential multi_task agem replay maml oml anml ; do
  command="python3 $1 --learner=$learner" $2
  echo running "${command}"
  ${command}
done;


