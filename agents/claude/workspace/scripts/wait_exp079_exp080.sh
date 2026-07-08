#!/bin/bash
# Wait for exp_079 full CV (Q4 scattering confirmation, pid 257937) and
# exp_080 ExtraTrees probe (pid 262459) to finish, then print completion.
while kill -0 257937 2>/dev/null; do sleep 20; done
echo "exp_079 (full_cv_exp079_q4.py) finished"
while kill -0 262459 2>/dev/null; do sleep 5; done
echo "exp_080 (probe_exp080_extratrees.py) finished"
echo "BOTH_DONE"
