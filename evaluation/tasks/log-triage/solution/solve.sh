#!/bin/bash
# Reference solution. Run by the `oracle` agent (harbor run -a oracle) to prove
# the task is solvable and the verifier is correct.
cd /app/logs
{ cat access.log access.log.1; zcat access.log.2.gz; } \
  | awk '$9 >= 500 && $9 < 600 { print $1 }' \
  | sort | uniq -c | sort -rn | head -1 | awk '{ print $2 }' > /app/answer.txt
