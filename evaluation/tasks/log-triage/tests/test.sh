#!/bin/bash
# Runs in a separate verifier container built from tests/Dockerfile, *after*
# the agent finishes; /app/answer.txt is copied in from the agent container.
# Whatever lands in /logs/verifier/reward.txt (or reward.json) is the reward.
EXPECTED="10.0.0.99"
ACTUAL="$(tr -d '[:space:]' < /app/answer.txt 2>/dev/null)"

echo "expected: $EXPECTED"
echo "actual:   ${ACTUAL:-<missing /app/answer.txt>}"

if [ "$ACTUAL" = "$EXPECTED" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
