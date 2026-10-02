# Separate verifier image (build context: tests/). The agent's /app/stats.py is
# uploaded as an artifact before test.sh runs.
FROM python:3.12-slim@sha256:09f7da3bc104798d0afb40bc08d23ab2da20a76130cec1f2ef170848f5d85217
COPY test.sh test_hidden.py /tests/
