Our web server has been throwing server errors and we need to know which client is triggering the most of them.

The nginx access logs are in `/app/logs/`. Find the client IP address responsible for the most HTTP **5xx** responses across **all** of the logs there.

Write just that IP address (nothing else) to `/app/answer.txt`.
