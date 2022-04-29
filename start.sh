#!/bin/bash
cd /python_backend
pm2-runtime "yarn dev"
wait -n
exit $?