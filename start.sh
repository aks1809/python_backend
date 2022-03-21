#!/bin/bash
cd /python_backend
pm2 start "yarn dev" --name "backend" &
bash &
wait -n
exit $?