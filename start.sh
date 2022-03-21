#!/bin/bash
cd /python_backend
pm2 start "yarn dev" --name "backend" &
curl parrot.live
wait -n
exit $?