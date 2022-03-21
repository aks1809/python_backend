#!/bin/bash
yarn dev &
wait -n
exit $?