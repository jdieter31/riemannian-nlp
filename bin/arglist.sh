#!/bin/bash
# arglist.sh
# A simple script to print arguments received. Used to debug other
# scripts.

E_BADARGS=85

if [[ $# == 0 ]]; then
  echo "Usage: `basename $0` <arguments>";
  exit $E_BADARGS;
fi

echo "Printing arguments to $0"
echo "====================================="
index=1          # Initialize count.
for arg in "$@"; do
  echo "Arg #$index = $arg";
  let "index+=1";
done;
echo "====================================="
