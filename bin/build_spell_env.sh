#!/bin/bash
# Build an Conda environment for spell from a virtual environment.
#
# This script creates a Conda environment with the right version of
# Python and the required packages to be able to run our own versions of
# these files.
#
# We are using Conda here because Spell has native support for Conda; in
# particular using Conda allows us to use a specific Python version
# (3.7.3).

set -o errexit

if [[ $# != 1 ]]; then
  echo "Usage: $0 <output>"
  exit 1
fi
output=$1;

# Enter the pynlp directory. This makes this script safe to call from anywhere.
# Directory magic from here: https://stackoverflow.com/questions/59895/get-the-source-directory-of-a-bash-script-from-within-the-script-itself
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "${DIR}/.."

PYTHONROOT := $(shell poetry run env | grep "VIRTUAL_ENV" | sed -r 's/VIRTUAL_ENV=(.*)/\1/' 2> /dev/null)
if [ ! -d $PYTHONROOT ]; then
  echo 'Missing virtual environment in $PYTHONROOT.'
  exit 1;
fi;

# A list of packages to filter from 'pip freeze'. These packages are
# require a link to be # installed (embedding_evaluation) that will be
# explicitly provided in EXTRAS.
BLACKLIST="embedding_evaluation riemannian-nlp torch faiss";

# Additional pacakges to add to the Conda environment we're building
# We have to specify the exact revision and egg here.
EXTRAS=('git+https://github.com/arunchaganty/embedding_evaluation.git#egg=embedding_evaluation');

# 1. Create conda file header
cat > $output <<EOF
name: riemann
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.7.3
  - graph-tool
  - cudatoolkit=10.0
  - faiss-gpu
  - pytorch=1.3
  - pip
  - pip:
    - spell
EOF
# 2. Add all the dependencies from pip (except those in the blacklist)
$PYTHONROOT/bin/pip freeze | grep -Ev $(echo $BLACKLIST | sed 's/ /|/g') | sed 's/^/    - /' >> $output;

# 3. Add any additional dependencies
for dep in ${EXTRAS[@]}; do
  echo "    - $dep" >> $output;
done;
