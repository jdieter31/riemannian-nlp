# This makefile is necessary to setup graph tool in your local virtual
# environment.

# Get the path to graph_tool
GRAPH_TOOL:= $(dir $(shell python3 -c "import graph_tool; print(graph_tool.__file__)" 2> /dev/null))
$(info using ${GRAPH_TOOL})
ifeq (${GRAPH_TOOL},)
$(error "Could not find an installation of `graph_tool`. \
Make sure you are running `make` outside of a virtual environment and \
that graph-tool is installed: \
https://git.skewed.de/count0/graph-tool/wikis/installation-instructions")
endif

# Try to get the python root from poetry 
# NOTE: this target will fail when run on spell, where 'poetry' is not
# installed.
PYTHONROOT := $(shell poetry run env | grep "VIRTUAL_ENV" | sed -r 's/VIRTUAL_ENV=(.*)/\1/' 2> /dev/null)
ifeq (${PYTHONROOT},)
$(info "Could not find a poetry virtual environment. \
Make sure you have poetry installed and have run `poetry install` at \
least once.")
endif

all: graph-tool-installed poetry-installed
	

poetry-installed:
	poetry install

graph-tool-installed: $(PYTHONROOT)/lib/python3.7/site-packages/graph_tool

# Create a symlink to graph-tool
$(PYTHONROOT)/lib/python3.7/site-packages/graph_tool:
	ln -s ${GRAPH_TOOL} $@

$(PYTHONROOT)/bin/python3:
	$(error "Please run 'poetry install' to setup your own virtual \
	environment")

poetry.lock: pyproject.toml
	poetry update

# Create a conda environment that matches our pip environment
.spell-conda.yml: #pyproject.toml poetry.lock | $(PYTHONROOT)/bin/python3
	bin/build_spell_env.sh $@


# A Spell-specific set of compilation steps
# Mostly, this builds the Cython files.
compile-in-spell:
	python3 build.py build_ext --inplace


.PHONY: graph-tool-installed poetry-installed compile-in-spell
