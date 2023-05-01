#BOUT_TOP = /home/kristoffer/BOUT-dev

SOURCEC		=  hesel.cxx
# compile HeselParameters
DIRS	= HeselParameters Neutrals Parallel BoutFastOutput KineticNeutrals
# name of binary (executable)
TARGET = hesel

include $(BOUT_TOP)/make.config

PISAM_setup: PISAM/make_tables_main.py PISAM/make_tables.py PISAM/table_dictionary.py
	python3 $<
	python3 PISAM/table_dictionary.py

clean_all: clean_data clean_tables

clean_tables:
	rm input_data/*

clean_data:
	rm data/*.log*
	rm data/*.nc
