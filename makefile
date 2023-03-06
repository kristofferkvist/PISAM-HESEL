#BOUT_TOP = /home/kristoffer/BOUT-dev

SOURCEC		=  hesel.cxx
# compile HeselParameters
DIRS	= HeselParameters Neutrals Parallel BoutFastOutput KineticNeutrals
# name of binary (executable)
TARGET = hesel

include $(BOUT_TOP)/make.config

PISAM_setup: make_tables_main.py make_tables.py table_dictionary.py
	python3 $<
	python3 table_dictionary.py

clean_tables:
	rm input_data/*
