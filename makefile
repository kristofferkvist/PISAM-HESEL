#BOUT_TOP = /home/kristoffer/BOUT-dev

SOURCEC		=  hesel.cxx
# compile HeselParameters
DIRS	= HeselParameters Neutrals Parallel BoutFastOutput KineticNeutrals
# name of binary (executable)
TARGET = hesel
# Do not schange the following line. Simple define the appropriate BOUT_TOP in your .bashrc (or similar).
include $(BOUT_TOP)/make.config
#A target that makes all the tables and collects them in the dictionaries.
PISAM_setup: PISAM/make_tables_main.py PISAM/make_tables.py PISAM/table_dictionary.py
	python3 $<
	python3 PISAM/table_dictionary.py
#Clean all PISAM tables and data produces by PISAM-HESEL.
clean_all: clean_data clean_tables

clean_tables:
	rm PISAM/input_data/*

clean_data:
	rm data/*.log*
	rm data/*.nc
