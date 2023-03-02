# Find the home, since ~ does not work in general
# shell runs a shell command
# HOME	= $(shell echo ~)
# BOUT_TOP = $(HOME)/Documents/DTU/BOUT-dev
BOUT_TOP = /marconi/home/userexternal/kkvist00/BOUT-dev
#BOUT_TOP = /home/kristoffer/BOUT-dev

SOURCEC		=  hesel.cxx
# compile HeselParameters
DIRS	= HeselParameters Neutrals Parallel BoutFastOutput KineticNeutrals
# name of binary (executable)
TARGET = hesel

include $(BOUT_TOP)/make.config
