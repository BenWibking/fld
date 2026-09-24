# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME := ./amrex

DEBUG        = FALSE
USE_MPI      = TRUE
USE_OMP      = FALSE
COMP         = gnu
DIM          = 3
PRECISION    = DOUBLE

BL_NO_FORT = TRUE

ifneq ($(DIM),3)
$(error The FLD cloud and ICASE benchmarks require DIM=3)
endif

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package
include $(AMREX_HOME)/Src/LinearSolvers/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
