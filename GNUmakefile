# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME := ./amrex

DEBUG        = FALSE
USE_MPI      = TRUE
USE_OMP      = FALSE
COMP         = gnu
DIM          = 2
PRECISION    = DOUBLE

BL_NO_FORT = TRUE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package
include $(AMREX_HOME)/Src/LinearSolvers/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
