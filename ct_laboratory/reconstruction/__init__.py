"""Reconstruction area - complete reconstruction algorithms built on tomography +
sparse_eigen_preconditioner (rolling-window multi-rotation PCG cascade for step-and-shoot
scans).  Generic in the scanner: geometry is a StepAndShootGeometry built by the caller."""
from .rolling_window import *
