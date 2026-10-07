"""Scatter and off-focal models (see README.md in this folder for the equations and when to use which)."""
from .scatter_models import ZeroScatterModel, ConstantScatterModel
from .kernels import gaussian_matrix, blur_air_padded
from .off_focal import OffFocalRadiation, FocalSpotBlur
from .room_scatter import RoomScatter
from .object_scatter import CentroidKleinNishinaScatter, geometry_factors, klein_nishina, WATER_ELECTRONS_PER_MM3
from .binned_scatter import BinnedAdditiveScatter, module_bin_index, neighbour_pairs, bin_grid
from .scatter_spectrum import ScatterSpectrum
