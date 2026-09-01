from .gelsight_mini.gsmini_cfg import GelSightMiniCfg
from .gelsight_mini.gsmini_taxim import GELSIGHT_MINI_TAXIM_CFG
from .gelsight_mini.gsmini_taxim_fots import GELSIGHT_MINI_TAXIM_FOTS_CFG

from .xensews.xensews_cfg import XenseWSCfg

from .gf225.gf225_cfg import GF225Cfg
from .gf225 import *

try:
    from .gelsight_mini.gsmini_taxim_fem import GELSIGHT_MINI_TAXIM_FEM_CFG
except ImportError:
    pass
