# imports
import sys
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# for plotting
from tueplots import bundles
from tueplots.constants.color import rgb

plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.dpi": 200})
# plt.rcParams.update({'font.sans-serif': ['DejaVu Sans Mono'],'figure.dpi': 200})

# Get my_package directory path from Notebook
parent_dir = str(Path().resolve().parents[1])

# Add to sys.path
sys.path.insert(0, parent_dir)

cmap_wd = LinearSegmentedColormap.from_list("ow", ['w', rgb.tue_dark], N=1024)
cmap_wo = LinearSegmentedColormap.from_list("ow", ['w', rgb.tue_orange], N=1024)
cmap_wb = LinearSegmentedColormap.from_list("ow", ['w', rgb.tue_blue], N=1024)  
cmap_wg = LinearSegmentedColormap.from_list("ow", ['w', rgb.tue_green], N=1024)
cmap_wr = LinearSegmentedColormap.from_list("ow", ['w', rgb.tue_red], N=1024)