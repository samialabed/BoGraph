import logging

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

font_path = fm.findfont('Helvetica', fallback_to_default=True, rebuild_if_missing=True)
sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False})
sns.set_context("paper")
plt.rc("text", usetex=False)
plt.rc("xtick", labelsize="small")
plt.rc("ytick", labelsize="small")
plt.rc("axes", labelsize="medium")
plt.rc("pdf", use14corefonts=True)
plt.rcParams.update({
    "font.family": fm.FontProperties(fname=font_path).get_family(),
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",
    "axes.edgecolor": "k",
    "axes.linewidth": 1.5,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": False,
    "ytick.right": False,
    "grid.color": "lightgray",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "figure.figsize": (3.5, 2.5),
    "figure.dpi": 300,
})

logging.info("Loaded default font types.")