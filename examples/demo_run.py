# Import
from pathlib import Path
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

repo_root = Path.cwd().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from hydro_tipping_swi.plotting import (
    load_plot_bundle_npz,
    plot_model_sal_and_change_tidy,
    plot_extreme_salinization_two_panel,
    plot_fvert_three_panel_figure,
    plot_inundation_change_lollipop,
)



# Figure 1 is a conceptual figure, so no code is needed for that.

# Figure 2
mr2, d0, d1 = load_plot_bundle_npz( "../data/repro_bundle_salinity_maps.npz")
fig, axes, lims = plot_model_sal_and_change_tidy(
    mr2, d0, d1,
    sal_cmap=sns.cubehelix_palette(as_cmap=True),
    delta_cmap="coolwarm",
    sal_vmin=0, sal_vmax=35,
    delta_vlim=None,
    change_mode="absolute",
    panel_labels=("A", "B", "C", "D")
)
plt.show() 


# Figure 3
fig, axes, out_png = plot_extreme_salinization_two_panel(
    df=pd.read_csv("../data/df_models_with_salt_budget.csv"),
    dir_out="./dir_out",
    panel_a_offset="0m0",
    save=True,
    show=True,
)

# Figure 4

fig, axes, out_path = plot_fvert_three_panel_figure(
    df=pd.read_csv("../data/df_models_with_salt_budget.csv"),
    dir_out="./dir_out",
    save=True,
    show=True,
)

# Figure 5

plot_inundation_change_lollipop(
    swi_csv="../data/summary_with_areas_km2.csv",
    output_dir = "./dir_out",
    save_figure=True,
    show=True,
)

