"""Shared publication style (IEEE / MATLAB-lines palette) for all paper figures.
Import and call apply(). Exports method/state color maps. pdf.fonttype=42 keeps text
editable in Inkscape/Illustrator; savefig as PDF for vector quality in LaTeX."""
import matplotlib as mpl

# MATLAB R2014b+ "lines" palette — the de-facto IEEE look, colorblind-friendly.
LINES = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
METHOD = {'deep': '#0072BD', 'hybrid': '#D95319', 'gated': '#77AC30',
          'HSMM': '#7F7F7F', 'oracle': '#7E2F8E'}
# states: heart sounds (S1/S2) in blues, intervals (systole/diastole) in warms — semantic
STATE = {0: '#FFFFFF', 1: '#1F4E79', 2: '#9DC3E6', 3: '#C55A11', 4: '#F4B183'}
STATE_NAMES = {1: 'S1', 2: 'systole', 3: 'S2', 4: 'diastole'}


def apply():
    mpl.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.family': 'sans-serif', 'font.size': 8,
        'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.fontsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.linewidth': 0.8, 'lines.linewidth': 1.3,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
        'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,   # editable vector text
        'axes.prop_cycle': mpl.cycler(color=LINES),
    })


apply()
