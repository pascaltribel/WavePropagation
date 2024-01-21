from matplotlib.colors import LinearSegmentedColormap

def get_seismic_cmap():
    colors = [(1, 0, 0.5), (0, 0, 0), (0, 1, 0.5)]
    return LinearSegmentedColormap.from_list('seism', colors, N=250)