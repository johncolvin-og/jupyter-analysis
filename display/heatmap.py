import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def interpolate_color(c1, c2, pct):
    return [min_color[0] + pct * (max_color[0] - min_color[0]),
        min_color[1] + pct * (max_color[1] - min_color[1]),
        min_color[2] + pct * (max_color[2] - min_color[2])]


def gradient_color_list(col, color_tuples):
    max = col.max()
    min = col.min()
    min_color = (255, 0, 0)
    max_color = (0, 0, 255)

    def interp_color(value):
        pct = (max - value) / (max - min)
        return [min_color[0] + pct * (max_color[0] - min_color[0]),
                min_color[1] + pct * (max_color[1] - min_color[1]),
                min_color[2] + pct * (max_color[2] - min_color[2])]

    def interp_color_int(value):
        if np.isnan(value) or str(value) == 'nan' or str(value) == 'NaN':
            return[0, 0, 0]
        rgb = interp_color(value)
        return [int(round(v)) for v in rgb]
    colored_cells = []
    for v in col:
        rgb = interp_color_int(v)
        colored_cells.append('background-color: #{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2]))
    return colored_cells


def empty_color_dict():
    return {
        'red': [],
        'green': [],
        'blue': []
    }

def linear_segmented_column_heatmap(anchors):
    cdict = empty_color_dict()
    for a in anchors:
        cdict['red'].append((a[0], a[1], a[1]))
        cdict['green'].append((a[0], a[2], a[2]))
        cdict['blue'].append((a[0], a[3], a[3]))
    return cdict

def gradient_color_dict(color_tuples):
    assert len(color_tuples) > 1
    cdict = empty_color_dict()
    step_size = 1 / (len(color_tuples) - 1)
    i = 0
    for c in color_tuples:
        cdict['red'].append((step_size * i, c[0], c[0]))
        cdict['green'].append((step_size * i, c[1], c[1]))
        cdict['blue'].append((step_size * i, c[2], c[2]))
        i += 1
    return cdict
