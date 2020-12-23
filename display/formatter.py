import pandas as pd
from pandas.io.formats.style import Styler

# bc i hate looking at -0.0
def no_neg_zero(val):
    if val == 0:
        if isinstance(val, float):
            return 0.0
        else: return 0
    return val

def print_pretty(df: pd.DataFrame) -> Styler:
    col_styles = {}
    for c in df.columns:
       if 'price' in c or 'bid_p_' in c or 'ask_p_' in c or '_var_' in c or '_avg_' in c or '_std_' in c:
          col_styles[c] = "{:.2f}"
       elif 'time' in c:
          col_styles[c] = lambda t: f'{t.total_seconds():.2f}'
    return df.style.set_properties(**{'text-align': 'right'}).format(col_styles).set_table_styles([
       dict(selector='td', props=[('text-align', 'right')])
       #dict(selector='tr', props=[('background', '#88ff0000')])
    ])