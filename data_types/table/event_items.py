import pandas as pd

def cast_column_types(table: pd.DataFrame):
    table['side'] = table['side'].astype('category')
    table['symbol'] = table['symbol'].astype('category')
    table['status'] = table['status'].astype('category')
    table['type'] = table['type'].astype('category')
    table['change'] = table['change'].astype('category')
    table['agg'] = table['agg'].astype('category')
