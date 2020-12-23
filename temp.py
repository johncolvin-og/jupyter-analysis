import pandas as pd
import numpy as np
from pandas.io.stata import ValueLabelTypeMismatch

class LogData:
    log_by_run = None
    orders = None
    fills = None
    fills_by_run = None
    clean_fills = None
    saved_fills = None
    forced_fills = None

 
def demux_log(log: pd.DataFrame) -> LogData:
    # cast columns
    log['t_time'] = log['t_time'].apply(lambda t: pd.Timedelta(t, unit='ns'))
    # tables
    log_by_run = log.groupby('run_id')
    log_by_type = log.groupby('entry_type')
    # orders
    orders = log_by_type.get_group('algo_order')
    orders.drop(columns=['entry_type', 'src', 'h_t_time', 'val8', 'val9', 'val10'], inplace=True)
    orders.rename(columns={'val1': 'opp_id', 'val2': 'change_type', 'val3': 'symbol', 'val4': 'price', 'val5': 'qty', 'val6': 'side', 'val7': 'ok'}, inplace=True)
    orders = orders.loc[orders.change_type == 'added']
    orders = orders.astype({'qty': 'int32', 'price': float, 'ok': np.uint64, 'opp_id': np.uint64, 'symbol': 'category', 'side': 'category', 'change_type': 'category' })
    # fills
    fills = log_by_type.get_group('algo_fill')
    fills.drop(columns=['entry_type', 'h_t_time', 'src', 'val7', 'val8', 'val9', 'val10'], inplace=True)
    fills.rename(columns={'val1': 'status', 'val2': 'side', 'val3': 'symbol', 'val4': 'price', 'val5': 'qty', 'val6': 'ok'}, inplace=True)
    fills = fills.astype({'qty': 'int32', 'price': float, 'ok': np.uint64, 'symbol': 'category', 'side': 'category'})
    # merge w/ orders
    fills = orders.merge(fills.drop(columns=['side', 'symbol']), how='left', on=['run_id', 'ok'], suffixes=['_order', '_fill'])
    fills['fill_price_delta'] = fills['price_fill'] - fills['price_order']
    fills['fill_t_time_delta'] = fills['t_time_fill'] - fills['t_time_order']
    fills['fill_ev_eid_delta'] = fills['ev_eid1_fill'] - fills['ev_eid1_order']
    # profit delta is side-agnostic (fill_price > order_price for BUY is loss, for SELL is profit)
    fills['fill_profit_delta'] = fills['fill_price_delta']
    fills.loc[fills['side'] == 'SIDE_BUY', 'fill_profit_delta'] *= -1
    fills_by_run = fills.groupby('run_id')
    # put it all together
    rv = LogData()
    rv.log_by_run = log_by_run
    rv.orders = orders
    rv.fills = fills
    rv.fills_by_run = fills_by_run
    rv.clean_fills = fills.loc[(fills.fill_price_delta == 0) & (fills.fill_t_time_delta == 0)]
    rv.saved_fills = fills.loc[(fills.fill_price_delta == 0) & (fills.fill_t_time_delta > 0)]
    rv.forced_fills = fills.loc[(fills.fill_profit_delta < 0) & (fills.fill_t_time_delta > 0)]
    return rv
