import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


class AlgoOrderFills:
    _log = None
    _log_groups = None
    _orders = None
    _new_orders = None
    _fills = None
    _order_fills = None
    _forced_fills = None
    _saved_fills = None
    _clean_fills = None
    
    def __init__(self, log: pd.DataFrame):
        self._log_groups = log.groupby('entry_type')

    @property
    def orders(self):
        if self._orders == None:
            self._orders = self._log_groups.get_group('algo_order')
            self._orders.drop(columns=['entry_type', 'src', 'h_t_time', 'val8', 'val9', 'val10'], inplace=True)
            self._orders.rename(columns={'val1': 'opp_id', 'val2': 'change_type', 'val3': 'symbol', 'val4': 'price', 'val5': 'qty', 'val6': 'side', 'val7': 'ok'}, inplace=True)
            self._new_orders = self._orders.loc[self._orders.change_type == 'added']
            #self._orders = self._orders.astype({'qty': 'int32', 'price': float, 'ok': np.uint64, 'opp_id': np.uint64, 'symbol': 'category', 'side': 'category', 'change_type': 'category' })
            # this is so stupid, but .loc[category_column == 'blah'] doesn't work.
            # neither does .loc[category_column == pd.Categorical('blah')]
            # wtf pandas!?  Apparently no documentation on this exists, or if it does, it is locked in a super secret vault somewhere
            self._orders = self._orders.astype({'qty': 'int32', 'price': float, 'ok': np.uint64, 'opp_id': np.uint64, 'symbol': 'category', 'side': 'category'})
            self._new_orders.drop(columns=['change_type'], inplace=True)
            self._orders = self._orders.astype({'change_type': 'category'})
        return self._orders
    
    @property
    def new_orders(self):
        if self._new_orders == None:
            o = self.orders
            #self._new_orders = self.orders.loc[(self.orders.change_type.eq(pd.Categorical('added'))]
        return self._new_orders

    @property
    def fills(self) -> pd.DataFrame:
        if self._fills == None:
            self._fills = self._log_groups.get_group('algo_fill')
            self._fills.drop(columns=['entry_type', 'h_t_time', 'src', 'val7', 'val8', 'val9', 'val10'], inplace=True)
            self._fills.rename(columns={'val1': 'status', 'val2': 'side', 'val3': 'symbol', 'val4': 'price', 'val5': 'qty', 'val6': 'ok'}, inplace=True)
            self._fills = self._fills.astype({'qty': 'int32', 'price': float, 'ok': np.uint64, 'symbol': 'category', 'side': 'category'})
        return self._fills

    @property
    def order_fills(self) -> pd.DataFrame:
        if self._order_fills == None:
            self._order_fills = self.new_orders.merge(self.fills.drop(columns=['side', 'symbol']), how='left', on=['run_id', 'ok'], suffixes=['_order', '_fill'])
            self._order_fills['fill_price_delta'] = self._order_fills['price_fill'] - self._order_fills['price_order']
            self._order_fills['fill_t_time_delta'] = self._order_fills['t_time_fill'] - self._order_fills['t_time_order']
            self._order_fills['fill_ev_eid_delta'] = self._order_fills['ev_eid1_fill'] - self._order_fills['ev_eid1_order']
            self._order_fills['fill_t_timestamp_delta'] = self._order_fills['fill_t_time_delta'].apply(lambda t: pd.Timedelta(t, unit='n'))
            self._order_fills['fill_profit_delta'] = self._order_fills['fill_price_delta']
            self._order_fills.loc[self._order_fills['side'] == 'SIDE_BUY', 'fill_profit_delta'] *= -1
        return self._order_fills
        
    @property
    def forced_fills(self) -> pd.DataFrame:
        if self._forced_fills == None:
            self._forced_fills = self.order_fills[self.order_fills.fill_price_delta != 0]
        return self._forced_fills

    @property
    def saved_fills(self) -> pd.DataFrame:
        if self._saved_fills == None:
            self._saved_fills = self.order_fills.loc[(self.order_fills.fill_price_delta == 0) & (self.order_fills.fill_t_time_delta != 0)]
        return self._saved_fills

    @property
    def clean_fills(self) -> pd.DataFrame:
        if self._clean_fills == None:
            self._clean_fills = self.order_fills.loc[(self.order_fills.fill_price_delta == 0) & (self.order_fills.fill_t_time_delta == 0)]
        return self._clean_fills


def display(data_frame: pd.DataFrame, view=None):
    col_styles = {}
    for c in data_frame.columns:
       if 'price' in c:
          col_styles[c] = "{:.2f}"
       elif 'timestamp' in c:
          col_styles[c] = lambda t: f'{t.total_seconds():.2f}'
    display = data_frame[['run_id', 'ok', 'ev_eid1_order', 'fill_ev_eid_delta', 'fill_t_timestamp_delta',
       'side', 'price_order', 'price_fill', 'fill_price_delta', 'fill_profit_delta', 'qty_order', 'qty_fill']]
    display.style.set_properties(**{'text-align': 'right'})
    display = display.head(25)
    return display.style.format(col_styles).set_table_styles([
       dict(selector='td', props=[('text-align', 'right')])
       #dict(selector='tr', props=[('background', '#88ff0000')])
    ]).apply(lambda x: LinearSegmentedColormap.from_list('', [(0,0,1),(0,0,0),(1,0,0)]), subset=['fill_profit_delta'])
