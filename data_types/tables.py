import pandas as pd
import numpy as np
from pandas.core.groupby.generic import DataFrameGroupBy
from .data_frame_sanitation import ensure_column_type, cast_column_types, default_type_term_dict
from .data_frame_group_cache import DataFrameGroupCache
from .known_columns import bbo_avg_cols, bbo_cols, bbo_var_cols, bbo_var_inter_period_delta, book_cols
from pandas.io.stata import ValueLabelTypeMismatch


class LogData:
    _log = None
    _log_by_date = {}
    _log_by_date_groups = None
    _log_by_run = {}
    _log_by_run_groups = None
    _runs_by_date = {}
    _fills_by_run = None
    _orders = None
    _fills = None
    _clean_fills = None
    _saved_fills = None
    _forced_fills = None
    
    def link_runs(self, runs_by_date: dict, force_overwrite = False):
        """Provide date/run mapping to enable 'log_for_run' Once linked to a non-empty runs_by_date dict,
        subsequent calls to 'link_runs' will be ignored, unless force_overwrite is True."""
        if len(self._runs_by_date) > 0 and not force_overwrite:
            return
        self._runs_by_date = runs_by_date
        self._log_by_date = {}

    def __init__(self, log: pd.DataFrame, runs_by_date = None):
        ensure_column_type(log, 't_time', 'timestamp')
        # tables
        log_by_run_groups = log.groupby('run_id')
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
        fills_by_run_group = fills.groupby('run_id')
        # put it all together
        self._log = log
        self._log_by_run_groups = log_by_run_groups
        self._orders = orders
        self._fills = fills
        self._fills_by_run_group = fills_by_run_group
        # filled on entry
        self._clean_fills = fills.loc[(fills.fill_profit_delta >= 0) & (fills.fill_ev_eid_delta == 0)]
        # filled after entry (passive)
        self._saved_fills = fills.loc[(fills.fill_profit_delta == 0) & (fills.fill_ev_eid_delta > 0)]
        # filled after entry (aggressor)
        self._forced_fills = fills.loc[(fills.fill_profit_delta < 0) & (fills.fill_ev_eid_delta > 0)]
        if runs_by_date != None:
            if not isinstance(runs_by_date, dict):
                raise TypeError("'runs_by_date' must be of type pd.DataFrame.")
            self.link_runs(runs_by_date)
    
    @property
    def orders(self) -> pd.DataFrame:
        return self._orders
    
    @property
    def fills(self) -> pd.DataFrame:
        return self._fills
    
    @property
    def clean_fills(self) -> pd.DataFrame:
        return self._clean_fills
    
    @property
    def saved_fills(self) -> pd.DataFrame:
        return self._saved_fills
    
    @property
    def forced_fills(self) -> pd.DataFrame:
        return self._forced_fills
    
    @property
    def actual_log(self) -> pd.DataFrame:
        return self._log
    
    def log_for_run(self, run_id):
        if not run_id in self._log_by_run:
            self._log_by_run[run_id] = LogData(self._log_by_run_groups.get_group(run_id))
        return self._log_by_run[run_id]
    
    def log_for_date(self, date):
        if not date in self._log_by_date:
            if not date in self._runs_by_date:
                #raise ValueError("No runs for the specified date.")
                return LogData(self._log.loc[False])
            run_ids = set(self._runs_by_date[date]['run_id'].unique())
            date_log = self._log.loc[self._log.run_id.isin(run_ids)]
            date_entry_type_groups = date_log.groupby("entry_type")
            # unf the code in __init__ that initialize '_orders' depends
            # on the existence of an 'algo_order' group.
            # TODO: initialize orders to a DataFrame with the proper columns
            # (but no rows) if the 'algo_order' group is absent.
            if not 'algo_order' in date_entry_type_groups.groups.keys():
                raise ValueError(f"No algo order group for date {date}")
            self._log_by_date[date] = LogData(self._log.loc[self._log.run_id.isin(run_ids)])
        return self._log_by_date[date]
    
    def fills_for_run(self, run_id) -> pd.DataFrame:
        return self._fills_by_run.get_group(run_id)
 

def add_rolling_book_stats(data: pd.DataFrame, stats=['avg', 'std', 'var'], secs=[1,5], sides=['bid', 'ask'], props=['p', 'q'], levels=[0]):
    for side in sides:
        for level in levels:
            for prop in props:
                for stat in stats:
                    for sec in secs:
                        prefix = f'{side}_{prop}_{level}'
                        if stat == 'avg' or stat == 'mean':
                            data[f'{prefix}_{stat}_{sec}s'] = data[prefix].rolling(pd.Timedelta(sec, unit='s')).mean()
                        elif stat == 'variance' or stat == 'var':
                            data[f'{prefix}_{stat}_{sec}s'] = data[prefix].rolling(pd.Timedelta(sec, unit='s')).var()
                        elif stat == 'deviation' or stat == 'dev' or stat == 'std':
                            data[f'{prefix}_{stat}_{sec}s'] = data[prefix].rolling(pd.Timedelta(sec, unit='s')).std()

def add_rolling_book_stat_deltas(data: pd.DataFrame, stats=['avg', 'std', 'var'], secs=[1,5], sides=['bid', 'ask'], props=['p', 'q'], levels=[0]):
    """Adds the difference between rolling window columns of different window sizes."""
    for side in sides:
        for level in levels:
            for prop in props:
                for stat in stats:
                    for i in range(0, len(secs) - 1):
                        prefix = f'{side}_{prop}_{level}'
                        data[f'{prefix}_{stat}_{secs[i]}_{secs[i+1]}s_delta'] = data[f'{prefix}_{stat}_{secs[i]}s'] - data[f'{prefix}_{stat}_{secs[i+1]}s']

def add_row_rolling_book_stat_deltas(data: pd.DataFrame, stats=['avg', 'std', 'var'], secs=[1,5], sides=['bid', 'ask'], props=['p', 'q'], levels=[0]):
    for side in sides:
        for level in levels:
            for prop in props:
                for stat in stats:
                    for sec in secs:
                        prefix = f'{side}_{prop}_{level}'
                        col = f'{prefix}_{stat}_{sec}s'
                        data[f'{col}_delta'] = data[col] - data[col].shift(-1)

def add_row_book_deltas(data: pd.DataFrame, sides=['bid', 'ask'], props=['p', 'q'], levels=[0]):
    for side in sides:
        for level in levels:
            for prop in props:
                col = f'{side}_{prop}_{level}'
                data[f'{col}_delta'] = data[col] - data[col].shift(-1)
    

def add_row_book_stat_deltas(data: pd.DataFrame, sides=['bid', 'ask'], props=['avg_price', 'variance', 'total_qty']):
    for side in sides:
        for prop in props:
            col = f'{side}_{prop}'
            data[f'{col}_delta'] = data[col] - data[col].shift(-1)

def merge_book_stat_sides(book_stats):
    bid_stats = book_stats.loc[book_stats.side == 'Buy']
    ask_stats = book_stats.loc[book_stats.side == 'Sell']
    # cols per side (bid, ask).  Note that 'price_i' is the top price 4*i qty into the market
    #   * price_0 is always the best price (i.e., the top bid/ask)
    #   * if there were 4 qty at the top price level, then price_1 would be the
    #     second price level
    #   * if there were 1 qty at the 1st level, 1 at the second, and 3 at the third,
    #     then price_1 would be the third price level.  If there were only 2 qty
    #     at the third level, then price_1 would be the 4th price level
    #   * To really hammer it home, if there were 37 qty at the first price level,
    #     then price_i, i=0,...,9 would all be the top price.
    # avg_price_i, and variance_i work similarly to price_i.  They represent the
    # avg price, and variance of the ith block of qty in the market.
    side_cols = ['price', 'avg_price', 'avg_price_delta', 'variance', 'variance_delta', 'total_qty']
    for i in range(0, 10):
        side_cols.append(f'price_{i}')
        side_cols.append(f'avg_price_{i}')
        side_cols.append(f'variance_{i}')
    
    bid_col_map = {}
    ask_col_map = {}
    for col in side_cols:
        bid_col_map[col] = f'bid_{col}'
        ask_col_map[col] = f'ask_{col}'
    bid_stats = bid_stats.rename(columns=bid_col_map).drop(columns='side')
    ask_stats = ask_stats.rename(columns=ask_col_map).drop(columns='side')
    return bid_stats.merge(ask_stats, on=['sid', 'eid'])


class SecBookData:
    _books = None
    _book_stats = None
    _book_stats_by_time = None
    _book_stats_by_time_books_join = None

    def __init__(self, books, book_stats, book_stats_by_time, book_stats_by_time_books_join):
        self._books = books
        self._book_stats = book_stats
        self._book_stats_by_time = book_stats_by_time
        self._book_stats_by_time_books_join = book_stats_by_time_books_join

    @property
    def book_stats_by_time_books_join(self):
        return self._book_stats_by_time_books_join


class BookStatsData:
    _sec_books_groups = None
    _sec_book_stats_groups = None
    _sec_books_dict = None
    _sec_book_stats = {}
    _events = None

    def __init__(self, books, book_stats, events):
        if not isinstance(book_stats, pd.DataFrame):
            raise ValueError(f"'book_stats' must be a 'DataFrame' (actually '{type(book_stats)}'').")
        book_stats = book_stats.set_index('eid')
        self._events = events
        self._sec_books_groups = DataFrameGroupCache(books)
        self._sec_book_stats_groups = DataFrameGroupCache(book_stats.groupby('sid'))
    
    def filter_security(self, ids):
        def filter_single_security(id):
            # if not id in self._sec_book_stats_by_time:
            #     if id in self._sec_book_stats:
            #         raise ValueError("Internal dictionaries are out of sync: '_sec_book_stats_by_time' doesn't have entry for security '{sym},' but '_sec_book_stats' does.")
            if not id in self._sec_book_stats:
                sbs = self._sec_book_stats_groups.get_group(id).sort_values(by='eid')
                sbs = merge_book_stat_sides(sbs)
                add_row_book_stat_deltas(sbs)
                sbs = sbs.merge(self._events[['eid', 't_time']], on='eid')

                # sec_books['u0'].set_index('eid', inplace=True, drop=True)
                # sec_books['u0'].sort_values(by='eid', inplace=True)
                # sec_books['u0'] = sec_books['u0'].merge(sec_book_stats_by_time['u0'], on='eid')
                
                bytime = sbs.set_index('t_time')
                props=['price', 'avg_price']
                add_rolling_book_stats(bytime, props=props)
                add_rolling_book_stat_deltas(bytime, props=props)
                add_row_rolling_book_stat_deltas(bytime, props=props)
                bytime['cross_bbo_delta'] = bytime['ask_price'] - bytime['bid_price']

                sec_books = self._sec_books_groups.get_group(id)
                sec_books = sec_books.set_index('eid').sort_values(by='eid')
                bytime_join = sec_books.merge(bytime.drop(columns='sid'), on='eid')
                self._sec_book_stats[id] = SecBookData(sec_books, sbs, bytime, bytime_join)
            return self._sec_book_stats[id].book_stats_by_time_books_join
        if isinstance(ids, list):
            return self.filter_security(set(ids))
        if isinstance(ids, set):
            if len(ids) == 1:
                return filter_single_security(next(iter(ids)))
            rv = None
            for id in ids:
                id_bks = self.filter_security(id).set_index(['eid', 'sid'], drop=False)
                rv = id_bks if rv is None else pd.concat([rv, id_bks])
            return rv
        return filter_single_security(ids)

class SecuritiesData:
    _securities = None
    
    def __init__(self, securities):
        if not isinstance(securities, pd.DataFrame):
            raise ValueError(f"'securities' must be a 'DataFrame' (actually '{type(securities)}'').")
        cast_column_types(securities, default_type_term_dict('securities'))
        securities = securities.set_index('sid')
        self._securities = securities

    @property
    def securities(self):
        return self._securities

    def symbol_for_id(self, id):
        return self._securities.at[id, 'symbol']

    def id_for_symbol(self, symbol):
        return self._securities.loc[self._securities.symbol == symbol].index.values[0]
 

def merge_run_with_books(log: LogData, securities: pd.DataFrame, books: pd.DataFrame, date: str, sym: str, run_id: int) -> pd.DataFrame:
    l = log.log_for_date(date).log_for_run(run_id)
    saved = l.saved_fills.loc[l.saved_fills.symbol == sym]
    clean = l.clean_fills.loc[l.clean_fills.symbol == sym]
    forced = l.forced_fills.loc[l.forced_fills.symbol == sym]
    clean['rgtm'] = 1
    saved['rgtm'] = 1
    forced['rgtm'] = -1
    #sample_size = min(len(saved), len(forced))
    sample = pd.concat([clean, saved, forced])
    sample = sample.set_index('ok', drop=False)
    sample.index.name = '_ok'
    sample = sample.sort_values(by='ev_eid1_order')
    books = books.filter_security(securities.id_for_symbol(sym))
    return pd.merge_asof(sample, books, left_on='ev_eid1_order', right_on='eid')

def runs_with_parameters(table: pd.DataFrame, params: dict):
    def fmt_is_match_col(kv):
        return f'{kv[0]}_is_match'
    # First, add 'param_is_match' boolean cols for each param.
    # Then, combine those cols into an 'all_params_match' boolean col,
    # which can be used to filter the rows.
    param_cols = [f'param{i}' for i in range(1, 12)]
    for kv in params.items():
        table[fmt_is_match_col(kv)] = False
        for pcol in param_cols:
            table[fmt_is_match_col(kv)] |= table[pcol] == f'{kv[0]}: {kv[1]}'
    table['all_params_match'] = True
    for pcol in param_cols:
        for kv in params.items():
            table['all_params_match'] &= table[fmt_is_match_col(kv)]
    rows_with_all_params = table.loc[table.all_params_match]
    # remove the temp cols
    table.drop(columns=[fmt_is_match_col(kv) for kv in params.items()] + ['all_params_match'], inplace=True)
    return rows_with_all_params

def demux_run_by_date(run: pd.DataFrame) -> dict:
    """Map runs by date."""
    run_date_groups = run.groupby('market_date')
    run_by_date = {}
    for date in run['market_date'].unique():
        run_by_date[date] = run_date_groups.get_group(date)
    return run_by_date

def add_bbo_cross_delta(data: pd.DataFrame):
    data['bbo_cross_delta'] = data['ask_p_0'] - data['bid_p_0']

# def add_weighted_price_movement(data: pd.DataFrame, sides=['bid', 'ask']):
#     for side in sides:
#         data['{side}_weighted_move_0'] = 

#def merge_book_with_book_stats(book: pd.DataFrame, book_stat: pd.DataFrame):
#    book


def fills_core_columns():
    return ['run_id', 'ok', 'ev_eid1_order', 'fill_ev_eid_delta', 'fill_t_time_delta',
        'side', 'price_order', 'price_fill', 'fill_price_delta', 'fill_profit_delta', 'qty_order', 'qty_fill']