import pandas as pd
import numpy as np
from pandas.io.stata import ValueLabelTypeMismatch
    
def is_column_type(data_frame, col_name, type_name):
    type_name = type_name.lower()
    if type_name == 'timestamp':
        # this is a bit of a hack, but an effective way to verify if the column
        # is a timestamp/datetime nonethless.
        return len(data_frame[[col_name]].select_dtypes(include=[np.datetime64]).columns) == 1
    elif type_name == 'float':
        return data_frame[col_name].dtype == np.float

def ensure_column_type(data_frame, col_name, type_name):
    # converting types can be time-consuming, so verify that the column
    # is not already of the appropriate type before executing the cast.
    if is_column_type(data_frame, col_name, type_name):
        return
    type_name = type_name.lower()
    if type_name == 'timestamp':
        # this is a bit of a hack, but an effective way to verify if the column
        # is a timestamp/datetime nonethless.
        data_frame[col_name] = data_frame[col_name].apply(lambda t: pd.Timestamp(t, unit='ns'))
    elif type_name == 'float':
        data_frame[col_name] = data_frame[col_name].astype(float)
    elif isinstance(type_name, str) and len(type_name) > 0:
        data_frame[col_name] = data_frame[col_name].astype(type_name)
    else:
        raise ValueError(f"Inappropriate type name '{type_name}' for column '{col_name}' (expected string).")

def default_type_term_dict(table_name = None):
    """The default type_term_dict used by 'cast_column_types.'"""
    if table_name == None or table_name == '':
        return {
            'float': ['price', 'bid_p_', 'ask_p_', 'profit', '_std_', 'deviation', '_var_', 'variance', '_avg_', '_med_', '_quantile_', '_quant_'],
            'category': ['symbol', 'side', 'asset', 'match_algo', 'sid', 'status'],
            'int64': ['bid_q_', 'bid_no_', 'ask_q_', 'ask_no_']
        }
    if table_name == 'event_items':
        return {
            'symbol': 'category',
            'side': 'category',
            'status': 'category',
            'type': 'category',
            'change': 'category',
            'agg': 'category',
            't_time': 'timestamp'
        }
    if table_name == 'securities':
        return {
            'symbol': 'category',
            'sec_type': 'category',
            'cme_sec_type': 'category',
            'sec_group': 'category',
            'asset': 'category',
            'match_algo': 'category'
        }
    if table_name == 'book_stat':
        rv = {
            'price': 'float',
            'avg_price': 'float',
            'variance': 'float',
            'side': 'category'
        }
        for i in range(0, 10):
            for prop in ['price', 'avg_price', 'variance']:
                rv[f'{prop}_{i}'] = 'float'
        return rv
    if table_name == 'log':
        return {
            'src': 'category',
            'entry_type': 'category',
            't_time': 'timestamp',
            'ev_t_time': 'timestamp',
            'ev_o_time': 'timestamp'
        }
    return {}

def cast_column_types(tables, type_term_dict = default_type_term_dict(), require_exact_match = False):
    """Convert one or more DataFrame's column types to something more approrpiate.
    The default type_term_dict accounts for most column-type misnomers in DataFrames read directly from a sql db,
    as numeric columns are frequently considered objects, also converts t_time from int64 to pd.Timestamp.
    For each column, the key in type_term_dict that is contained in the column name
    determines the type to which the column is cast."""
    if isinstance(tables, type(dict)):
        for kv in tables.items():
            cast_column_types(kv[1], kv[0])
        return
    table = tables
    if not isinstance(table, pd.DataFrame):
        raise ValueLabelTypeMismatch("'tables' should be a dict(name, DataFrame), or a DataFrame.")

    def is_match(key, col_name):
        return key == col_name or ((not require_exact_match) and key in col_name)

    for col in table.columns:
        for kv in type_term_dict.items():
            if is_match(kv[0], col):
                ensure_column_type(table, col, kv[1])
                break
            
class DataFrameGroupCache:
    _groups = None
    _cached_groups = None
    
    def __init__(self, groups):
        self._groups = groups
        self._cached_groups = {}
    
    def get_group(self, name):
        if isinstance(self._groups, dict):
            if name in self._groups:
                return self._groups[name]
            print(self._groups)
            raise ValueError(f"Did not find entry for name '{name}.'")
            return None
        if not name in self._cached_groups:
            self._cached_groups[name] = self._groups.get_group(name)
        return self._cached_groups[name]



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
                raise TypeError("'runs_by_date' must be of type pd.DataFrame.");
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

# If value is iterable, return a set containing
# all of those values.  Otherwise, return a set
# with a single element that is the specified value.
# def normalize_to_set(value):
#     if isinstance(value, set):
#         return value
#     try:
#         it = iter(value)


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
    # _sym_to_id = {}
    
    def __init__(self, securities):
        if not isinstance(securities, pd.DataFrame):
            raise ValueError(f"'securities' must be a 'DataFrame' (actually '{type(securities)}'').")
        cast_column_types(securities, default_type_term_dict('securities'))
        securities = securities.set_index('sid')
        self._securities = securities

    # def _ensure_sym_to_id(self):
    #     if len(self._sym_to_id) == 0 and len(self._securities) > 0:
    #         for sym in self._securities['symbol']:
    #             sym_id = self._securities.loc[self._securities.symbol == sym].index.values
    #             if len(sym_id) > 0:
    #                 self._sym_to_id[sym] = sym_id[0]

    @property
    def securities(self):
        return self._securities

    def symbol_for_id(self, id):
        return self._securities.at[id, 'symbol']
        # self._ensure_sym_to_id()
        # for kv in self._sym_to_id.items():
        #     if kv[1] == id:
        #         return kv[0]
        # return None

    def id_for_symbol(self, symbol):
        return self._securities.loc[self._securities.symbol == symbol].index.values[0]
        # self._ensure_sym_to_id()
        # if symbol in self._sym_to_id:
        #     return self._sym_to_id[symbol]
        # return None
 

def merge_run_with_books(date = '2020-09-10', symbols = ['ESU0'], run_id=2) -> pd.DataFrame:
    l = log.log_for_date(date).log_for_run(run_id)
    def add_sid_col(df):
        df = df.merge(securities['sid'], on='symbol')
    saved = l.saved_fills.loc[l.saved_fills.symbol.isin(symbols)]
    clean = l.clean_fills.loc[l.clean_fills.symbol.isin(symbols)]
    forced = l.forced_fills.loc[l.forced_fills.symbol.isin(symbols)]
    clean['rgtm'] = 1
    saved['rgtm'] = 1
    forced['rgtm'] = -1
    #sample_size = min(len(saved), len(forced))
    sample = pd.concat([clean, saved, forced])
    sample = sample.set_index('ok', drop=False)
    sample.index.name = '_ok'
    sample = sample.sort_values(by='ev_eid1_order')
    books = book_stats.filter_security([securities.id_for_symbol(s) for s in symbols])
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

def bbo_cols():
    return ['bid_q_0', 'bid_p_0', 'ask_p_0', 'ask_q_0']
    
def bbo_avg_cols(secs):
    return [f'{c}_avg_{secs}s' for c in bbo_cols()]

def bbo_var_cols(secs):
    return [f'{c}_var_{secs}s' for c in bbo_cols()]

def bbo_var_inter_period_delta(secs1, secs2):
    return [f'{c}_var_{secs1}s_{secs2}s_delta' for c in bbo_cols()]

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