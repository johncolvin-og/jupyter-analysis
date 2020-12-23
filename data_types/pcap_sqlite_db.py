import sqlite3 as sql
import pandas as pd

def all_table_names() -> list:
    return ['securities', 'events', 'event_items', 'books']

def read_tables(db_path = '', con = None, table_names = all_table_names()) -> dict:
    if con == None:
        con = sql.connect(db_path)
    tables = {}
    for tn in table_names:
        tables[tn] = pd.read_sql(f'select * from {tn}', con)
    return tables
