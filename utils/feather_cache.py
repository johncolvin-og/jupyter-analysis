import pandas as pd
import pyarrow.feather as feather
import os
import sqlite3 as sql

class Cache:
   tables = {}
   path = ""
   feather_path = ""

 
class _FileStore:
   _caches = {}
   _default_path = "data/310-short"
   _default_feather_path = "data/cache/310-short"

   @property
   def caches(self):
      return type(self)._caches
   
   @property
   def default_path(self):
      return type(self)._default_path

   @default_path.setter
   def caches(self, value):
      type(self).default_path = value
   
   @property
   def default_feather_path(self):
      return type(self)._default_path

   @default_feather_path.setter
   def default_feather_path(self, value):
      type(self)._default_feather_path = value



def configure(sql_path: str, feather_path: str):
   _FileStore()._default_path = sql_path
   _FileStore()._default_feather_path = feather_path

def cache(path, feather_path) -> Cache:
   if path == None:
      return None
   if path not in _FileStore().caches:
      if feather_path == None:
         dirs = path.split('/')
         if dirs[0] == 'data':
            del dirs[0]
         feather_path = f"data/cache/{'/'.join(dirs)}"
      c = Cache()
      c.path = path
      c.feather_path = feather_path
      _FileStore().caches[path] = c
   return _FileStore().caches[path]
 
# non-recursive
def files_of_type(dir, ext):
    return (f for f in os.listdir(dir) if f.endswith('.' + ext))

def read_data_frame(table_name, sql_path = None, feather_path = None):
   if sql_path == None:
      sql_path = _FileStore().default_path
   c = cache(sql_path, feather_path)
   if table_name in c.tables:
      return c.tables[table_name]
   fp = f'{c.feather_path}/{table_name}.feather'
   if os.path.isfile(fp):
      print(f"Reading table from '{fp}'.")
      c.tables[table_name] = feather.read_feather(fp)
      return c.tables[table_name]
   else:
      # write table to feather file
      print(f"No feather file detected in path '{c.feather_path}'.  Will attempt to read table from database in '{sql_path}'.")
      dbs = files_of_type(sql_path, '.sqlite')
      if len(dbs) == 0:
         print(f"No sqlite file found in dir '{sql_path}'.")
         return None
      elif len(dbs) > 1:
         print(f"More than one sqlite file found in dir '{sql_path}'.")
         return None
      con = sql.connect(dbs[0])
      table = pd.read_sql(f'select * from {table_name}', con)
      c.tables[table_name] = table
      print(f"Finished reading table '{table_name}'.  Begin writing to feather '{fp}'.")
      feather.write_feather(table, fp)
      print(f"Done writing '{fp}'.")
      c.tables = table
      return table