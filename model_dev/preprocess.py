import pandas as pd
import datetime
from tqdm import tqdm
import pickle
import os

import pdb

with open('end2end_labels_map.pkl', 'rb') as f:
    default_column_mapping = pickle.load(f)

def get_keys_for_columns(column_map):
    result = {}
    for key, values in column_map.items():
        for value in values:
            result[value] = key
    return result

debug = False
if debug:
    print(default_column_mapping)

def merge_data_with_time(
    df, 
    redundancy_func='max', base_sample_period='15S', resample_rate=None, resample_func='median',
    verbose=False, get_series_list=False, column_map=default_column_mapping
):
    
    raw_series_list = []
    processed_series_lists = { 
        key: [] for key in ['EX1', 'EX2', 'EX3', 'EX4']
    }
    merged_dfs = { 
        key: None for key in ['EX1', 'EX2', 'EX3', 'EX4'] 
    }
    
    column2extruder_map = get_keys_for_columns(column_map)

    for column in df.columns:
        if not column.endswith('.TIME'):
            time_column = column + '.TIME'
            if time_column in df.columns:
                if column in column2extruder_map.keys():
                    extruder_id = column2extruder_map[column]
                else:
                    print(f'[WARNING] no entry for {column} in mapping, skipping.')
                    continue

                series = pd.Series(df[column].values, 
                    index=pd.to_datetime(df[time_column]), 
                    name=column
                )
                raw_series_list.append(series)
                series = series[series.index.notnull()]
                series.sort_index(inplace=True)
                if 'PT' in column:
                    local_std = series.rolling(base_sample_period).std()
                    local_std.name = f'{column}_{base_sample_period}_std'
                if redundancy_func == 'max':
                    series = series.groupby(level=0).max()
                elif redundancy_func == 'median':
                    series = series.groupby(level=0).median()
                else:
                    series = series.groupby(level=0).mean()

                if verbose:
                    print(series.head())
                processed_series_lists[extruder_id].append(series)
                
                if 'PT' in column:
                    local_std = local_std.loc[series.index].groupby(level=0).max()
                    processed_series_lists[extruder_id].append(local_std)
    
    for extruder_id in merged_dfs.keys():
        merged_dfs[extruder_id] = pd.concat(processed_series_lists[extruder_id], axis=1, join='outer')
        merged_dfs[extruder_id]['sample_gap_minutes'] = \
            merged_dfs[extruder_id].index.to_series().diff().apply(lambda x: x.total_seconds() / 60)
        
        if resample_rate:
            merged_dfs[extruder_id] = merged_df[extruder_id].resample(resample_rate).apply(resample_func)
        
    if get_series_list:
        return merged_dfs, raw_series_list
    else:
        return merged_dfs

def resample_and_fill(df, period='15S', sample_gap_threshold = 0.15):
    #df['timestamp'] = pd.to_datetime(df['timestamp'])
    #df = df.set_index('timestamp')
    sample_gap = df['sample_gap_minutes'].resample(period).max()
    max_gap = sample_gap.rolling('24H').max()
    df_resampled = df.resample(period).median()
    
    df_resampled = df_resampled.ffill()
    df_resampled['sample_gap_minutes'] = sample_gap.loc[df_resampled.index]
    df_resampled['max_gap_24h'] = max_gap.loc[df_resampled.index]
    #print(df_resampled.columns)

    #df_resampled = df_resampled[(df_resampled['pressure'] > 0) & (df_resampled['sample_gap_minutes'] < sample_gap_threshold)]
    df_resampled = df_resampled.dropna()
    
    return df_resampled

def process_column_names(df, extruder):
    for col in df.columns:
        if col == 'sample_gap_minutes':
            continue
        else:
            if 'std' in col:
                df['pressure_std_15S'] = df[col]
            else:
                df[default_column_mapping[extruder][col]] = df[col]
        df = df.drop(col, axis=1)
    return df