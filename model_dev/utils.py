import pandas as pd
import datetime
from tqdm import tqdm
import pickle
import os

import pdb

tqdm.pandas()

def get_training_sample_conditions(
    df, 
    sample_gap_thresh=.2, 
    time_to_hazard_thresh = 75, 
    pressure_thresh=0
):

    train_conditions = (df['sample_gap_minutes'] < sample_gap_thresh) \
                         & (df['time_to_hazard'] < time_to_haxard_thresh) \
                             & (df['pressure'] > 0)
    return train_conditions

def get_taget_condition(df, hours=6):
    return df['time_to_hazard'] < hours

def time_until_next_true(bool_series, debug=False):
    """
    For each datetime index in the series, calculates the time until the next occurrence
    where the series is True.
    
    :param bool_series: A boolean series with a datetime index.
    :return: A series with the time difference until the next True value.
    """
    # Find indices where the condition is True
    true_indices = bool_series.index[bool_series]

    # Function to find time until next True timestamp
    def find_time_until_next_true(current_index):
        future_indices = true_indices[true_indices > current_index]
        return future_indices[0] - current_index if not future_indices.empty else pd.NaT

    # Apply the function to each index
    time_differences = bool_series.index.to_series().apply(find_time_until_next_true)

    if debug:
        return time_differences.progress_apply(lambda x: x.total_seconds() / 3600)
    else:
        return time_differences.apply(lambda x: x.total_seconds() / 3600)

def hours_to_next_pause(time, pauses):
    return (pauses[pauses >= time].min() - time).total_seconds()/3600

def preprocess_data(df, output_var='Extruder Pressure', output_threshold=5500):
    #start_time = datetime.datetime.now()
    processed_df = df.copy()
#    print(f'here 0: calculating sample_gap_minutes, time={start_time}')
#    processed_df['sample_gap_minutes'] = df.index.to_series().diff().apply(lambda x: x.total_seconds() / 60)
    
    #elapsed_str = format_elapsed_time(start_time, datetime.datetime.now())
    #print(f'here 1: calculating rolling maxes, elapsed_time={elapsed_str}')
    processed_df['48H_max_gap_backward'] = processed_df['sample_gap_minutes'].rolling('48H').max()
    processed_df['1H_max_gap_forward'] = processed_df[::-1]['sample_gap_minutes'].rolling('1H').max()
    processed_df['well_sampled'] = (processed_df['48H_max_gap_backward'] < 15) & (processed_df['1H_max_gap_forward'] < 15)

    #elapsed_str = format_elapsed_time(start_time, datetime.datetime.now())
    #print(f'here 2: filling forward, elapsed_time={elapsed_str}')
    processed_df = processed_df.fillna(method='ffill')
    processed_df['HAZARD_CONDITION'] = processed_df[output_var] > output_threshold
    hazard_times = processed_df[processed_df['HAZARD_CONDITION']].index

    #elapsed_str = format_elapsed_time(start_time, datetime.datetime.now())
    #print(f'here 3: calculating time until next hazard region, elapsed_time={elapsed_str}')
    #processed_df['hours_to_hazard'] = time_until_next_true(processed_df['HAZARD_CONDITION'])
    processed_df['hours_to_hazard'] = processed_df.index.to_series().apply(
        lambda x : hours_to_next_pause(x, hazard_times)
    )
    processed_df['hazard_within_24h'] = processed_df['hours_to_hazard'] < 24

#    elapsed_str = format_elapsed_time(start_time, datetime.datetime.now())
#    print(f'COMPLETE, elapsed_time={elapsed_str}')

    return processed_df

def get_extruder_metadata(path, id_str=None):
    
    ex_name_mapping = pd.read_csv(
        path, 
        delimiter='\t',
        encoding='utf-16',
        nrows=1
    )

    field_mappings = dict(zip(ex_name_mapping.iloc[0], ex_name_mapping.columns ))

    col_mapping = {
        key : key.split(' (')[0] for key, value in field_mappings.items() if key != 'Time'
    }

    raw_fieldnames = sorted(col_mapping.values())
    time_keys = [fieldname + '.TIME' for fieldname in raw_fieldnames]

    keys_of_interest = sorted(raw_fieldnames + time_keys)

    return keys_of_interest, time_keys, field_mappings, col_mapping, raw_fieldnames

def get_keys_for_columns(column_map):
    result = {}
    for key, values in column_map.items():
        for value in values:
            result[value] = key
    return result

def resample_and_fill(df, period='15S', sample_gap_threshold = 0.15):
    #print(df.columns)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
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

with open('end2end_labels_map_2.pkl', 'rb') as f:
    default_column_mapping = pickle.load(f)

def process_column_names(df, extruder):
    for col in df.columns:
        if col == 'sample_gap_minutes':
            continue
        else:
            if 'std' in col:
                df['pressure_std_15S'] = df[col]
            else:
                try:
                    df[default_column_mapping[extruder][col]] = df[col]
                except:
                    breakpoint()
        df = df.drop(col, axis=1)
    return df

def merge_data_with_time(
    df, 
    redundancy_func='max', base_sample_period='15S', resample_rate=None, resample_func='median',
    verbose=False, get_series_list=False, column_map=default_column_mapping
):
    """
    Merge data columns with their corresponding time columns in a DataFrame.

    :param df: pandas DataFrame with data and time columns
    :param resample_rate: time period for resampling (e.g., '1H' for 1 hour)
    :param resample_func: aggregation function (e.g., 'mean', 'sum', 'max')
    :return: pandas DataFrame after merging
    """

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
        merged_df = merged_df.resample(resample_rate).apply(resample_func)
    
    if get_series_list:
        return merged_dfs, raw_series_list
    else:
        return merged_dfs


def format_elapsed_time(start, end):
    """
    Format the elapsed time between two datetime objects into a human-readable string.

    :param start: The start time (datetime object).
    :param end: The end time (datetime object).
    :return: A formatted string representing the elapsed time.
    """
    elapsed = end - start
    days = elapsed.days
    hours, remainder = divmod(elapsed.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days > 0:
        return f"{days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"