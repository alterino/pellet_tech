import pandas as pd
import datetime
from tqdm import tqdm
tqdm.pandas()

def time_until_next_true(bool_series):
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
    #print(keys_of_interest)
    return keys_of_interest, time_keys, field_mappings, col_mapping, raw_fieldnames

def merge_data_with_time(
    df, 
    resample_rate='30S', resample_func='max',
    verbose=False, get_series_list=False
):
    """
    Merge data columns with their corresponding time columns in a DataFrame.

    :param df: pandas DataFrame with data and time columns
    :param resample_rate: time period for resampling (e.g., '1H' for 1 hour)
    :param resample_func: aggregation function (e.g., 'mean', 'sum', 'max')
    :return: pandas DataFrame after merging
    """

    raw_series_list = []
    processed_series_list = []

    # throwing errors when trying to use custom functions
#    if isinstance(resample_func, str):
#        resample_func = getattr(pd.core.groupby.SeriesGroupBy, resample_func)

    for column in df.columns:
        if not column.endswith('.TIME'):
            time_column = column + '.TIME'

            if time_column in df.columns:
                series = pd.Series(df[column].values, index=pd.to_datetime(df[time_column]), name=column)
                raw_series_list.append(series)
                series = series.groupby(level=0).max()
                if 'PT' in column:
                    local_variance = series.rolling(resample_rate).var()
                    local_variance.name = f'{column}_variance'
                    processed_series_list.append(local_variance)   

                #print(f'HERE resample_func = {resample_func}')

                if verbose:
                    print(series.head())
                processed_series_list.append(series)
                     
    try:
        merged_df = pd.concat(processed_series_list, axis=1, join='outer')
    except Exception as e:
        print(f'error trying to merge dataframe\nerror: {e}')
        if get_series_list:
            return None, raw_series_list
        else:
            return None

    merged_df['sample_gap_minutes'] = merged_df.index.to_series().diff().apply(lambda x: x.total_seconds() / 60)
    
    if resample_rate:
        merged_df = merged_df.resample(resample_rate).apply(resample_func)
    
    if get_series_list:
        return merged_df, raw_series_list
    else:
        return merged_df


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