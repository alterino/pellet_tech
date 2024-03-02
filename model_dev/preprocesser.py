from pydantic import BaseModel. validatory
from datetime import datetime
from typing import Optional, Union

import pandas as pd

class RawData(BaseModel):
    timestamp: datetime
    machine_id: int
    pressure: float
    temperature: float
    thrust: Optional[float]
    velocity_screw_output: Optional[float]
    velocity_screw: Optional[float]
    current_screw : Optional[float]
    current_conveyor : Optional[float]
    velocity_conveyor : Optional[float]
    fault_detected: Optional[bool]
    noise_level: Optional[float] = None
    production_rate: Optional[float]
    fault_detected: Optional[bool] = None  # For training data

    class Config:
        orm_mode = True


class CleanedDate(BaseModel):
    timestamp: datetime
    machine_id: int
    pressure: float
    temperature: float
    thrust: float
    velocity_screw_output: float
    velocity_screw: float
    current_screw : float
    current_conveyor : float
    velocity_conveyor : float
    fault_detected: Optional[bool]
    noise_level: Optional[float] = None
    production_rate: Optional[float]
    fault_detected: Optional[bool] = None  # For training data

    class Config:
        orm_mode = True


class ProcessData(BaseModel):
    id: int
    device_id: int
    date: datetime
    fault_code: str
    cleaned_data: Optional[CleanedData]
    raw_data: Optional[RawData]

    class Config:
        orm_mode=True

class Preprocessor:
    def __init__(self, time_column, downsample_freq=None):
        """
        Initialize the Preprocessor with configuration for processing.
        
        Parameters:
        - time_column: The name of the column containing the timestamp data.
        - downsample_freq: The frequency for downsampling. Example: '1H' for 1 hour.
        """
        self.time_column = time_column
        self.downsample_freq = downsample_freq

    def parse_and_merge(self, df_list):
        """
        Parse and merge multiple DataFrames, sort by time column.
        
        Parameters:
        - df_list: A list of pandas DataFrames to be merged and sorted.
        
        Returns:
        - A single pandas DataFrame, merged and sorted by the time column.
        """
        #df_merged = pd.concat(df_list)
        #df_merged[self.time_column] = pd.to_datetime(df_merged[self.time_column])
         #df_sorted = df_merged.sort_values(by=self.time_column)
        #return df_sorted.reset_index(drop=True)

    if type(machine_date == str:
        machine_data = pd.read_csv(machine_data)

    merged_df, series_list = utils.merge_data_with_time(
        ex_data, resample_func='median',
        verbose=False, get_series_list=True
    )

    for extid in [1, 2, 3, 4]:
        extruder = f'EX{extid}'
        for raw_colname in column_map[f'{extruder}'].keys():
            merged_df[field_mappings[extruder][raw_colname]] = merged_df[raw_colname]
            processed_colnames = list(field_mappings[extruder].values())
            processed_colnames.remove('Date and Time')
            merged_df = merged_df[processed_colnames + ['sample_gap_seconds']]

    processed_df = utils.preprocess_data(merged_df)

    return None

    def downsample(self, df):
        """
        Downsample the DataFrame to the specified frequency.
        
        Parameters:
        - df: The pandas DataFrame to downsample.
        
        Returns:
        - A downsampled pandas DataFrame.
        """
        if self.downsample_freq:
            df.set_index(self.time_column, inplace=True)
            df_downsampled = df.resample(self.downsample_freq).mean()
            df_downsampled.reset_index(inplace=True)
            return df_downsampled
        else:
            return df

    def generate_features(self, df):
        """
        Generate new features from the existing data.
        
        Parameters:
        - df: The pandas DataFrame for which to generate features.
        
        Returns:
        - A pandas DataFrame with new features added.
        """
        # Example feature generation (customize as needed)
        df['feature1'] = df['column1'] / df['column2']
        # Add more feature generation steps here
        return df

    def process_data(self, df_list):
        """
        Process data through all steps: parse, merge, downsample, and generate features.
        
        Parameters:
        - df_list: A list of pandas DataFrames to be processed.
        
        Returns:
        - A pandas DataFrame ready for machine learning models.
        """
        df_merged_sorted = self.parse_and_merge(df_list)
        df_downsampled = self.downsample(df_merged_sorted)
        df_final = self.generate_features(df_downsampled)
        return df_final