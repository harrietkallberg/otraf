import gc
from tqdm import tqdm
import pykoda as pk
import pandas as pd
from datetime import datetime, timedelta

class DataRetriever:
    def __init__(self, start_date=str, end_date=str, merge_static=True):
        self.start_date = start_date
        self.end_date = end_date
        self.merge_static = merge_static
        self.data = self.collect_multiple_days_for_all_routes()

    def collect_day_data(self, date='2021-09-01') -> pd.DataFrame:
        return pk.get_data_range(feed='TripUpdates', company='otraf', start_date=date, end_date=date, merge_static=True)

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Define column categories
        ints = ['direction_id', 'stop_sequence']
        floats = ['departure_delay']
        strings = ['route_id', 'route_long_name', 'route_short_name', 'trip_id', 'stop_id', 'parent_station', 'stop_name']
        datetimes = ['start_date', 'scheduled_departure_time', 'observed_departure_time']

        # 1. Downcast integer columns (integers and floats)
        for col in ints:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast='unsigned')  # Downcast integer types to the smallest unsigned type
        
        for col in floats:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast='float')  # Downcast float types to the smallest float type (float32)

        # Convert object columns to category if they have a small number of unique values
        for col in strings:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

        return df

    def collect_multiple_days_for_all_routes(self) -> pd.DataFrame:
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        # Create date range
        date_list = []
        current = start
        while current <= end:
            date_list.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        columns_to_keep = [
            'route_id', 'route_long_name', 'route_short_name', 'trip_id', 'start_date', 
            'direction_id', 'stop_id', 'stop_name', 'stop_sequence', 'scheduled_departure_time', 
            'observed_departure_time', 'departure_delay', 'parent_station'
        ]

        # Process data for each day
        chunk = {}
        for day in tqdm(date_list, desc="Processing days"):
            print(f"Processing data for {day}...")
            day_data = self.collect_day_data(day)
            if not day_data.empty:
                # Drop unnecessary columns
                keep_cols = [col for col in columns_to_keep if col in day_data.columns]
                day_data = day_data[keep_cols].copy()

                # Add a date column for reference
                day_data['start_date'] = day
                day_data['route_long_name'] = day_data['route_id'].str[8:11]

                # Optimize memory usage
                optimized_data = self.optimize_dataframe(day_data)
                chunk[day] = optimized_data
                del day_data  # Free memory after processing the day’s data
                gc.collect()  # Explicitly call garbage collection

        all_data = pd.concat(chunk.values(), ignore_index=True)
        print(f"Total records collected: {len(all_data)}")
        return all_data
