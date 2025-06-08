import pykoda as pk
import pandas as pd
from datetime import timedelta, datetime

class DataRetriever:
    def __init__(self, start_date = str, end_date = str, merge_static = True):
        self.start_date = start_date
        self.end_date = end_date
        self.merge_static = merge_static
        self.data = self.collect_multiple_days_for_all_routes()

    def collect_day_data(self, date='2021-09-01') -> pd.DataFrame:
        return pk.get_data_range(feed='TripUpdates', company='otraf', start_date=date, end_date=date, merge_static=True)
    
    def collect_multiple_days_for_all_routes(self) -> pd.DataFrame:
        # Convert string dates to datetime
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        # Create date range
        date_list = []
        current = start
        while current <= end:
            date_list.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        # Collect and process data for each day
        all_data = []
        for day in date_list:
            print(f"Processing data for {day}...")
            day_data = self.collect_day_data(day)
            # Add a date column for reference
            day_data['date'] = day
            
            all_data.append(day_data)
        
        # Combine all days' data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Total records collected: {len(combined_data)}")
        
        return combined_data
