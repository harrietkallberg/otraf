#!/usr/bin/env python
"""
Script to find continuous periods of TripUpdates and Static data in the PyKoDa cache directory.
"""

import os
import glob
import datetime
from typing import List, Tuple, Dict

# Import PyKoDa modules
import pykoda as pk

class DataAvailability:
    def __init__(self, cache_dir=None):
        """
        Initialize DataAvailability with optional cache directory
        
        :param cache_dir: Path to cache directory. If None, uses PyKoDa default.
        """
        self.cache_dir = cache_dir or pk.config.CACHE_DIR

    def _extract_date_from_filename(self, filename: str, file_type: str) -> datetime.date:
        """
        Extract date from filename based on file type
        
        :param filename: Full path or filename
        :param file_type: 'TripUpdates' or 'static'
        :return: Extracted date
        """
        base = os.path.basename(filename)
        parts = base.split('_')
        
        if file_type == 'TripUpdates':
            # Format: 'otraf_TripUpdates_2021_12_10.feather'
            year = int(parts[2])
            month = int(parts[3])
            day = int(parts[4].split('.')[0])
        elif file_type == 'static':
            # Format: 'otraf_static_2021_12_10'
            year = int(parts[-3])
            month = int(parts[-2])
            day = int(parts[-1])
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return datetime.date(year, month, day)

    def _find_dates(self, file_pattern: str, file_type: str) -> List[datetime.date]:
        """
        Find dates for a specific file pattern
        
        :param file_pattern: Glob pattern to match files
        :param file_type: 'TripUpdates' or 'static'
        :return: Sorted list of unique dates
        """
        files = glob.glob(os.path.join(self.cache_dir, file_pattern))
        
        dates = []
        for file in files:
            try:
                date = self._extract_date_from_filename(file, file_type)
                dates.append(date)
            except (IndexError, ValueError):
                # Skip files that don't match the expected format
                continue
        
        return sorted(set(dates))

    def find_continuous_periods(self, file_type: str, company: str = None) -> List[Tuple[datetime.date, datetime.date]]:
        """
        Find continuous periods for a specific file type
        
        :param file_type: 'TripUpdates' or 'static'
        :param company: Optional company filter
        :return: List of continuous date periods
        """
        # Determine file pattern based on file type and optional company
        if file_type == 'TripUpdates':
            pattern = f"{company or '*'}_TripUpdates_*.feather" if company else "*_TripUpdates_*.feather"
        elif file_type == 'static':
            pattern = f"{company or '*'}_static_*" if company else "*_static_*"
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        dates = self._find_dates(pattern, file_type)
        
        if not dates:
            return []
        
        # Find continuous periods
        periods = []
        start_date = dates[0]
        current_date = dates[0]
        
        for i in range(1, len(dates)):
            next_date = dates[i]
            
            # Check if this date is the day after the current date
            if (next_date - current_date).days == 1:
                # Continue the current period
                current_date = next_date
            else:
                # End the current period and start a new one
                periods.append((start_date, current_date))
                start_date = next_date
                current_date = next_date
        
        # Add the final period
        periods.append((start_date, current_date))
        
        return periods

    def find_common_dates(self, companies: List[str] = None) -> Dict[str, List[datetime.date]]:
        """
        Find dates with both TripUpdates and static data for specified companies
        
        :param companies: List of companies to check. If None, checks all companies.
        :return: Dictionary of common dates for each company
        """
        if companies is None:
            # Find unique companies from files
            trip_files = glob.glob(os.path.join(self.cache_dir, "*_TripUpdates_*.feather"))
            companies = set(filename.split('_')[0] for filename in trip_files)
        
        common_dates = {}
        for company in companies:
            # Find dates for this company
            trip_dates = set(self._find_dates(f"{company}_TripUpdates_*.feather", 'TripUpdates'))
            static_dates = set(self._find_dates(f"{company}_static_*", 'static'))
            
            # Find common dates
            common = sorted(list(trip_dates.intersection(static_dates)))
            common_dates[company] = common
        
        return common_dates
