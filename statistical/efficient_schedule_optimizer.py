import pandas as pd
import numpy as np
import gc
from typing import Dict, Optional, Tuple, Any


class ScheduleOptimizer:
    """Memory-efficient transit schedule optimizer."""
    
    def __init__(self, form_data: pd.DataFrame, analyzer: Any):
        """Initialize with form data and analyzer results."""
        # Store references without copying
        self._form_data_ref = form_data
        self._group_cols = analyzer.group_cols
        self._route_analysis_ref = analyzer._route_analysis

        # Extract the clustering of the analysis
        self.month_type_list = [col for col in analyzer._route_analysis if col.startswith('month_type_')]
        self.month_type_label = self.month_type_list[0]

        # Optimize data types
        self._optimize_dtypes(self._form_data_ref)
        self._optimize_dtypes(self._route_analysis_ref)
        
        # Storage for results and enriched data
        self._enriched_data = None
        self.optimized_data = {}
        self.optimization_summaries = {}
        self.optimization_improvements = {}
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> None:
        """Optimize data types in-place to reduce memory usage."""
        if df is None or df.empty:
            return
            
        # Convert float64 to float32
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        
        # Convert object to category where appropriate
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < 100:
                df[col] = df[col].astype('category')
    
    def enrich_data(self) -> pd.DataFrame:
        """Create enriched data if needed and return it."""
        if self._enriched_data is None:
            # Merge analysis data onto form data
            self._enriched_data = pd.merge(
                self._form_data_ref,
                self._route_analysis_ref,
                on=self._group_cols,
                how='left'
            )
            
            # Sort by trip and sequence
            self._enriched_data.sort_values(
                ['trip_id', 'direction_id', 'start_date', 'stop_sequence'], 
                inplace=True
            )
            
            # Calculate aggregated incremental delays
            skip_list = ['std', 'count']
            for col in self._route_analysis_ref.columns:
                if col.startswith('incremental_delay') and not any(sk in col for sk in skip_list):
                    self._enriched_data[f'aggregated_{col}'] = (
                        self._enriched_data
                        .groupby(['trip_id', 'direction_id', 'start_date'])[col]
                        .transform(lambda x: x.fillna(0).cumsum())
                    )
            
            # Remove redundant columns
            drop_cols = [col for col in self._enriched_data.columns if col.startswith("incremental_delay_")]
            self._enriched_data.drop(columns=drop_cols, inplace=True, errors='ignore')
            
            # Optimize memory usage
            self._optimize_dtypes(self._enriched_data)
        
        return self._enriched_data
    
    def find_new_departure_times(
        self, 
        delay_type: str = 'incremental_delay', 
        opt_method: str = 'percentile_80', 
        scaling_factor: float = 0.2
    ) -> pd.DataFrame:
        """Generate optimized departure times."""
        # Create unique key for this optimization
        strategy_key = f"{delay_type}_{opt_method}_sf{scaling_factor}"
        
        # Check if we already have this result
        if strategy_key in self.optimized_data:
            return self.optimized_data[strategy_key]
        
        # Get enriched data
        enriched_data = self.enrich_data()
        
        # Determine which column to use
        if delay_type == 'incremental_delay':
            adjustment_col = f"aggregated_{delay_type}_{opt_method}"
        else:
            adjustment_col = f"{delay_type}_{opt_method}"
        
        # Create optimized data
        optimized_data = enriched_data.copy()
        
        # Compute new departure times with scaling factor
        optimized_data['new_departure_time'] = (
            pd.to_datetime(optimized_data['scheduled_departure_time']) +
            pd.to_timedelta(optimized_data[adjustment_col], unit='s') * scaling_factor
        )
        
        # Calculate new delays
        optimized_data['new_departure_delay'] = (
            pd.to_datetime(optimized_data['observed_departure_time']) - 
            pd.to_datetime(optimized_data['new_departure_time'])
        ).dt.total_seconds()
        
        # Store optimized data
        self.optimized_data[strategy_key] = optimized_data

        # Calculate and store summary
        summary = self._calculate_summary(
            optimized_data, delay_type, opt_method, scaling_factor
        )
        self.optimization_summaries[strategy_key] = summary
        
        # Extract on-time improvement
        on_time_row = summary[summary['metric'] == 'On-Time %']
        if not on_time_row.empty:
            self.optimization_improvements[strategy_key] = on_time_row['percent_improvement'].values[0]
        else:
            self.optimization_improvements[strategy_key] = 0.0
        
        return optimized_data
    
    def _calculate_summary(
        self, 
        df: pd.DataFrame, 
        delay_type: str, 
        opt_method: str, 
        scaling_factor: float
    ) -> pd.DataFrame:
        """Calculate optimization summary statistics with group breakdowns."""
        # Extract non-null delays
        valid_data = df.dropna(subset=['departure_delay', 'new_departure_delay'])
        
        # Create the main summary (overall statistics)
        summary_rows = self._calculate_summary_for_data(
            valid_data, 
            group_label="Overall"
        )
        
        # Add month breakdown
        if 'month' in valid_data.columns:
            for month_type in valid_data['month'].dropna().unique():
                month_data = valid_data[valid_data['month'] == month_type]
                if len(month_data) > 10:  # Only calculate if we have enough data
                    month_rows = self._calculate_summary_for_data(
                        month_data, 
                        group_label=f"month_{month_type}"
                    )
                    summary_rows.extend(month_rows)
        
        # Add day type breakdown
        if 'day_type' in valid_data.columns:
            for day_type in valid_data['day_type'].dropna().unique():
                day_data = valid_data[valid_data['day_type'] == day_type]
                if len(day_data) > 10:
                    day_rows = self._calculate_summary_for_data(
                        day_data, 
                        group_label=f"day_{day_type}"
                    )
                    summary_rows.extend(day_rows)
        
        # Add time type breakdown
        if 'time_type' in valid_data.columns:
            for time_type in valid_data['time_type'].dropna().unique():
                time_data = valid_data[valid_data['time_type'] == time_type]
                if len(time_data) > 10:
                    time_rows = self._calculate_summary_for_data(
                        time_data, 
                        group_label=f"time_{time_type}"
                    )
                    summary_rows.extend(time_rows)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(
            summary_rows, 
            columns=['group', 'metric', 'original', 'optimized', 'absolute_improvement', 'percent_improvement']
        )
        
        # Add optimization parameters as metadata
        summary_df.attrs['clust_type'] = self.month_type_label
        summary_df.attrs['delay_type'] = delay_type
        summary_df.attrs['opt_method'] = opt_method
        summary_df.attrs['scaling_factor'] = scaling_factor
        
        return summary_df

    def _calculate_summary_for_data(self, data, group_label="Overall"):
        """Calculate summary statistics for a specific data subset."""
        orig, new = data['departure_delay'], data['new_departure_delay']
        
        # Thresholds for early/late
        early, late = -30, 179
        
        # Safe calculation helpers
        smean = lambda x: x.mean() if len(x) else np.nan
        spct = lambda x, q: np.percentile(x, q) if len(x) else np.nan
        sfun = lambda x, f: f(x) if len(x) else np.nan
        spct_on = lambda x: ((x > early) & (x < late)).mean() * 100 if len(x) else np.nan
        spct_early = lambda x: (x < early).mean() * 100 if len(x) else np.nan
        spct_late = lambda x: (x > late).mean() * 100 if len(x) else np.nan
        
        # Calculate statistics
        stats_map = {
            'Mean Delay': (smean(orig), smean(new)),
            'Median Delay': (orig.median(), new.median()),
            '80th Percentile Delay': (spct(orig, 80), spct(new, 80)),
            '90th Percentile Delay': (spct(orig, 90), spct(new, 90)),
            '95th Percentile Delay': (spct(orig, 95), spct(new, 95)),
            'Std Dev Delay': (sfun(orig, lambda x: x.std()), sfun(new, lambda x: x.std())),
            'Min Delay': (orig.min(), new.min()),
            'Max Delay': (orig.max(), new.max()),
            'On-Time %': (spct_on(orig), spct_on(new)),
            'Too Early %': (spct_early(orig), spct_early(new)),
            'Too Late %': (spct_late(orig), spct_late(new)),
            'Total Trips': (len(orig), len(new))
        }
        
        # Calculate improvements
        rows = []
        for metric, (ov, nv) in stats_map.items():
            lowers_better = any(x in metric for x in ['Delay', 'Too Early', 'Too Late', 'Std Dev'])
            if pd.isna(ov) or pd.isna(nv):
                abs_imp, pct_imp = np.nan, np.nan
            else:
                abs_imp = (ov - nv) if lowers_better else (nv - ov)
                pct_imp = (abs_imp / abs(ov) * 100) if ov != 0 else (np.inf if abs_imp > 0 else -np.inf)
            rows.append([group_label, metric, ov, nv, abs_imp, pct_imp])
        
        return rows
        
    def get_optimization_summary(
        self, 
        delay_type: str = 'incremental_delay', 
        opt_method: str = 'percentile_80', 
        scaling_factor: float = 0.2
    ) -> pd.DataFrame:
        """Get optimization summary, running the optimization if needed."""
        strategy_key = f"{delay_type}_{opt_method}_sf{scaling_factor}"
        
        if strategy_key not in self.optimization_summaries:
            self.find_new_departure_times(delay_type, opt_method, scaling_factor)
            
        return self.optimization_summaries.get(strategy_key, pd.DataFrame())
        
    def clear_optimization_data(self, strategy_key=None):
        """Clear optimization data for a specific strategy or all if not specified."""
        if strategy_key is not None and strategy_key in self.optimized_data:
            del self.optimized_data[strategy_key]
        else:
            self.optimized_data.clear()
        gc.collect()
    
    def force_gc(self):
        """Force garbage collection."""
        gc.collect()
        gc.collect()  # Double collection helps with fragmentation
        
    def memory_usage(self):
        """Return current memory usage in MB."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)