import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


class DataAnalyzer:
    """Analyze transit data for delay patterns with hierarchical output options."""

    def __init__(self, form_data = None, group_cols=None):
        self.form_data = form_data
        self.group_cols = group_cols
        self._route_analysis = None
        self._has_stops = False

    def do_route_analysis(self, percentiles=[20, 60, 80]):
        df = self.form_data.copy()

        # Build your base aggregations
        base_aggs = ['mean', 'median', 'std', 'count']

        # Build a list of (alias, function) tuples for your custom percentiles
        percentile_aggs = []
        for p in percentiles:
            # Create a function capturing this percentile
            def percentile_func(x, percentile=p):
                valid = x.dropna()
                return np.percentile(valid, percentile) if len(valid) else np.nan
            
            # Give the aggregator a name so pandas can use it in the column
            percentile_aggs.append((f'percentile_{int(p)}', percentile_func))

        # Now combine them into your agg_dict
        agg_dict = {
            'departure_delay': base_aggs + percentile_aggs,
            'incremental_delay': base_aggs + percentile_aggs
        }

        # Group and aggregate
        analysis = df.groupby(self.group_cols, observed=True).agg(agg_dict).reset_index()
        # Flatten MultiIndex columns
        renamed_columns = []
        for col in analysis.columns:
            if isinstance(col, tuple):
                # e.g. ('departure_delay', 'percentile_20') => 'departure_delay_percentile_20'
                renamed_columns.append('_'.join(col).strip('_'))
            else:
                renamed_columns.append(col)
        analysis.columns = renamed_columns

        # Convert high-precision float columns to lower precision
        for col in analysis.select_dtypes(include=['float64']).columns:
            analysis[col] = analysis[col].astype('float32')
        
        # Convert object columns to categories if they have few unique values
        for col in analysis.select_dtypes(include=['object']).columns:
            if analysis[col].nunique() < 100:  # Adjust threshold as needed
                analysis[col] = analysis[col].astype('category')

        self._route_analysis = analysis
        return analysis


    def output_analysis_hierarchical(self, output_type='print', delay_type = 'incremental_delay', output_dir=None, optim = False, figsize=(10, 6), bins=20):
        """Output analysis results in hierarchical structure (text or histograms)."""
        # Ensure analysis is done
        if self._route_analysis is None:
            print("Running analysis first...")
            self.do_route_analysis()
        
        has_stops = any(col in self.group_cols for col in ['stop_id', 'stop_name'])

        if output_type == 'hist':
            if output_dir is None:
                # Extract start and end dates from formed data
                start_date = self.form_data['start_date'].dt.date.min()
                end_date = self.form_data['start_date'].dt.date.max()
                
                if has_stops:
                    prefix = 'stop_specific'
                else:
                    prefix = 'whole_route'
                
                output_dir = f"{'optim_' if optim else ''}{prefix}_{delay_type}_histograms_{start_date}_{end_date}"
            
                # Setup for histograms
                os.makedirs(output_dir, exist_ok=True)
        
        # Define metrics and check for stops
        metric_prefixes = ['departure_delay_', 'incremental_delay_']
        metrics = ['departure_delay_mean', 'departure_delay_median', 'departure_delay_percentile_80', 'departure_delay_percentile_20'
                'incremental_delay_mean', 'incremental_delay_median', 'incremental_delay_percentile_80', 'incremental_delay_percentile_20'
                'departure_delay_count']
        
        # Print analysis header if needed
        if output_type == 'print':
            print("\n===== ROUTE DELAY ANALYSIS =====")
            if not has_stops:
                print("\nNOTE: Analysis is time-dependent without stop-specific grouping.")
        
       # Sort orders for hierarchical grouping
        month_order = {
            0: 0, 
            1: 1,
            2: 2,
            3: 3
        }

        day_order = {
            'weekday': 0, 
            'saturday': 1, 
            'sunday': 2,
        }

        time_order = {
            'night': 0, 
            'am_rush': 1, 
            'day': 2, 
            'pm_rush': 3, 
            'weekend': 4,
            }
        
        # Process by month type
        df = self._route_analysis.copy()
        month_types = ['month_type_inc', 'month_type_dep', 'month_type_ord']
        used_month_type = next((col for col in month_types if col in self.group_cols), None)
        has_month = used_month_type is not None
        has_day = 'day_type' in df.columns
        has_time = 'time_type' in df.columns
        
        for month in sorted(df[used_month_type].unique(), key=lambda x: month_order.get(x, 99)) if has_month else [None]:
            month_df = df[df[used_month_type] == month] if has_month else df
            
            # Create month directory if needed
            month_dir = output_dir
            if output_type == 'hist' and has_month:
                month_dir = os.path.join(output_dir, f"month_{month}")
                os.makedirs(month_dir, exist_ok=True)
            
            # Print month header if needed
            if output_type == 'print':
                print(f"\n===== Month Type: {month} =====" if has_month else "")
            
            # Process day types within each month type
            for day in sorted(month_df['day_type'].unique(), key=lambda x: day_order.get(x, 99)) if has_day else [None]:
                day_df = month_df[month_df['day_type'] == day] if has_day else month_df
                
                # Create day directory if needed
                day_dir = month_dir
                if output_type == 'hist' and has_day:
                    day_dir = os.path.join(month_dir, f"day_{day}")
                    os.makedirs(day_dir, exist_ok=True)
                
                # Print day header if needed
                if output_type == 'print':
                    print(f"\n\t===== Day Type: {day} =====" if has_day else "")
                
                # Process time periods for weekdays
                if day == 0 and has_time:
                    for time in sorted(day_df['time_type'].unique(), key=lambda x: time_order.get(x, 99)):
                        time_df = day_df[day_df['time_type'] == time]
                        
                        # Create time directory if needed
                        time_dir = day_dir
                        if output_type == 'hist':
                            time_dir = os.path.join(day_dir, f"time_{time}")
                            os.makedirs(time_dir, exist_ok=True)
                        
                        # Print time header if needed
                        if output_type == 'print':
                            print(f"\n\t\t----- Time Type: {time} -----")
                        
                        # Process directions
                        self._process_directions(time_df, day, time, has_stops, 
                                                metric_prefixes, metrics, indent=3,
                                                output_type=output_type, delay_type=delay_type, output_dir=time_dir,
                                                figsize=figsize, bins=bins)
                else:
                    # Process directions directly for weekends or no time periods
                    self._process_directions(day_df, day, None, has_stops, 
                                            metric_prefixes, metrics, indent=2,
                                            output_type=output_type, delay_type = delay_type, output_dir=day_dir,
                                            figsize=figsize, bins=bins)
        
        self._has_stops = has_stops

    def _process_directions(self, df, day, time, has_stops, metric_prefixes, metrics, 
                       indent, output_type, delay_type, output_dir, figsize, bins):
        """Process each direction for text output or histograms."""
        indent_str = '\t' * indent
        
        for direction in sorted(df['direction_id'].unique()):
            dir_df = df[df['direction_id'] == direction].copy()
            
            # Create direction directory if needed
            dir_dir = output_dir
            if output_type == 'hist':
                try:
                    # Use a simpler directory name structure
                    dir_name = f"direction_{str(direction).replace('.', '_')}"
                    dir_dir = os.path.join(output_dir, dir_name)
                    os.makedirs(dir_dir, exist_ok=True)
                except Exception as e:
                    print(f"Error creating directory: {e}")
                    # Fallback to a simpler directory
                    dir_dir = output_dir
            
            # Print direction header if needed
            if output_type == 'print':
                print(f"\n{indent_str}----- Direction {direction} -----")
            
            # Sequence stops if needed
            if 'stop_id' in dir_df.columns and has_stops and 'stop_sequence' in self.form_data.columns:
                # Get stops for this direction
                direction_stops = self.form_data[self.form_data['direction_id'] == direction]
                
                # Apply filters
                if day and 'day_type' in self.form_data.columns:
                    direction_stops = direction_stops[direction_stops['day_type'] == day]
                if time and 'time_type' in self.form_data.columns:
                    direction_stops = direction_stops[direction_stops['time_type'] == time]
                
                # Create sequence map and sort
                seq_map = direction_stops.groupby('stop_id', observed= True)['stop_sequence'].median().to_dict()
                dir_df['_seq'] = dir_df['stop_id'].apply(lambda x: seq_map.get(x, 9999))
                dir_df = dir_df.sort_values('_seq')
            
            if output_type == 'print':
                # Print text output
                exclude = ['day_type', 'direction_id', 'time_type']
                cols = [c for c in dir_df.columns if c not in exclude and 
                    not any(c.startswith(p) for p in metric_prefixes)]
                cols += [m for m in metrics if m in dir_df.columns]
                
                table_str = dir_df[cols].to_string(index=False)
                indented_table = '\n'.join(f"{indent_str}{line}" for line in table_str.split('\n'))
                print(indented_table)
            else:
                # Create histograms
                for idx, row in dir_df.iterrows():
                    # Get filter combination
                    combo = {col: row[col] for col in self.group_cols if col in row}
                    if not combo:
                        continue
                    
                    # Filter data and get delays
                    filtered_data = self.form_data.copy()
                    for col, val in combo.items():
                        filtered_data = filtered_data[filtered_data[col] == val]
                    
                    delays = filtered_data[delay_type].dropna()
                    if len(delays) == 0:
                        continue
                    
                    # Create histogram
                    fig, ax = plt.subplots(figsize=figsize)
                    delay_max = min(delays.max(), 600)
                    delay_min = max(delays.min(), -600)
                    ax.hist(delays, bins=bins, range=(delay_min, delay_max), alpha=0.6, color='royalblue')
                    
                    # Add statistics lines
                    mean_val = delays.mean()
                    median_val = delays.median()
                    percentile_80_val = np.percentile(delays, 80)
                    percentile_20_val = np.percentile(delays, 20)

                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                            label=f'Mean: {mean_val:.1f}s')
                    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                            label=f'Median: {median_val:.1f}s')
                    ax.axvline(percentile_80_val, color='purple', linestyle='--', linewidth=2,
                            label=f'80th Percentile: {percentile_80_val:.1f}s')
                    ax.axvline(percentile_20_val, color='orange', linestyle='--', linewidth=2,
                            label=f'20th Percentile: {percentile_20_val:.1f}s')
                    
                    # Create title
                    title_parts = []
                    for col, val in combo.items():
                            title_parts.append(f"{col}: {val}")
                    
                    title = "\n".join(title_parts)
                    ax.set_title(title, fontsize=12)
                    ax.set_xlabel(f'{delay_type} (seconds)', fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                    
                    # Add sample size
                    ax.text(0.95, 0.95, f'n={len(delays)}', transform=ax.transAxes, 
                        va='top', ha='right', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
                    
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    try:
                        # Use a simpler filename - just use the index or row number
                        # This avoids issues with special characters and long paths
                        # Extract the sequence (with default/fallback of 999) and sanitize stop name:
                        stop_seq = int(row['_seq']) if '_seq' in row and not np.isnan(row['_seq']) else 999
                        stop_name_sanitized = str(row['stop_name']).replace("/", "_").replace("\\", "_")

                        # Build the filename using zero-padded sequence and sanitized stop name:
                        filename = os.path.join(dir_dir, f"{stop_seq:02d}_{stop_name_sanitized}.png")
                        plt.tight_layout()
                        plt.savefig(filename, dpi=300)
                    except Exception as e:
                        print(f"Error saving plot: {e}")
                    finally:
                        plt.close(fig)  # Make sure we close the figure even if saving fails

    def plot_histogram(self, **filter_values):
        """Plot a single histogram for a specific combination of values."""
        # Extract plotting parameters
        bins = filter_values.pop('bins', 20)
        figsize = filter_values.pop('figsize', (10, 6))
        delay_type = filter_values.pop('delay_type', 'incremental_delay')
        x_limit = filter_values.pop('x_limit', 600)

        # Ensure data is processed
        if self._route_analysis is None:
            print("Processing data first...")
            self.do_route_analysis()
        
        # Validate filters
        for key in filter_values:
            if key not in self.group_cols:
                print(f"Warning: '{key}' is not in group_cols {self.group_cols}.")
        
        # Apply filters and get delays
        filtered_data = self.form_data.copy()
        for col, val in filter_values.items():
            if col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col] == val]
        
        delays = filtered_data[delay_type].dropna()
        if len(delays) == 0:
            print("No data matches the specified criteria.")
            return None
        
        # Create histogram
        fig, ax = plt.subplots(figsize=figsize)
        delay_max = min(delays.max(), x_limit)
        delay_min = max(delays.min(), -600)
        ax.hist(delays, bins=bins, range=(delay_min, delay_max), alpha=0.6, color='royalblue')
        
        # Add statistics
        mean_val = delays.mean()
        median_val = delays.median()
        percentile_80_val = np.percentile(delays, 80)
        percentile_20_val = np.percentile(delays, 20)

        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                 label=f'Mean: {mean_val:.1f}s')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                 label=f'Median: {median_val:.1f}s')
        ax.axvline(percentile_80_val, color='purple', linestyle='--', linewidth=2,
                            label=f'80th Percentile: {percentile_80_val:.1f}s')
        ax.axvline(percentile_20_val, color='orange', linestyle='--', linewidth=2,
                            label=f'20th Percentile: {percentile_20_val:.1f}s')
        
        # Create title and styling
        title_parts = []
        for col, val in filter_values.items():
            if col == 'stop_id' and 'stop_name' in filtered_data.columns:
                stop_names = filtered_data['stop_name'].unique()
                if len(stop_names) == 1:
                    title_parts.append(f"{stop_names[0]}")
                else:
                    title_parts.append(f"Stop ID: {val}")
            else:
                title_parts.append(f"{col}: {val}")
        
        title = ", ".join(title_parts) or "All Data"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f'{delay_type} (seconds)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        
        # Add statistics text box
        stats_text = (f"Mean: {mean_val:.1f}s\n"
                     f"Median: {median_val:.1f}s\n"
                     f"Std Dev: {delays.std():.1f}s\n"
                     f"Sample size: {len(delays)}")
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
               va='top', ha='right', fontsize=10,
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_analysis_and_form_data_to_csv(self, filename_ana= f'route_analysis.csv', filename_form = 'form_data.csv'):
        """
        Save the route analysis results to a CSV file.
        
        Parameters:
        filename (str): Name of the file to save the analysis to
        
        Returns:
        str: Path to the saved file
        """
        if self._route_analysis is None:
            print("No analysis data available. Please run do_route_analysis() first.")
            return None
        
        if self.form_data is None:
            print("No processed data available. Please run do_route_analysis() first.")
            return None
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filename_ana)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            # Save to CSV
            self._route_analysis.to_csv(filename_ana, index=False)
            print(f"Analysis saved to {filename_ana}")
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(filename_form)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # Save to CSV
            self.form_data.to_csv(filename_form, index=False)
            print(f"Form data saved to {filename_form}")
            return filename_ana, filename_form
        except Exception as e:
            print(f"Error saving analysis to CSV: {str(e)}")
            return None
    
    def load_analysis_and_form_data_from_csv(self, filename_ana = 'route_analysis.csv', filename_form = 'form_data.csv'):
        """
        Load route analysis results from a CSV file.
        
        Parameters:
        filename (str): Path to the CSV file containing the analysis
        
        Returns:
        bool: True if loading was successful, False otherwise
        """
        try:
            if not os.path.exists(filename_ana):
                print(f"File not found: {filename_ana}")
                return False  
            # Load from CSV
            self._route_analysis = pd.read_csv(filename_ana)
            print(f"Analysis loaded from {filename_ana}")

            if not os.path.exists(filename_form):
                print(f"File not found: {filename_form}")
                return False   
            # Load from CSV
            self.form_data = pd.read_csv(filename_form)
            print(f"Form data loaded from {filename_form}")

            return True
        except Exception as e:
            print(f"Error loading analysis from CSV: {str(e)}")
            return False

    def plot_pie_in_time(self, stop_id=None, stop_name=None):
        """
        Plot a pie chart showing distribution of buses that are:
        - Too early (delay < -30 seconds)
        - On time (-30 <= delay <= 200 seconds)
        - Too late (delay > 200 seconds)
        
        Parameters:
        stop_id (str or list, optional): Filter by specific stop_id(s)
        stop_name (str or list, optional): Filter by specific stop_name(s)
        """
        import matplotlib.pyplot as plt
        
        # Start with all data
        filtered_data = self.form_data
        
        # Apply stop_id filter if provided
        if stop_id is not None:
            if isinstance(stop_id, list):
                filtered_data = filtered_data[filtered_data['stop_id'].isin(stop_id)]
            else:
                filtered_data = filtered_data[filtered_data['stop_id'] == stop_id]
        
        # Apply stop_name filter if provided
        if stop_name is not None:
            if isinstance(stop_name, list):
                filtered_data = filtered_data[filtered_data['stop_name'].isin(stop_name)]
            else:
                filtered_data = filtered_data[filtered_data['stop_name'] == stop_name]
        
        # Define the delay categories
        early_mask = filtered_data['departure_delay'] < -30
        late_mask = filtered_data['departure_delay'] > 200
        ontime_mask = (~early_mask) & (~late_mask)
        
        # Count records in each category
        early_count = early_mask.sum()
        late_count = late_mask.sum()
        ontime_count = ontime_mask.sum()
        
        # Calculate percentages
        total = len(filtered_data)
        early_pct = round(early_count / total * 100, 1) if total > 0 else 0
        late_pct = round(late_count / total * 100, 1) if total > 0 else 0
        ontime_pct = round(ontime_count / total * 100, 1) if total > 0 else 0
        
        # Create title based on filters
        title = 'Bus Punctuality Distribution'
        if stop_id is not None:
            if isinstance(stop_id, list):
                title += f' for Stop IDs: {", ".join(stop_id[:3])}{"..." if len(stop_id) > 3 else ""}'
            else:
                title += f' for Stop ID: {stop_id}'
        elif stop_name is not None:
            if isinstance(stop_name, list):
                title += f' for Stops: {", ".join(stop_name[:3])}{"..." if len(stop_name) > 3 else ""}'
            else:
                title += f' for Stop: {stop_name}'
        
        # Create the pie chart
        labels = [f'Too Early (<-30s): {early_pct}%',
                f'On Time: {ontime_pct}%',
                f'Too Late (>200s): {late_pct}%']
        sizes = [early_count, ontime_count, late_count]
        colors = ['#66B2FF','#99FF99', '#FF9999']
        explode = (0.1, 0, 0.1)  # Explode the 'too early' and 'too late' slices
        
        plt.figure(figsize=(10, 7))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title(title, fontsize=16)
        
        # Add actual counts to the plot
        plt.text(-1.3, -1.3, f'Total buses: {total}', fontsize=12)
        plt.text(-1.3, -1.4, f'Too early: {early_count}', fontsize=12)
        plt.text(-1.3, -1.5, f'On time: {ontime_count}', fontsize=12)
        plt.text(-1.3, -1.6, f'Too late: {late_count}', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'too_early': {'count': early_count, 'percentage': early_pct},
            'on_time': {'count': ontime_count, 'percentage': ontime_pct},
            'too_late': {'count': late_count, 'percentage': late_pct},
            'total': total
        }