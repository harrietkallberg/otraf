import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import re


class StatisticalSignificanceTesterTest:
    """Tests statistical significance of schedule optimization improvements."""
    
    def __init__(self, optimizer, data_group, strategy_key, by_stop = False):
        """Initialize with a ScheduleOptimizer instance."""
        self.optimizer = optimizer
        self._group_cols = optimizer._group_cols
        self.data_group = data_group
        self.strategy_key = strategy_key
        self.by_stop = by_stop

        if self.by_stop and data_group == 'Overall':
            self.data_to_analyze = None
            
        else: 
            self.data_to_analyze = self.collect_subset()

        self.significance_results = None
        self.test_results_detailed = None
        self.visualization_data = None
    
    def collect_subset(self):
        if self.by_stop:
            return self.data_to_analyze
        # Copy the optimized data of the given strategy key
        data_to_analyze = self.optimizer.optimized_data[self.strategy_key].copy()
        data_to_analyze = data_to_analyze.dropna(subset=['departure_delay', 'new_departure_delay'])
        # After getting the analysis_data, add this block to drop columns with mean, median, or percentile
        if data_to_analyze is not None:
        # Drop columns with mean, median, or percentile in the name
            columns_to_drop = [col for col in data_to_analyze.columns if any(term in col for term in ['mean', 'median', 'percentile_','std', 'count'])]
            data_to_analyze = data_to_analyze.drop(columns=columns_to_drop, errors='ignore')
             # Default values
            column_name = ""
            value = ""
            # Process special group format (e.g., "month*1")
            if '_' in self.data_group:
                parts = self.data_group.split('_', 1)
                if len(parts) == 2:
                    column_name = parts[0]
                    if column_name == 'day' or column_name == 'time':
                        column_name = column_name + '_type'
                    value = parts[1]
                    
                    # Try to convert value to appropriate type if needed
                    try:
                        # If the value is numeric, convert it
                        if value.isdigit():
                            value = int(value)
                        elif value.replace('.', '', 1).isdigit():
                            value = float(value)
                    except:
                        pass
                        
                    # Filter data based on the split group
                    if column_name in data_to_analyze.columns:
                        data_to_analyze = data_to_analyze[data_to_analyze[column_name] == value]
                    else:
                        print(f"Column {column_name} not found in data")
            #data_to_analyze.to_csv(f'Optimized_{data_group}_{column_name}_{value}_{strategy_key}.csv')
            return data_to_analyze
    
    def evaluate_best_strategy_by_stop(self) -> pd.DataFrame:
        """
        For the given optimizer (which has already run find_new_departure_times
        under best_strategy_key), compute On-Time % improvement per stop_id.
        Returns a DataFrame with one row per stop_id and the percent-improvement.
        """
        # 1) make sure we have the optimized data for that strategy
        if self.strategy_key not in self.optimizer.optimized_data:
            raise KeyError(f"{self.strategy_key} not found in optimized_data")
        opt_df = self.optimizer.optimized_data[self.strategy_key].copy()

        # 2) drop any rows missing delays
        opt_df = opt_df.dropna(subset=['departure_delay','new_departure_delay'])

        # 3) Collect unique combinations of stop_id, stop_name, and direction
        stop_combinations = opt_df[['stop_id', 'stop_name', 'direction_id']].drop_duplicates().values.tolist()

        # 4) For each unique combination, compute summary
        combined_results = []

        for stop_id, stop_name, direction_id in stop_combinations:
            # Get subset for this specific combination
            sub = opt_df[(opt_df['stop_id']==stop_id) & 
                        (opt_df['stop_name']==stop_name) & (opt_df['direction_id']==direction_id)]
            
            # CHANGE: First run significance tests to get CI data
            self.data_to_analyze = sub
            self.data_group = f'{stop_name}_{direction_id}'
            sign_long_df = self.run_ontime_performance_tests(opt_input_data=sub)
            
            # Store significance data in a dictionary for easy lookup
            significance_data = {}
            for _, row in sign_long_df.iterrows():
                significance_data[row['metric']] = row['value']
            
            # Now get summary metrics
            summary = self.optimizer._calculate_summary_for_data(sub, stop_name)
            summary_df = pd.DataFrame(summary, 
                columns=['group', 'metric', 'original', 'optimized', 'absolute_improvement', 'percent_improvement'])
            
            summary_df['direction_id'] = direction_id
            
            # CHANGE: Modify the summary dataframe to include confidence intervals
            for idx, row in summary_df.iterrows():
                row_dict = row.to_dict()
                combined_results.append(row_dict)
            
            # Add the significance metrics
            for _, sl in sign_long_df.iterrows():
                combined_results.append(sl.to_dict())
            
            # 3) Produce & save the figures
            self.visualize_ontime_performance(
                save_file=f"sign_per_stop/{stop_name}_{direction_id}_Overall_{self.strategy_key}.png"
            )
        
        # at the end, dump to CSV
        combined_df = pd.DataFrame(combined_results)
        
        # extract "base_group" and "dir" from any group ending in _number
        extracted = combined_df['group'].str.extract(r'^(.+?)_(\d+(?:\.\d+)?)$')

        # where there was a match, put the base name back into group
        mask = extracted[1].notna()
        combined_df.loc[mask, 'group'] = extracted.loc[mask, 0]

        # and write that suffix into direction_id (converting to float if you like)
        combined_df.loc[mask, 'direction_id'] = extracted.loc[mask, 1].astype(float)

        # now save
        combined_df.to_csv('sign_per_stop/combined_results.csv', index=False)

        return combined_df


    def calculate_summary_for_subset(self):
        summary = self.optimizer._calculate_summary_for_data(self.data_to_analyze, self.data_group)
        # Create summary DataFrame
        summary_df = pd.DataFrame(
            summary, 
            columns=['group', 'metric', 'original', 'optimized', 'absolute_improvement', 'percent_improvement']
        )
        return summary_df

    def run_ontime_performance_tests(self, opt_input_data = None,
                                early_threshold=-30, 
                                late_threshold=179,
                                original_delay_col='departure_delay', 
                                optimized_delay_col='new_departure_delay'):
        """
        Run z-tests for on-time performance improvements comparing three categories:
        too early, on-time, and too late.

        Returns a long-form DataFrame with columns:
        - group
        - strategy
        - metric           (e.g. "too_early_z_statistic", "ontime_p_value", "summary_any_significant")
        - value            (numeric or boolean)
        """
        # 1) collect the data subset
        if not self.by_stop:
            data = self.collect_subset()
        else:
            data = opt_input_data

        data = data.copy()

        if original_delay_col not in data.columns or optimized_delay_col not in data.columns:
            print(f"Missing delay columns: {original_delay_col} or {optimized_delay_col}")
            return pd.DataFrame(columns=['group','strategy','metric','value'])

        # 2) label each row as too_early, ontime, too_late for original & optimized
        data['original_too_early']   = data[original_delay_col] <= early_threshold
        data['optimized_too_early']  = data[optimized_delay_col] <= early_threshold
        data['original_ontime']      = (data[original_delay_col] > early_threshold) & (data[original_delay_col] < late_threshold)
        data['optimized_ontime']     = (data[optimized_delay_col] > early_threshold) & (data[optimized_delay_col] < late_threshold)
        data['original_too_late']    = data[original_delay_col] >= late_threshold
        data['optimized_too_late']   = data[optimized_delay_col] >= late_threshold

        # 3) compute proportions
        n = len(data)
        proportions = {
            cat: {
                'original': data[f'original_{cat}'].mean(),
                'optimized': data[f'optimized_{cat}'].mean()
            }
            for cat in ['too_early','ontime','too_late']
        }

        # 4) run z-tests and build a results dict
        results = {}
        for cat in ['too_early','ontime','too_late']:
            # 1) pull raw proportions
            p_opt = proportions[cat]['optimized']
            p_orig = proportions[cat]['original']
            
            # 2) compute improvement and its CI 
            diff = p_opt - p_orig
            
            # 3) percent improvement relative to original
            percent_diff = diff / p_orig * 100 if p_orig else float('inf')
            
            pooled_p = (data[f'original_{cat}'].sum() + data[f'optimized_{cat}'].sum()) / (2 * n)
            se_diff = np.sqrt(pooled_p * (1 - pooled_p) * (2 / n))
            
            # 4) Define z-statistic with appropriate sign based on category
            # For "ontime", positive difference is good
            # For "too_early" and "too_late", negative difference is good
            if cat == 'ontime':
                # Keep original calculation (positive z = improvement)
                z_stat = diff / se_diff if se_diff else float('inf')
            else:
                # Flip sign for too early/too late (negative diff is improvement, so flip to make positive z = improvement)
                z_stat = -diff / se_diff if se_diff else float('-inf')
            
            # Now p-value calculation is consistent: small p-value always means significant improvement
            p_value = 1 - stats.norm.cdf(z_stat)
            
            diff_ci_low = diff - 1.96 * se_diff
            diff_ci_high = diff + 1.96 * se_diff
            
            # 5) store everything
            results[cat] = {
                'original_proportion': p_orig,
                'optimized_proportion': p_opt,
                'absolute_change': diff,
                'percent_change': percent_diff,
                'z_statistic': z_stat,  # This will now have consistent interpretation
                'p_value': p_value,     # Small p always means significant improvement
                'significant': p_value < 0.05,
                'diff_ci_lower': diff_ci_low,
                'diff_ci_upper': diff_ci_high
            }
        # 5) build summary block
        sig_cats = [c for c in results if results[c]['significant']]

        summary = {
            'ontime_change':       results['ontime']['absolute_change'],
            'ontime_percent_change': results['ontime']['percent_change'],
            'significant_categories':   ', '.join(sig_cats) or 'none',
            'all_significant':          all(results[c]['significant'] for c in results),
            'any_significant':          any(results[c]['significant'] for c in results),
            'sample_size':              n,
            'early_threshold':          early_threshold,
            'late_threshold':           late_threshold
        }
        results['summary'] = summary

        # store for visualize_ontime_performance
        self.significance_results = results

        # 6) convert into long‐form DataFrame
        rows = []
        for cat, info in results.items():
            for key, val in info.items():
                rows.append({
                    'group':    self.data_group,
                    'strategy': self.strategy_key,
                    'metric':   f"{cat}_{key}",
                    'value':    val
                })

        return pd.DataFrame(rows)

    def visualize_ontime_performance(self, figsize=(10, 6.5), save_file=None):
        """
        5-panel layout:
        Row 0: [Improvement (pp)] [Before vs. After proportions]
        Row 1: [Histogram]         [CDF]
        Row 2: [Summary text (spans 2 columns)]
        """
        if self.significance_results is None:
            print(f"No results for {self.data_group} / {self.strategy_key}")
            return None

        res      = self.significance_results
        summ     = res['summary']
        cats     = ['too_early','ontime','too_late']
        labels   = ['Too Early','On Time','Too Late']
        df       = self.collect_subset()
        orig_del = df['departure_delay'].values
        opt_del  = df['new_departure_delay'].values

        # 1) Create figure with adjusted height (slightly taller than 6 but shorter than 7)
        fig = plt.figure(figsize=(10, 6))
        
        # Add a shared title
        fig.suptitle(f"{self.data_group} — {'Best Pick' if not self.by_stop else 'Overall Best Pick'}: {self.strategy_key}",
            fontsize=12, y=0.99, va='top')

        # 2) Create GridSpec with improved spacing
        gs = fig.add_gridspec(
            nrows=3, ncols=2,
            left=0.1, right=0.9,           # controlled horizontal margins
            top=0.92, bottom=0.1,          # controlled vertical margins
            hspace=0.45,                   # increased vertical spacing between plots
            wspace=0.2,                    # horizontal spacing
            height_ratios=[0.9, 0.9, 0.4]  # slightly taller plot panels, shorter summary area
        )

        # 3) Create the 4 data visualization subplots
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[0,1])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])

        # Panel (0,0): absolute change
        raw_delta   = [res[c]['optimized_proportion'] - res[c]['original_proportion'] for c in cats]
        change_pp   = [d * 100 for d in raw_delta]
        err_pp      = [((res[c]['diff_ci_upper'] - res[c]['diff_ci_lower'])/2) * 100 for c in cats]
        x           = np.arange(len(cats))
        
        # Choose colors based on improvement direction
        good_dir = {'ontime': 1, 'too_early': -1, 'too_late': -1}
        bar_colors = [
            'seagreen' if good_dir[c] * raw_delta[i] >= 0 else 'tomato'
            for i, c in enumerate(cats)
        ]

        # Draw bars for absolute change
        bars = ax0.bar(
            x, change_pp,
            yerr=err_pp,
            capsize=5,
            alpha=0.8,
            color=bar_colors
        )
        ax0.axhline(0, color='gray', linewidth=1)
        ax0.set_xticks(x)
        ax0.set_xticklabels(labels, fontsize=12)
        ax0.set_ylabel('Change (pp)', fontsize=12)
        ax0.set_title('a) Percentage Point Change (pp) with 95% CI', fontsize=12)
        
        # Add more y-axis ticks for finer grading
        y_min, y_max = ax0.get_ylim()
        ax0.yaxis.set_major_locator(plt.MaxNLocator(10))  # Increase number of ticks

        # Panel (0,1): before vs after proportions
        orig_pct = [res[c]['original_proportion']*100 for c in cats]
        opt_pct  = [res[c]['optimized_proportion']*100 for c in cats]
        w = 0.35
        ax1.bar(x - w/2, orig_pct, w, label='Original', alpha=0.7)
        ax1.bar(x + w/2, opt_pct,  w, label='Optimized', alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=11)
        ax1.set_ylabel('Percent (%)', fontsize=11)
        ax1.set_title('b) Proportions: Before vs After', fontsize=12)
        ax1.legend(fontsize=10)
        
        # Add more y-axis ticks for finer grading
        y_min, y_max = ax1.get_ylim()
        ax1.yaxis.set_major_locator(plt.MaxNLocator(10))  # Increase number of ticks

        # Panel (1,0): histogram
        ax2.hist(orig_del, bins=50, alpha=0.5, label='Original', color='tomato')
        ax2.hist(opt_del,  bins=50, alpha=0.5, label='Optimized', color='seagreen')
        ax2.axvline(summ['early_threshold'], linestyle='--', color='blue')
        ax2.axvline(summ['late_threshold'],  linestyle='--', color='blue')
        ax2.set_title('c) Delay Distribution', fontsize=12)
        ax2.set_xlabel('Delay (s)', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_xlim(-300,700)
        ax2.legend(fontsize=10, loc='upper right')
        
        # Add more y-axis ticks for finer grading
        y_min, y_max = ax2.get_ylim()
        ax2.yaxis.set_major_locator(plt.MaxNLocator(8))  # Increase number of ticks

        # Panel (1,1): CDF
        for arr, lbl, clr in [(orig_del,'Original','tomato'), (opt_del,'Optimized','seagreen')]:
            xs = np.sort(arr)
            ys = np.arange(1, len(xs)+1) / len(xs)
            ax3.plot(xs, ys, label=lbl, color=clr)
        ax3.axvline(summ['early_threshold'], linestyle='--', color='blue')
        ax3.axvline(summ['late_threshold'],  linestyle='--', color='blue')
        ax3.fill_between([summ['early_threshold'], summ['late_threshold']],
                        0, 1, color='blue', alpha=0.1)
        ax3.set_title('d) Cumulative Distribution', fontsize=12)
        ax3.set_xlabel('Delay (s)', fontsize=11)
        ax3.set_ylabel('Cum. Prob.', fontsize=11)
        ax3.set_xlim(-300,700)
        ax3.legend(fontsize=11, loc='lower right')
        
        # Add more y-axis ticks for CDF plot
        ax3.yaxis.set_major_locator(plt.MaxNLocator(10))  # More ticks on y-axis

        # 4) Create a bottom-row axis that spans both columns for summary text
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')  # Turn off axis, we just want the space for text

        # 5) Add summary text to bottom panel with improved vertical spacing
        fontsize = 11
        
        # Adjusted y-positions to distribute text more evenly in the available space
        ax4.text(0.01, 0.97, f"Overall On-Time Change: {summ['ontime_change']*100:+.2f} pp ({summ['ontime_percent_change']:+.2f} %), Sample size: {summ['sample_size']},   * p < 0.05, ** p < 0.01", 
                va='top', fontsize=fontsize)
                
        # Table headers with better spacing
        y_headers = 0.75  # Adjusted down
        ax4.text(0.01, y_headers, "Category", va='top', fontsize=fontsize, weight='bold')
        ax4.text(0.34, y_headers, "Change (pp)", va='top', fontsize=fontsize, weight='bold')
        ax4.text(0.60, y_headers, "Z-statistic", va='top', fontsize=fontsize, weight='bold')
        ax4.text(0.85, y_headers, "P-value", va='top', fontsize=fontsize, weight='bold')
        
        # Row data with more evenly distributed spacing
        y_positions = [0.50, 0.255, 0.01]  # More evenly spaced
        for i, (cat, lbl) in enumerate(zip(cats, labels)):
            y_pos = y_positions[i]
            
            # Get values
            delta = (res[cat]['optimized_proportion'] - res[cat]['original_proportion']) * 100
            good = (delta>=0) if cat=='ontime' else (delta<=0)
            color = 'green' if good else 'red'
            z_stat = res[cat]['z_statistic']
            p_value = res[cat]['p_value']
            # Calculate confidence interval half-width
            ci_half = ((res[cat]['diff_ci_upper'] - res[cat]['diff_ci_lower'])/2) * 100
            # Add significance markers
            sig_marker = "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            # Format change value with confidence interval and marker
            change_text = f"{delta:+.2f} ± {ci_half:.2f}{sig_marker}"
            # Add table row
            ax4.text(0.01, y_pos, lbl, va='top', fontsize=fontsize)
            ax4.text(0.34, y_pos, change_text, va='top', fontsize=fontsize, color=color, weight='bold')

            # Format z-stat-value with scientific notation for very small values
            if abs(z_stat) < 0.0001:
                z_stat_text = f"{z_stat:.2e}"  # Scientific notation with 2 decimal places
            else:
                z_stat_text = f"{z_stat:.4f}"  # Regular format with 4 decimal places

            ax4.text(0.60, y_pos, z_stat_text, va='top', fontsize=fontsize)

            # Format p-value with scientific notation for very small values
            if p_value < 0.0001:
                if p_value == 0:
                    # For literally zero p-values (computational precision limit)
                    p_value_text = "< 1e-16"
                else:
                    # For very small non-zero p-values, use scientific notation
                    # With a custom format that preserves significant digits
                    magnitude = int(np.floor(np.log10(p_value)))
                    mantissa = p_value / (10 ** magnitude)
                    p_value_text = f"{mantissa:.1f}e{magnitude}"
            else:
                p_value_text = f"{p_value:.4f}"  # Regular format with 4 decimal places

            ax4.text(0.85, y_pos, p_value_text, va='top', fontsize=fontsize)
        
        # Fine-tune the spacing
        fig.tight_layout(rect=[0, 0, 1, 0.94])  # Adjust rect to make room for suptitle
        
        if save_file:
            fig.savefig(save_file, dpi=300, bbox_inches='tight')
        self.visualization_data = fig
        return fig