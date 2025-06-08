import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any


class StatisticalSignificanceTester:
    """Tests statistical significance of schedule optimization improvements."""
    
    def __init__(self, optimizer):
        """Initialize with a ScheduleOptimizer instance."""
        self.optimizer = optimizer
        self.significance_results = {}
        self.test_results_detailed = {}
        self.visualization_data = {}
    
    def run_significance_tests_by_best_strategy_per_group(
        self,
        best_strategies_df: pd.DataFrame,
        alpha: float = 0.05,
        test_types: List[str] = ['t_test', 'wilcoxon', 'ks_test', 'bootstrap']
    ) -> pd.DataFrame:
        """
        Run statistical significance tests for each group using its best strategy.

        Parameters:
        - best_strategies_df: DataFrame where each row has 'group', 'delay_type', 'method', and 'scaling_factor'.
        - alpha: Significance level (default 0.05)
        - test_types: List of statistical tests to run

        Returns:
        - DataFrame of significance test results for each group
        """
        all_results = []

        for _, row in best_strategies_df.iterrows():
            group = row['group']
            delay_type = row['delay_type']
            opt_method = row['method']
            scaling_factor = row['scaling_factor']

            print(f"Running significance test for group: {group} | method: {opt_method} | scale: {scaling_factor}")

            # Run test for this strategy
            result_df = self.run_significance_tests(
                delay_type=delay_type,
                opt_method=opt_method,
                scaling_factor=scaling_factor,
                alpha=alpha,
                test_types=test_types
            )

            # Keep only the row that matches the current group
            result_row = result_df[result_df['group'] == group]
            if not result_row.empty:
                # Add strategy metadata
                result_row['delay_type'] = delay_type
                result_row['opt_method'] = opt_method
                result_row['scaling_factor'] = scaling_factor
                result_row['strategy'] = f"{delay_type}_{opt_method}_sf{scaling_factor}"
                all_results.append(result_row)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    def run_significance_tests_by_stop(
        self,
        delay_type: str = 'incremental_delay',
        opt_method: str = 'percentile_80',
        scaling_factor: float = 0.2,
        alpha: float = 0.05,
        test_types: List[str] = ['t_test', 'wilcoxon', 'ks_test', 'bootstrap'],
        stop_col: str = 'stop_id',
        group_name: Optional[str] = None
    ) -> pd.DataFrame:
        strategy_key = f"{delay_type}_{opt_method}_sf{scaling_factor}"

        if strategy_key not in self.optimizer.optimized_data:
            optimized_data = self.optimizer.find_new_departure_times(
                delay_type=delay_type,
                opt_method=opt_method,
                scaling_factor=scaling_factor
            )
        else:
            optimized_data = self.optimizer.optimized_data[strategy_key]

        if stop_col not in optimized_data.columns:
            raise ValueError(f"'{stop_col}' column not found in data.")

        valid_data = optimized_data.dropna(subset=['departure_delay', 'new_departure_delay'])

        early, late = -30, 179
        results = []

        for stop_id, stop_df in valid_data.groupby(stop_col):
            if len(stop_df) < 10:
                continue

            original = stop_df['departure_delay'].values
            optimized = stop_df['new_departure_delay'].values

            orig_flags = ((original > early) & (original < late)).astype(int)
            opt_flags = ((optimized > early) & (optimized < late)).astype(int)

            on_time_orig = orig_flags.sum()
            on_time_opt = opt_flags.sum()
            on_time_improvement = on_time_opt - on_time_orig
            sample_size = len(stop_df)

            # % improvement
            on_time_improvement_pct = on_time_improvement / sample_size

            # CI: absolute
            var_diff = orig_flags.var(ddof=1) + opt_flags.var(ddof=1)
            se = (var_diff / sample_size) ** 0.5
            h_abs = stats.t.ppf((1 + alpha) / 2, sample_size - 1) * se

            # CI: percentage
            se_pct = se / sample_size
            h_pct = stats.t.ppf((1 + alpha) / 2, sample_size - 1) * se_pct

            stop_result = {
                'group': group_name,
                'stop_id': stop_id,
                'sample_size': sample_size,
                'on_time_original_count': on_time_orig,
                'on_time_optimized_count': on_time_opt,
                'on_time_improvement_count': on_time_improvement,
                'on_time_improvement_pct': on_time_improvement_pct * 100,  # %
                'ci_lower_abs': on_time_improvement - h_abs,
                'ci_upper_abs': on_time_improvement + h_abs,
                'ci_lower_pct': (on_time_improvement_pct - h_pct) * 100,
                'ci_upper_pct': (on_time_improvement_pct + h_pct) * 100,
                'confidence_level': 1 - alpha
            }

            for test_type in test_types:
                test_result = self._run_statistical_test(original, optimized, test_type, alpha)
                stop_result[f'{test_type}_pvalue'] = test_result.get('p_value', np.nan)
                stop_result[f'{test_type}_significant'] = test_result.get('significant', False)

            results.append(stop_result)

        df = pd.DataFrame(results)
        self.significance_results[f'{strategy_key}_by_stop'] = df
        return df


    def _test_data_subset(
        self, 
        data: pd.DataFrame, 
        group_label: str,
        alpha: float = 0.05,
        test_types: List[str] = ['t_test', 'wilcoxon', 'ks_test', 'bootstrap']
    ) -> Dict:
        """Run statistical tests on a subset of data."""
        original_delays = data['departure_delay'].values
        optimized_delays = data['new_departure_delay'].values
        
        # Calculate basic metrics
        mean_diff = np.mean(original_delays) - np.mean(optimized_delays)
        median_diff = np.median(original_delays) - np.median(optimized_delays)
        std_diff = np.std(original_delays) - np.std(optimized_delays)
        
        # Initialize results
        results = {
            'group': group_label,
            'sample_size': len(data),
            'mean_improvement': mean_diff,
            'median_improvement': median_diff,
            'std_reduction': std_diff,
        }
        
        # Detailed results for later reference
        detailed = {
            'original_delays': original_delays,
            'optimized_delays': optimized_delays,
            'basic_stats': results.copy()
        }
        
        # Run the requested statistical tests
        for test_type in test_types:
            test_result = self._run_statistical_test(
                original_delays, 
                optimized_delays, 
                test_type,
                alpha
            )
            results.update({
                f'{test_type}_pvalue': test_result.get('p_value', np.nan),
                f'{test_type}_significant': test_result.get('significant', False)
            })
            detailed[test_type] = test_result
        
        # Store detailed results
        if group_label not in self.test_results_detailed:
            self.test_results_detailed[group_label] = {}
        self.test_results_detailed[group_label] = detailed
        
        return results
    
    def _run_statistical_test(
        self, 
        original: np.ndarray, 
        optimized: np.ndarray, 
        test_type: str,
        alpha: float = 0.05
    ) -> Dict:
        """Run a specific statistical test and return results."""
        result = {
            'test_type': test_type,
            'sample_size': len(original)
        }
        
        try:
            if test_type == 't_test':
                # Paired t-test (assumes normal distribution)
                t_stat, p_value = stats.ttest_rel(original, optimized)
                result.update({
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'test_details': 'Paired t-test comparing means'
                })
                
            elif test_type == 'wilcoxon':
                # Wilcoxon signed-rank test (non-parametric, doesn't assume normal distribution)
                stat, p_value = stats.wilcoxon(original, optimized)
                result.update({
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'test_details': 'Wilcoxon signed-rank test'
                })
                
            elif test_type == 'ks_test':
                # Kolmogorov-Smirnov test (tests if distributions are different)
                stat, p_value = stats.ks_2samp(original, optimized)
                result.update({
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'test_details': 'Kolmogorov-Smirnov test comparing distributions'
                })
                
            elif test_type == 'bootstrap':
                # Bootstrap test of mean difference
                n_bootstrap = 1000
                bootstrap_diffs = []
                
                for _ in range(n_bootstrap):
                    # Resample with replacement from the paired differences
                    diffs = original - optimized
                    resampled_diffs = np.random.choice(diffs, size=len(diffs), replace=True)
                    bootstrap_diffs.append(np.mean(resampled_diffs))
                
                # Calculate p-value from bootstrap distribution
                observed_diff = np.mean(original - optimized)
                p_value = np.mean(bootstrap_diffs <= 0) if observed_diff > 0 else np.mean(bootstrap_diffs >= 0)
                
                result.update({
                    'statistic': observed_diff,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'test_details': f'Bootstrap test with {n_bootstrap} resamples',
                    'bootstrap_distribution': bootstrap_diffs
                })
                
            else:
                result.update({
                    'error': f'Unknown test type: {test_type}'
                })
                
        except Exception as e:
            result.update({
                'error': str(e),
                'p_value': np.nan,
                'significant': False
            })
            
        return result
    
    def visualize_delay_distributions(
        self, 
        delay_type: str = 'incremental_delay', 
        opt_method: str = 'percentile_80',
        scaling_factor: float = 0.2,
        group: str = 'Overall',
        bins: int = 50,
        xlim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Visualize the original and optimized delay distributions.
        
        Parameters:
        - delay_type, opt_method, scaling_factor: Strategy parameters
        - group: Data group to visualize ('Overall' or specific subgroup)
        - bins: Number of histogram bins
        - xlim: Optional x-axis limits as (min, max)
        - figsize: Figure size
        """
        # Ensure we have run the tests first
        strategy_key = f"{delay_type}_{opt_method}_sf{scaling_factor}"
        if group not in self.test_results_detailed:
            self.run_significance_tests(delay_type, opt_method, scaling_factor)
        
        if group not in self.test_results_detailed:
            print(f"No test results found for group '{group}'")
            return
        
        # Get the data
        detailed = self.test_results_detailed[group]
        original = detailed['original_delays']
        optimized = detailed['optimized_delays']
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Histogram comparison
        ax = axes[0, 0]
        ax.hist(original, bins=bins, alpha=0.5, label='Original', color='blue')
        ax.hist(optimized, bins=bins, alpha=0.5, label='Optimized', color='green')
        if xlim:
            ax.set_xlim(xlim)
        ax.set_title(f'Delay Distribution - {group}')
        ax.set_xlabel('Delay (seconds)')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # 2. Empirical CDF
        ax = axes[0, 1]
        x_sorted_orig = np.sort(original)
        x_sorted_opt = np.sort(optimized)
        y = np.arange(1, len(original) + 1) / len(original)
        
        ax.plot(x_sorted_orig, y, label='Original', color='blue')
        ax.plot(x_sorted_opt, y, label='Optimized', color='green')
        if xlim:
            ax.set_xlim(xlim)
        ax.set_title('Empirical CDF')
        ax.set_xlabel('Delay (seconds)')
        ax.set_ylabel('Cumulative Probability')
        ax.legend()
        
        # 3. Boxplot
        ax = axes[1, 0]
        ax.boxplot([original, optimized], labels=['Original', 'Optimized'])
        ax.set_title('Boxplot of Delays')
        ax.set_ylabel('Delay (seconds)')
        
        # 4. QQ Plot
        ax = axes[1, 1]
        
        # Handle different array lengths
        min_len = min(len(original), len(optimized))
        orig_q = np.percentile(original, np.linspace(0, 100, min_len))
        opt_q = np.percentile(optimized, np.linspace(0, 100, min_len))
        
        ax.scatter(orig_q, opt_q, alpha=0.5)
        
        # Add reference line
        min_val = min(np.min(orig_q), np.min(opt_q))
        max_val = max(np.max(orig_q), np.max(opt_q))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_title('QQ Plot (Original vs Optimized)')
        ax.set_xlabel('Original Delay Quantiles')
        ax.set_ylabel('Optimized Delay Quantiles')
        
        # Add text with test results
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        tests_text = []
        for test_type in ['t_test', 'wilcoxon', 'ks_test', 'bootstrap']:
            if test_type in detailed:
                p_val = detailed[test_type].get('p_value', np.nan)
                sig = detailed[test_type].get('significant', False)
                tests_text.append(f"{test_type}: p={p_val:.4f} {'*' if sig else ''}")
        
        stats_text = '\n'.join([
            f"Mean diff: {detailed['basic_stats']['mean_improvement']:.2f}",
            f"Median diff: {detailed['basic_stats']['median_improvement']:.2f}",
            f"St.dev reduction: {detailed['basic_stats']['std_reduction']:.2f}",
            f"Sample size: {detailed['basic_stats']['sample_size']}",
            "---",
        ] + tests_text)
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=props)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.suptitle(f"Statistical Analysis of Delay Optimization - {group}", fontsize=14)
        
        # Store reference to the figure
        if strategy_key not in self.visualization_data:
            self.visualization_data[strategy_key] = {}
        self.visualization_data[strategy_key][group] = fig
        
        return fig
    
    def create_significance_summary(self, include_all_tests=False) -> pd.DataFrame:
        """
        Create a summary of all significance test results.
        
        Parameters:
        - include_all_tests: If True, include columns for all test types
        
        Returns:
        - DataFrame with significance summary
        """
        if not self.significance_results:
            return pd.DataFrame()
        
        rows = []
        
        for strategy_key, results_df in self.significance_results.items():
            # Parse the strategy key
            strategy_parts = strategy_key.split('_')
            if len(strategy_parts) >= 3:
                delay_type = strategy_parts[0]
                opt_method = strategy_parts[1]
                scaling_factor = float(strategy_parts[2].replace('sf', ''))
            else:
                delay_type, opt_method, scaling_factor = 'unknown', 'unknown', 0.0
            
            # Process each group in the results
            for _, row in results_df.iterrows():
                group = row['group']
                
                # Basic result info
                result_row = {
                    'strategy': strategy_key,
                    'delay_type': delay_type,
                    'opt_method': opt_method,
                    'scaling_factor': scaling_factor,
                    'group': group,
                    'sample_size': row['sample_size'],
                    'mean_improvement': row['mean_improvement'],
                    'median_improvement': row['median_improvement'],
                    'std_reduction': row['std_reduction'],
                }
                
                # Get the most conservative significant test
                if include_all_tests:
                    # Include all test p-values and significance
                    for test in ['t_test', 'wilcoxon', 'ks_test', 'bootstrap']:
                        if f'{test}_pvalue' in row:
                            result_row[f'{test}_pvalue'] = row[f'{test}_pvalue']
                            result_row[f'{test}_significant'] = row[f'{test}_significant']
                
                # Determine overall significance (most conservative approach)
                test_results = []
                for test in ['t_test', 'wilcoxon', 'ks_test', 'bootstrap']:
                    if f'{test}_significant' in row:
                        test_results.append(row[f'{test}_significant'])
                
                result_row['any_test_significant'] = any(test_results) if test_results else False
                result_row['all_tests_significant'] = all(test_results) if test_results else False
                
                # Add test count info
                result_row['significant_test_count'] = sum(test_results)
                result_row['total_test_count'] = len(test_results)
                
                rows.append(result_row)
        
        # Create and return the summary DataFrame
        return pd.DataFrame(rows)
    
    def run_full_analysis(
        self,
        delay_types: List[str] = ['incremental_delay'],
        opt_methods: List[str] = ['percentile_80', 'mean'],
        scaling_factors: List[float] = [0.2, 0.5, 1.0],
        visualize_best: bool = True
    ) -> pd.DataFrame:
        """
        Run significance tests for multiple optimization strategies.
        
        Parameters:
        - delay_types: List of delay types to test
        - opt_methods: List of optimization methods to test
        - scaling_factors: List of scaling factors to test
        - visualize_best: Whether to visualize the best strategy
        
        Returns:
        - Summary DataFrame of all test results
        """
        # Run all combinations
        for delay_type in delay_types:
            for opt_method in opt_methods:
                for sf in scaling_factors:
                    print(f"Testing {delay_type}_{opt_method}_sf{sf}...")
                    self.run_significance_tests(
                        delay_type=delay_type,
                        opt_method=opt_method,
                        scaling_factor=sf
                    )
        
        # Create summary
        summary = self.create_significance_summary(include_all_tests=True)
        
        # Find the best strategy based on mean improvement
        if visualize_best and not summary.empty:
            # Group by strategy and calculate mean of mean_improvement
            strategy_performance = summary.groupby('strategy')['mean_improvement'].mean().sort_values(ascending=False)
            
            if not strategy_performance.empty:
                best_strategy = strategy_performance.index[0]
                print(f"Best strategy: {best_strategy}")
                
                # Parse the strategy
                parts = best_strategy.split('_')
                if len(parts) >= 3:
                    delay_type = parts[0]
                    opt_method = parts[1]
                    scaling_factor = float(parts[2].replace('sf', ''))
                    
                    # Visualize overall results for best strategy
                    self.visualize_delay_distributions(
                        delay_type=delay_type,
                        opt_method=opt_method,
                        scaling_factor=scaling_factor,
                        group='Overall'
                    )
        
        return summary


# Example usage:
"""
# Create the tester with your optimizer instance
tester = StatisticalSignificanceTester(optimizer)

# Run basic significance test with default parameters
results = tester.run_significance_tests()
print(results)

# Visualize the delay distributions
tester.visualize_delay_distributions()

# Run a comprehensive analysis of multiple optimization strategies
summary = tester.run_full_analysis(
    delay_types=['incremental_delay', 'departure_delay'],
    opt_methods=['mean', 'percentile_80'],
    scaling_factors=[0.2, 0.5, 0.8, 1.0]
)
print(summary)
"""