"""
Statistical Analysis Module
ANOVA, Tukey HSD, and statistical significance tests
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, shapiro, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Performs statistical analysis on simulation results"""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def normality_test(self, data: List[float]) -> Dict:
        """
        Shapiro-Wilk test for normality

        H0: Data is normally distributed
        """
        if len(data) < 3:
            return {'is_normal': True, 'statistic': 0.0, 'p_value': 1.0}

        statistic, p_value = shapiro(data)
        is_normal = p_value > self.alpha

        return {
            'is_normal': is_normal,
            'statistic': statistic,
            'p_value': p_value
        }

    def homogeneity_test(self, *groups) -> Dict:
        """
        Levene's test for homogeneity of variances

        H0: Variances are equal across groups
        """
        statistic, p_value = levene(*groups)
        is_homogeneous = p_value > self.alpha

        return {
            'is_homogeneous': is_homogeneous,
            'statistic': statistic,
            'p_value': p_value
        }

    def anova_test(self, *groups) -> Dict:
        """
        One-way ANOVA test

        H0: All group means are equal
        """
        statistic, p_value = f_oneway(*groups)
        is_significant = p_value < self.alpha

        return {
            'is_significant': is_significant,
            'f_statistic': statistic,
            'p_value': p_value
        }

    def tukey_hsd(self, data: List[float], groups: List[str]) -> pd.DataFrame:
        """
        Tukey's Honestly Significant Difference test for pairwise comparisons
        """
        result = pairwise_tukeyhsd(data, groups, alpha=self.alpha)

        df = pd.DataFrame({
            'group1': [r[0] for r in result.summary().data[1:]],
            'group2': [r[1] for r in result.summary().data[1:]],
            'meandiff': [r[2] for r in result.summary().data[1:]],
            'p_value': [r[3] for r in result.summary().data[1:]],
            'reject': [r[4] for r in result.summary().data[1:]]
        })

        return df

    def t_test(self, sample1: List[float], sample2: List[float]) -> Dict:
        """
        Independent two-sample t-test
        """
        statistic, p_value = stats.ttest_ind(sample1, sample2)
        is_significant = p_value < self.alpha

        return {
            'is_significant': is_significant,
            't_statistic': statistic,
            'p_value': p_value
        }

    def confidence_interval(self, data: List[float]) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for data
        """
        mean = np.mean(data)
        se = stats.sem(data)
        ci = se * stats.t.ppf((1 + self.confidence_level) / 2, len(data) - 1)

        return mean, mean - ci, mean + ci

    def compare_methods(self, results: Dict) -> Dict:
        """
        Complete statistical comparison of all methods
        """
        comparisons = {}

        # Extract delay data for each method
        method_names = list(results.keys())
        method_delays = [results[m]['avg_delay']['all'] for m in method_names]

        # ANOVA test
        anova_result = self.anova_test(*method_delays)
        comparisons['anova'] = anova_result

        # If ANOVA significant, perform Tukey HSD
        if anova_result['is_significant']:
            # Prepare data for Tukey
            all_delays = []
            all_labels = []
            for name, delays in zip(method_names, method_delays):
                all_delays.extend(delays)
                all_labels.extend([name] * len(delays))

            tukey_result = self.tukey_hsd(all_delays, all_labels)
            comparisons['tukey_hsd'] = tukey_result

        # Pairwise t-tests with Game-Theoretic
        gt_name = 'game_theoretic'
        if gt_name in results:
            for name in method_names:
                if name != gt_name:
                    t_test_result = self.t_test(
                        results[gt_name]['avg_delay']['all'],
                        results[name]['avg_delay']['all']
                    )
                    comparisons[f'{gt_name}_vs_{name}'] = t_test_result

        return comparisons

    def generate_statistical_report(self, comparisons: Dict) -> str:
        """Generate a formatted statistical report"""
        report = []
        report.append("=" * 70)
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 70)

        # ANOVA results
        if 'anova' in comparisons:
            anova = comparisons['anova']
            report.append("\n📊 ONE-WAY ANOVA:")
            report.append(f"   F-statistic: {anova['f_statistic']:.4f}")
            report.append(f"   P-value: {anova['p_value']:.6f}")
            report.append(f"   Significant: {anova['is_significant']}")

        # Tukey HSD results
        if 'tukey_hsd' in comparisons:
            report.append("\n🔍 TUKEY HSD POST-HOC TEST:")
            df = comparisons['tukey_hsd']
            report.append(df.to_string(index=False))

        # Pairwise comparisons
        report.append("\n⚖️ PAIRWISE T-TESTS (vs Game-Theoretic):")
        for key, result in comparisons.items():
            if key.startswith('game_theoretic_vs_'):
                method = key.replace('game_theoretic_vs_', '')
                report.append(f"\n   {method}:")
                report.append(f"      t-statistic: {result['t_statistic']:.4f}")
                report.append(f"      p-value: {result['p_value']:.6f}")
                report.append(f"      Significant: {result['is_significant']}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)