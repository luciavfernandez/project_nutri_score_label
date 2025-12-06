"""
Nutri-Score Calculation Algorithm
Based on the official French Nutri-Score methodology (March 2025)
"""

import pandas as pd
import numpy as np


class NutriScoreCalculator:
    """
    Official Nutri-Score calculator implementing the French algorithm.

    References:
    - ANSES scientific and technical support report
    - Tables 1, 2, 3 from Project documentation
    """

    # Table 1: Negative component thresholds (nutrients to limit)
    ENERGY_THRESHOLDS = [335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350]
    SATURATED_FAT_THRESHOLDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    SUGARS_THRESHOLDS = [3.4, 6.8, 10, 14, 17, 20, 24, 27, 31, 34, 37, 41, 44, 48, 51]
    SALT_THRESHOLDS = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4]

    # Table 2: Positive component thresholds (nutrients to favor)
    PROTEINS_THRESHOLDS = [2.4, 4.8, 7.2, 9.6, 12, 14, 17]
    FIBER_THRESHOLDS = [3.0, 4.1, 5.2, 6.3, 7.4]
    FRUIT_VEG_THRESHOLDS = [40, 60, 80]  # Percentage

    # Table 3: Score to class mapping
    CLASS_THRESHOLDS = {
        'A': (-17, 0),
        'B': (1, 2),
        'C': (3, 10),
        'D': (11, 18),
        'E': (19, 55)
    }

    @staticmethod
    def _get_points_from_thresholds(value, thresholds):
        """Get points based on value and threshold list."""
        if pd.isna(value):
            return 0

        points = 0
        for threshold in thresholds:
            if value > threshold:
                points += 1
            else:
                break
        return points

    @staticmethod
    def calculate_negative_points(energy_kj, saturated_fat_g, sugars_g, salt_g):
        """
        Calculate negative component N (nutrients to limit).

        Args:
            energy_kj: Energy in kJ per 100g
            saturated_fat_g: Saturated fatty acids in g per 100g
            sugars_g: Sugars in g per 100g
            salt_g: Salt in g per 100g

        Returns:
            int: Negative points (0-55)
        """
        energy_pts = NutriScoreCalculator._get_points_from_thresholds(
            energy_kj, NutriScoreCalculator.ENERGY_THRESHOLDS
        )
        sat_fat_pts = NutriScoreCalculator._get_points_from_thresholds(
            saturated_fat_g, NutriScoreCalculator.SATURATED_FAT_THRESHOLDS
        )
        sugars_pts = NutriScoreCalculator._get_points_from_thresholds(
            sugars_g, NutriScoreCalculator.SUGARS_THRESHOLDS
        )
        salt_pts = NutriScoreCalculator._get_points_from_thresholds(
            salt_g, NutriScoreCalculator.SALT_THRESHOLDS
        )

        return energy_pts + sat_fat_pts + sugars_pts + salt_pts

    @staticmethod
    def calculate_positive_points(proteins_g, fiber_g, fruit_veg_pct, negative_points):
        """
        Calculate positive component P (nutrients to favor).

        Special rule: If N >= 11 and fruit/veg <= 80%, proteins are NOT counted.

        Args:
            proteins_g: Proteins in g per 100g
            fiber_g: Fiber in g per 100g
            fruit_veg_pct: Fruits/vegetables/nuts in % per 100g
            negative_points: Already calculated negative points N

        Returns:
            int: Positive points (0-17)
        """
        # Fiber points (0-5)
        fiber_pts = NutriScoreCalculator._get_points_from_thresholds(
            fiber_g, NutriScoreCalculator.FIBER_THRESHOLDS
        )

        # Fruit/veg points (0, 1, 2, or 5)
        fruit_veg_pct = fruit_veg_pct if not pd.isna(fruit_veg_pct) else 0
        if fruit_veg_pct > 80:
            fruit_veg_pts = 5
        elif fruit_veg_pct > 60:
            fruit_veg_pts = 2
        elif fruit_veg_pct > 40:
            fruit_veg_pts = 1
        else:
            fruit_veg_pts = 0

        # Protein points (0-7) - with special rule
        # If N >= 11 and fruit/veg <= 80%, do NOT count proteins
        if negative_points >= 11 and fruit_veg_pct <= 80:
            protein_pts = 0
        else:
            protein_pts = NutriScoreCalculator._get_points_from_thresholds(
                proteins_g, NutriScoreCalculator.PROTEINS_THRESHOLDS
            )

        return protein_pts + fiber_pts + fruit_veg_pts

    @staticmethod
    def calculate_score(energy_kj, saturated_fat_g, sugars_g, salt_g,
                       proteins_g, fiber_g, fruit_veg_pct):
        """
        Calculate Nutri-Score numerical score.

        Formula: Score = N - P

        Returns:
            tuple: (score, negative_points, positive_points)
        """
        N = NutriScoreCalculator.calculate_negative_points(
            energy_kj, saturated_fat_g, sugars_g, salt_g
        )

        P = NutriScoreCalculator.calculate_positive_points(
            proteins_g, fiber_g, fruit_veg_pct, N
        )

        score = N - P

        return score, N, P

    @staticmethod
    def score_to_class(score):
        """
        Convert numerical score to letter class (A, B, C, D, E).

        Table 3:
        - A (dark green): -17 to 0
        - B (light green): 1 to 2
        - C (yellow): 3 to 10
        - D (light orange): 11 to 18
        - E (red/dark orange): 19 to 55
        """
        for class_label, (min_score, max_score) in NutriScoreCalculator.CLASS_THRESHOLDS.items():
            if min_score <= score <= max_score:
                return class_label

        # Edge cases
        if score < -17:
            return 'A'
        if score > 55:
            return 'E'

        return 'Unknown'

    @staticmethod
    def calculate_nutriscore(energy_kj, saturated_fat_g, sugars_g, salt_g,
                            proteins_g, fiber_g, fruit_veg_pct):
        """
        Calculate complete Nutri-Score (score + class).

        Returns:
            dict: {
                'score': int,
                'class': str,
                'negative_points': int,
                'positive_points': int
            }
        """
        score, N, P = NutriScoreCalculator.calculate_score(
            energy_kj, saturated_fat_g, sugars_g, salt_g,
            proteins_g, fiber_g, fruit_veg_pct
        )

        class_label = NutriScoreCalculator.score_to_class(score)

        return {
            'score': score,
            'class': class_label,
            'negative_points': N,
            'positive_points': P
        }

    @staticmethod
    def calculate_for_dataframe(df):
        """
        Calculate Nutri-Score for an entire DataFrame.

        Expected columns:
        - energy_kj
        - saturated_fat_g
        - sugars_g
        - salt_g
        - proteins_g
        - fiber_g
        - fruit_veg_pct

        Returns:
            DataFrame with added columns: computed_score, computed_class, N, P
        """
        results = df.apply(
            lambda row: NutriScoreCalculator.calculate_nutriscore(
                row.get('energy_kj', 0),
                row.get('saturated_fat_g', 0),
                row.get('sugars_g', 0),
                row.get('salt_g', 0),
                row.get('proteins_g', 0),
                row.get('fiber_g', 0),
                row.get('fruit_veg_pct', 0)
            ),
            axis=1
        )

        df_result = df.copy()
        df_result['computed_score'] = results.apply(lambda x: x['score'])
        df_result['computed_class'] = results.apply(lambda x: x['class'])
        df_result['N'] = results.apply(lambda x: x['negative_points'])
        df_result['P'] = results.apply(lambda x: x['positive_points'])

        return df_result


def test_nutriscore():
    """Test on examples from the PDF."""

    print("=" * 60)
    print("TESTING NUTRI-SCORE CALCULATOR")
    print("=" * 60)

    # Example 1: Gerble-Sesame Cookie (from PDF Figure 2)
    # Expected: N=11, P=2, Score=9, Class=C
    print("\n1. Gerble-Sesame Cookie 230g")
    print("-" * 60)
    result = NutriScoreCalculator.calculate_nutriscore(
        energy_kj=1961,
        saturated_fat_g=2,
        sugars_g=17,
        salt_g=0.38,
        proteins_g=0,  # Not counted because N>=11 and fruit/veg<=80%
        fiber_g=4.6,
        fruit_veg_pct=0
    )
    print(f"Negative points (N): {result['negative_points']} (expected: 11)")
    print(f"Positive points (P): {result['positive_points']} (expected: 2)")
    print(f"Score: {result['score']} (expected: 9)")
    print(f"Class: {result['class']} (expected: C)")

    # Example 2: Pain de mie grandes tranches Seigle (from PDF Figure 3)
    # Expected: N=8, P=8, Score=0, Class=A
    print("\n2. Pain de mie grandes tranches Seigle & Graines")
    print("-" * 60)
    result = NutriScoreCalculator.calculate_nutriscore(
        energy_kj=1246,
        saturated_fat_g=1,
        sugars_g=4.2,
        salt_g=0.96,
        proteins_g=12,
        fiber_g=7.2,
        fruit_veg_pct=0.5
    )
    print(f"Negative points (N): {result['negative_points']} (expected: 8)")
    print(f"Positive points (P): {result['positive_points']} (expected: 8)")
    print(f"Score: {result['score']} (expected: 0)")
    print(f"Class: {result['class']} (expected: A)")

    print("\n" + "=" * 60)
    print("✓ Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_nutriscore()
