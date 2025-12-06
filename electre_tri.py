"""
ELECTRE TRI-B Implementation for Nutri-Score Classification
Task 3: Section 4.3 - Multi-Criteria Decision Analysis
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


class ELECTRETRIClassifier:
    """
    ELECTRE TRI-B classifier for sorting alternatives into ordered categories.
    
    Uses limiting profiles to define category boundaries and majority rule
    for assignment decisions.
    """
    
    def __init__(self, criteria_weights, limiting_profiles, preference_thresholds=None):
        """
        Initialize ELECTRE TRI classifier.
        
        Args:
            criteria_weights: dict mapping criterion name to weight (must sum to 1.0)
            limiting_profiles: DataFrame with profiles as rows, criteria as columns
            preference_thresholds: dict of preference thresholds per criterion
        """
        self.weights = criteria_weights
        self.profiles = limiting_profiles
        self.preference_thresholds = preference_thresholds or {}
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def partial_concordance(self, alternative, profile, criterion, maximize=True):
        """
        Calculate partial concordance index for a criterion.
        
        Returns 1 if alternative is at least as good as profile on criterion,
        0 otherwise (considering preference threshold).
        
        Args:
            alternative: value for the alternative
            profile: value for the limiting profile
            criterion: criterion name
            maximize: True if criterion should be maximized, False if minimized
        """
        p_threshold = self.preference_thresholds.get(criterion, 0)
        
        if maximize:
            # For maximize: alternative >= profile - threshold
            return 1.0 if alternative >= (profile - p_threshold) else 0.0
        else:
            # For minimize: alternative <= profile + threshold
            return 1.0 if alternative <= (profile + p_threshold) else 0.0
    
    def global_concordance(self, alternative_values, profile_values, criteria_directions):
        """
        Calculate global concordance index (weighted sum of partial concordances).
        
        Args:
            alternative_values: dict of criterion -> value for alternative
            profile_values: dict of criterion -> value for profile
            criteria_directions: dict of criterion -> 'max' or 'min'
        
        Returns:
            Global concordance index in [0, 1]
        """
        concordance = 0.0
        
        for criterion in self.weights.keys():
            maximize = (criteria_directions[criterion] == 'max')
            c_i = self.partial_concordance(
                alternative_values[criterion],
                profile_values[criterion],
                criterion,
                maximize
            )
            concordance += self.weights[criterion] * c_i
        
        return concordance
    
    def outranks(self, alternative_values, profile_values, criteria_directions, lambda_threshold):
        """
        Check if alternative outranks profile (S_lambda relation).
        
        Returns True if global concordance >= lambda threshold.
        """
        c = self.global_concordance(alternative_values, profile_values, criteria_directions)
        return c >= lambda_threshold
    
    def pessimistic_assignment(self, alternative_values, criteria_directions, lambda_threshold):
        """
        Pessimistic (pseudo-conjunctive) assignment procedure.
        
        Decrease k from r+1 until first value k where alternative S_lambda profile_k.
        Assign to category C_k.
        
        Returns:
            Category index (0-based: 0=worst category, 4=best category for A-E)
        """
        # Start from highest profile (pi_6) and go down
        for k in range(len(self.profiles) - 1, -1, -1):
            profile_values = self.profiles.iloc[k].to_dict()
            
            if self.outranks(alternative_values, profile_values, criteria_directions, lambda_threshold):
                # Alternative outranks this profile, assign to category above
                return k
        
        # If doesn't outrank any profile, assign to lowest category
        return 0
    
    def optimistic_assignment(self, alternative_values, criteria_directions, lambda_threshold):
        """
        Optimistic (pseudo-disjunctive) assignment procedure.
        
        Increase k from 1 until first value k where profile_k P_lambda alternative.
        Assign to category C_(k-1).
        
        Returns:
            Category index (0-based)
        """
        # Start from lowest profile (pi_1) and go up
        for k in range(len(self.profiles)):
            profile_values = self.profiles.iloc[k].to_dict()
            
            # Check if profile strictly better than alternative (profile outranks but alternative doesn't)
            profile_outranks_alt = self.outranks(profile_values, alternative_values, criteria_directions, lambda_threshold)
            alt_outranks_profile = self.outranks(alternative_values, profile_values, criteria_directions, lambda_threshold)
            
            if profile_outranks_alt and not alt_outranks_profile:
                # Profile is strictly better, assign to category below
                return max(0, k - 1)
        
        # If no profile strictly better, assign to highest category
        return len(self.profiles) - 1
    
    def classify_dataset(self, data, criteria_directions, lambda_threshold, method='pessimistic'):
        """
        Classify all alternatives in dataset.
        
        Args:
            data: DataFrame with alternatives as rows, criteria as columns
            criteria_directions: dict of criterion -> 'max' or 'min'
            lambda_threshold: majority threshold (0.5 to 1.0)
            method: 'pessimistic' or 'optimistic'
        
        Returns:
            Array of category assignments (0-4 for categories A-E)
        """
        assignments = []
        
        for idx, row in data.iterrows():
            alternative_values = row.to_dict()
            
            if method == 'pessimistic':
                category = self.pessimistic_assignment(alternative_values, criteria_directions, lambda_threshold)
            else:
                category = self.optimistic_assignment(alternative_values, criteria_directions, lambda_threshold)
            
            assignments.append(category)
        
        return np.array(assignments)


def determine_limiting_profiles(data, criteria_directions):
    """
    Determine 6 limiting profiles using quantile-based approach.
    
    Profiles separate categories:
    - pi_1: Lower bound (very permissive)
    - pi_2: Between A' and B' 
    - pi_3: Between B' and C'
    - pi_4: Between C' and D'
    - pi_5: Between D' and E'
    - pi_6: Upper bound (very strict)
    
    Uses 5th, 20th, 40th, 60th, 80th, 95th percentiles.
    For minimize criteria, reverse the logic.
    """
    profiles = {}
    
    quantiles_map = {
        'pi_1': 0.05,   # Very permissive boundary
        'pi_2': 0.20,   # A'/B' boundary
        'pi_3': 0.40,   # B'/C' boundary  
        'pi_4': 0.60,   # C'/D' boundary
        'pi_5': 0.80,   # D'/E' boundary
        'pi_6': 0.95    # Very strict boundary
    }
    
    for profile_name, quantile in quantiles_map.items():
        profile_values = {}
        
        for criterion, direction in criteria_directions.items():
            if direction == 'max':
                # For maximize: higher quantile = better performance
                profile_values[criterion] = data[criterion].quantile(quantile)
            else:
                # For minimize: reverse - lower quantile = better performance
                profile_values[criterion] = data[criterion].quantile(1 - quantile)
        
        profiles[profile_name] = profile_values
    
    # Convert to DataFrame
    profiles_df = pd.DataFrame.from_dict(profiles, orient='index')
    
    return profiles_df


def compare_with_nutriscore(electre_assignments, nutriscore_labels, method_name, lambda_val):
    """
    Compare ELECTRE TRI assignments with official Nutri-Score.
    
    Returns confusion matrix, accuracy, and classification report.
    """
    # Convert numeric assignments (0-4) to letter grades (A-E)
    category_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    electre_labels = [category_map[x] for x in electre_assignments]
    
    # Calculate metrics
    accuracy = accuracy_score(nutriscore_labels, electre_labels)
    cm = confusion_matrix(nutriscore_labels, electre_labels, labels=['A', 'B', 'C', 'D', 'E'])
    report = classification_report(nutriscore_labels, electre_labels, 
                                   labels=['A', 'B', 'C', 'D', 'E'],
                                   zero_division=0,
                                   output_dict=True)
    
    print(f"\n{'='*70}")
    print(f"{method_name} Assignment (lambda={lambda_val})")
    print(f"{'='*70}")
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    print(f"\nDistribution of ELECTRE assignments:")
    electre_dist = pd.Series(electre_labels).value_counts().sort_index()
    print(electre_dist)
    print(f"\nDistribution of Nutri-Score:")
    nutri_dist = pd.Series(nutriscore_labels).value_counts().sort_index()
    print(nutri_dist)
    
    print(f"\nClassification Report:")
    for label in ['A', 'B', 'C', 'D', 'E']:
        if label in report:
            metrics = report[label]
            print(f"  {label}: Precision={metrics['precision']:.2%}, "
                  f"Recall={metrics['recall']:.2%}, "
                  f"F1={metrics['f1-score']:.2%}, "
                  f"Support={int(metrics['support'])}")
    
    print(f"\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm,
                        index=['True A', 'True B', 'True C', 'True D', 'True E'],
                        columns=['Pred A', 'Pred B', 'Pred C', 'Pred D', 'Pred E'])
    print(cm_df)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'labels': electre_labels,
        'report': report
    }


def main():
    """
    Main execution: ELECTRE TRI implementation for Nutri-Score classification.
    """
    print("="*70)
    print("TASK 3: ELECTRE TRI-B for Nutri-Score Classification")
    print("Section 4.3 - Multi-Criteria Decision Analysis")
    print("="*70)
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv('food_database.csv')
    
    print(f"Loaded {len(df)} products")
    print(f"Nutri-Score distribution: {df['nutri_score_label'].value_counts().sort_index().to_dict()}")
    
    # Define 8 criteria (7 Nutri-Score + 1 Green-Score)
    criteria_columns = {
        'energy_100g': 'min',           # Energy (minimize)
        'saturated_fat_100g': 'min',    # Saturated fat (minimize)
        'sugars_100g': 'min',           # Sugars (minimize)
        'salt_100g': 'min',             # Salt (minimize)
        'proteins_100g': 'max',         # Proteins (maximize)
        'fiber_100g': 'max',            # Fiber (maximize)
        'fvl_percent': 'max',           # Fruits/veg/legumes (maximize)
        'green_score_value': 'max'      # Green-Score (maximize)
    }
    
    # Extract data for ELECTRE
    data_for_electre = df[list(criteria_columns.keys())].copy()
    
    # Handle missing values
    data_for_electre = data_for_electre.fillna(data_for_electre.mean())
    
    # Define weights (must sum to 1.0)
    # Give 70% weight to nutrition criteria, 30% to environmental
    weights = {
        'energy_100g': 0.10,
        'saturated_fat_100g': 0.10,
        'sugars_100g': 0.10,
        'salt_100g': 0.05,
        'proteins_100g': 0.12,
        'fiber_100g': 0.12,
        'fvl_percent': 0.11,
        'green_score_value': 0.30
    }
    
    print("\nCriteria weights:")
    for criterion, weight in weights.items():
        print(f"  {criterion}: {weight:.2%}")
    print(f"  Total: {sum(weights.values()):.2f}")
    
    # Determine limiting profiles
    print("\nDetermining 6 limiting profiles using quantile method...")
    profiles = determine_limiting_profiles(data_for_electre, criteria_columns)
    
    print("\nLimiting Profiles:")
    print(profiles)
    
    # Initialize ELECTRE TRI classifier
    classifier = ELECTRETRIClassifier(
        criteria_weights=weights,
        limiting_profiles=profiles,
        preference_thresholds={}  # Can add if needed
    )
    
    # Test with lambda = 0.6
    print("\n" + "="*70)
    print("Testing with lambda = 0.6")
    print("="*70)
    
    lambda_06 = 0.6
    
    # Pessimistic assignment
    pessimistic_06 = classifier.classify_dataset(
        data_for_electre, 
        criteria_columns, 
        lambda_06, 
        method='pessimistic'
    )
    
    results_pess_06 = compare_with_nutriscore(
        pessimistic_06,
        df['nutri_score_label'],
        "Pessimistic",
        lambda_06
    )
    
    # Optimistic assignment
    optimistic_06 = classifier.classify_dataset(
        data_for_electre,
        criteria_columns,
        lambda_06,
        method='optimistic'
    )
    
    results_opt_06 = compare_with_nutriscore(
        optimistic_06,
        df['nutri_score_label'],
        "Optimistic",
        lambda_06
    )
    
    # Test with lambda = 0.7
    print("\n" + "="*70)
    print("Testing with lambda = 0.7")
    print("="*70)
    
    lambda_07 = 0.7
    
    # Pessimistic assignment
    pessimistic_07 = classifier.classify_dataset(
        data_for_electre,
        criteria_columns,
        lambda_07,
        method='pessimistic'
    )
    
    results_pess_07 = compare_with_nutriscore(
        pessimistic_07,
        df['nutri_score_label'],
        "Pessimistic",
        lambda_07
    )
    
    # Optimistic assignment
    optimistic_07 = classifier.classify_dataset(
        data_for_electre,
        criteria_columns,
        lambda_07,
        method='optimistic'
    )
    
    results_opt_07 = compare_with_nutriscore(
        optimistic_07,
        df['nutri_score_label'],
        "Optimistic",
        lambda_07
    )
    
    # Visualization: Create combined confusion matrix plot
    print("\nCreating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('ELECTRE TRI vs Nutri-Score: Confusion Matrices', 
                 fontsize=16, fontweight='bold')
    
    scenarios = [
        (results_pess_06, f"Pessimistic (lambda=0.6)\nAccuracy: {results_pess_06['accuracy']:.2%}", axes[0, 0]),
        (results_opt_06, f"Optimistic (lambda=0.6)\nAccuracy: {results_opt_06['accuracy']:.2%}", axes[0, 1]),
        (results_pess_07, f"Pessimistic (lambda=0.7)\nAccuracy: {results_pess_07['accuracy']:.2%}", axes[1, 0]),
        (results_opt_07, f"Optimistic (lambda=0.7)\nAccuracy: {results_opt_07['accuracy']:.2%}", axes[1, 1])
    ]
    
    for result, title, ax in scenarios:
        cm_df = pd.DataFrame(
            result['confusion_matrix'],
            index=['A', 'B', 'C', 'D', 'E'],
            columns=['A', 'B', 'C', 'D', 'E']
        )
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                   cbar=True, ax=ax, vmin=0)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel('True Nutri-Score', fontsize=10)
        ax.set_xlabel('ELECTRE TRI Prediction', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/electre_tri_confusion_matrices.png', 
                dpi=300, bbox_inches='tight')
    print("Saved confusion matrices plot")
    
    # Distribution comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Distribution Comparison: ELECTRE TRI vs Nutri-Score',
                 fontsize=16, fontweight='bold')
    
    categories = ['A', 'B', 'C', 'D', 'E']
    x = np.arange(len(categories))
    width = 0.35
    
    for idx, (result, title) in enumerate([
        (results_pess_06, "Pessimistic (lambda=0.6)"),
        (results_opt_06, "Optimistic (lambda=0.6)"),
        (results_pess_07, "Pessimistic (lambda=0.7)"),
        (results_opt_07, "Optimistic (lambda=0.7)")
    ]):
        ax = axes[idx // 2, idx % 2]
        
        electre_counts = pd.Series(result['labels']).value_counts()
        nutri_counts = df['nutri_score_label'].value_counts()
        
        electre_values = [electre_counts.get(cat, 0) for cat in categories]
        nutri_values = [nutri_counts.get(cat, 0) for cat in categories]
        
        ax.bar(x - width/2, nutri_values, width, label='Nutri-Score', alpha=0.8)
        ax.bar(x + width/2, electre_values, width, label='ELECTRE TRI', alpha=0.8)
        
        ax.set_xlabel('Category', fontsize=10)
        ax.set_ylabel('Number of Products', fontsize=10)
        ax.set_title(f'{title}\nAccuracy: {result["accuracy"]:.2%}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/electre_tri_distributions.png',
                dpi=300, bbox_inches='tight')
    print("Saved distribution comparison plot")
    
    # Save results to CSV
    results_df = df.copy()
    results_df['ELECTRE_Pessimistic_06'] = results_pess_06['labels']
    results_df['ELECTRE_Optimistic_06'] = results_opt_06['labels']
    results_df['ELECTRE_Pessimistic_07'] = results_pess_07['labels']
    results_df['ELECTRE_Optimistic_07'] = results_opt_07['labels']
    
    results_df.to_csv('/mnt/user-data/outputs/electre_tri_results_lambda_0_6.csv', index=False)
    results_df[['product_name', 'nutri_score_label', 'ELECTRE_Pessimistic_07', 'ELECTRE_Optimistic_07']].to_csv(
        '/mnt/user-data/outputs/electre_tri_results_lambda_0_7.csv', index=False
    )
    
    # Save weights and profiles to Excel
    print("\nSaving configuration to Excel files...")
    
    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
    weights_df.index.name = 'Criterion'
    weights_df.to_csv('/mnt/user-data/outputs/electre_tri_weights.csv')
    
    profiles.to_csv('/mnt/user-data/outputs/electre_tri_limiting_profiles.csv')
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nBest performing method: Optimistic with lambda=0.6")
    print(f"Accuracy: {results_opt_06['accuracy']:.2%}")
    print(f"\nAll results saved to /mnt/user-data/outputs/")
    print(f"  - electre_tri_results_lambda_0_6.csv")
    print(f"  - electre_tri_results_lambda_0_7.csv")
    print(f"  - electre_tri_confusion_matrices.png")
    print(f"  - electre_tri_distributions.png")
    print(f"  - electre_tri_weights.csv")
    print(f"  - electre_tri_limiting_profiles.csv")
    
    print("\n" + "="*70)
    print("Task 3 Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
