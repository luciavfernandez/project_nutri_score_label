"""
Validate our Nutri-Score implementation against the friend's dataset
"""
import pandas as pd
import numpy as np
from nutriscore import NutriScoreCalculator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

print("=" * 60)
print("VALIDATING NUTRI-SCORE IMPLEMENTATION")
print("=" * 60)

# Read friend's dataset
df = pd.read_csv('project_nutri_score_label/food_database.csv')

# Standardize column names to match our code
df_clean = df.rename(columns={
    'energy_100g': 'energy_kj',
    'saturated_fat_100g': 'saturated_fat_g',
    'sugars_100g': 'sugars_g',
    'salt_100g': 'salt_g',
    'proteins_100g': 'proteins_g',
    'fiber_100g': 'fiber_g',
    'fvl_percent': 'fruit_veg_pct',
    'nutri_score_label': 'official_nutriscore',
    'green_score_label': 'ecoscore_grade'
}).copy()

print(f"\n📊 Loaded {len(df_clean)} products")

# Calculate Nutri-Score using our implementation
print(f"\n🔄 Computing Nutri-Scores using our algorithm...")
df_result = NutriScoreCalculator.calculate_for_dataframe(df_clean)

# Compare with official scores
print(f"\n✓ Comparison with official Nutri-Score:")
print(f"   Computed scores: {df_result['computed_class'].value_counts().sort_index().to_dict()}")
print(f"   Official scores: {df_result['official_nutriscore'].value_counts().sort_index().to_dict()}")

# Calculate accuracy
accuracy = accuracy_score(df_result['official_nutriscore'], df_result['computed_class'])
print(f"\n📈 Overall Accuracy: {accuracy*100:.2f}%")

# Show mismatches
mismatches = df_result[df_result['computed_class'] != df_result['official_nutriscore']]
print(f"\n⚠️  Mismatches: {len(mismatches)} products ({len(mismatches)/len(df_result)*100:.1f}%)")

if len(mismatches) > 0:
    print(f"\nFirst 10 mismatches:")
    print(mismatches[['product_name', 'official_nutriscore', 'computed_class',
                      'computed_score', 'N', 'P']].head(10).to_string())

# Confusion Matrix
print(f"\n📊 Confusion Matrix:")
cm = confusion_matrix(df_result['official_nutriscore'], df_result['computed_class'],
                     labels=['A', 'B', 'C', 'D', 'E'])
cm_df = pd.DataFrame(cm,
                     index=['Official A', 'Official B', 'Official C', 'Official D', 'Official E'],
                     columns=['Computed A', 'Computed B', 'Computed C', 'Computed D', 'Computed E'])
print(cm_df)

# Classification report
print(f"\n📋 Detailed Classification Report:")
print(classification_report(df_result['official_nutriscore'], df_result['computed_class'],
                           labels=['A', 'B', 'C', 'D', 'E'], zero_division=0))

# Save results
df_result.to_csv('nutriscore_validation_results.csv', index=False)
df_result.to_excel('nutriscore_validation_results.xlsx', index=False)

print(f"\n💾 Saved validation results to:")
print(f"   - nutriscore_validation_results.csv")
print(f"   - nutriscore_validation_results.xlsx")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title(f'Nutri-Score: Official vs Computed\nAccuracy: {accuracy*100:.2f}%', fontsize=14, weight='bold')
plt.ylabel('Official Nutri-Score', fontsize=12)
plt.xlabel('Computed Nutri-Score', fontsize=12)
plt.tight_layout()
plt.savefig('nutriscore_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"   - nutriscore_confusion_matrix.png")

print("\n" + "=" * 60)
print("✅ Validation complete!")
print("=" * 60)

# Analysis of score distribution
print(f"\n📊 Score Statistics:")
print(f"   Mean computed score: {df_result['computed_score'].mean():.2f}")
print(f"   Median computed score: {df_result['computed_score'].median():.2f}")
print(f"   Score range: [{df_result['computed_score'].min()}, {df_result['computed_score'].max()}]")
print(f"   Negative points (N) range: [{df_result['N'].min()}, {df_result['N'].max()}]")
print(f"   Positive points (P) range: [{df_result['P'].min()}, {df_result['P'].max()}]")
