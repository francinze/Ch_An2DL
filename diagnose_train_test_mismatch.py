"""
Diagnose Train vs Test Distribution Mismatch

This script helps identify why validation F1 is 90% but test F1 is only 60%.
Possible causes:
1. Train/test data come from different distributions
2. Pain survey features contain train-specific information (data leakage)
3. Sequence aggregation strategy doesn't work on test set
4. Test set has different class balance
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

print("=" * 80)
print("ANALYZING TRAIN/TEST DISTRIBUTION MISMATCH")
print("=" * 80)

# Load data
print("\n1. Loading data...")
df_train = pd.read_csv("pirate_pain_train.csv")
df_test = pd.read_csv("pirate_pain_test.csv")
target = pd.read_csv("pirate_pain_train_labels.csv")

print(f"   Train samples: {len(df_train['sample_index'].unique())}")
print(f"   Test samples: {len(df_test['sample_index'].unique())}")

# -------------------------------------------------------------------
# Check 1: Pain Survey Columns - POTENTIAL DATA LEAKAGE!
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("⚠️  CHECK 1: PAIN SURVEY COLUMNS - MAJOR CONCERN!")
print("=" * 80)

pain_survey_cols = [col for col in df_train.columns if 'pain_survey' in col]
print(f"\nFound {len(pain_survey_cols)} pain survey columns: {pain_survey_cols}")

# Check if pain surveys correlate with labels
merged = df_train.merge(target, on='sample_index')

print("\n📊 Pain survey distributions by class:")
for col in pain_survey_cols:
    print(f"\n{col}:")
    for label in sorted(merged['label'].unique()):
        values = merged[merged['label'] == label][col].value_counts().sort_index()
        label_name = {0: 'no_pain', 1: 'low_pain', 2: 'high_pain'}[label]
        print(f"  {label_name}: {values.to_dict()}")

# Check if pain surveys exist in test set
print("\n🔍 Pain survey columns in TEST set:")
if all(col in df_test.columns for col in pain_survey_cols):
    print("   ✅ All pain survey columns present in test set")
    print("\n   Sample test pain survey values:")
    print(df_test[pain_survey_cols].head())
    
    # Check if test has different distribution
    print("\n   Test pain survey distributions:")
    for col in pain_survey_cols:
        train_dist = df_train[col].value_counts(normalize=True).sort_index()
        test_dist = df_test[col].value_counts(normalize=True).sort_index()
        print(f"\n   {col}:")
        print(f"      Train: {train_dist.to_dict()}")
        print(f"      Test:  {test_dist.to_dict()}")
else:
    print("   ❌ Pain survey columns MISSING in test set!")
    print("   This could be why test performance is poor!")

# -------------------------------------------------------------------
# Check 2: Feature Distributions
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("CHECK 2: FEATURE DISTRIBUTIONS (Train vs Test)")
print("=" * 80)

joint_cols = [f"joint_{str(i).zfill(2)}" for i in range(31)]
number_cols = ['n_legs', 'n_hands', 'n_eyes']

print("\n📊 Categorical features (body measurements):")
for col in number_cols:
    train_dist = df_train[col].value_counts(normalize=True).sort_index()
    test_dist = df_test[col].value_counts(normalize=True).sort_index()
    print(f"\n{col}:")
    print(f"  Train: {train_dist.to_dict()}")
    print(f"  Test:  {test_dist.to_dict()}")

print("\n📊 Joint features (summary statistics):")
for col in joint_cols[:5]:  # Check first 5 joints
    train_mean = df_train[col].mean()
    train_std = df_train[col].std()
    test_mean = df_test[col].mean()
    test_std = df_test[col].std()
    
    # Check if distributions are similar
    diff_mean = abs(train_mean - test_mean) / (train_std + 1e-8)
    if diff_mean > 0.5:
        print(f"   ⚠️  {col}: Train μ={train_mean:.3f} σ={train_std:.3f}, Test μ={test_mean:.3f} σ={test_std:.3f} (DIFFERENT!)")
    else:
        print(f"   ✅ {col}: Train μ={train_mean:.3f} σ={train_std:.3f}, Test μ={test_mean:.3f} σ={test_std:.3f}")

# -------------------------------------------------------------------
# Check 3: Sequence Lengths
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("CHECK 3: SEQUENCE LENGTHS (affects windowing)")
print("=" * 80)

train_lengths = df_train.groupby('sample_index').size()
test_lengths = df_test.groupby('sample_index').size()

print(f"\nTrain sequence lengths:")
print(f"  Mean: {train_lengths.mean():.1f} frames")
print(f"  Std: {train_lengths.std():.1f}")
print(f"  Min: {train_lengths.min()} Max: {train_lengths.max()}")

print(f"\nTest sequence lengths:")
print(f"  Mean: {test_lengths.mean():.1f} frames")
print(f"  Std: {test_lengths.std():.1f}")
print(f"  Min: {test_lengths.min()} Max: {test_lengths.max()}")

if abs(train_lengths.mean() - test_lengths.mean()) > 100:
    print("\n⚠️  WARNING: Test sequences are significantly different length!")
    print("   This affects number of windows per sample → aggregation bias")

# -------------------------------------------------------------------
# Check 4: Missing Values
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("CHECK 4: MISSING VALUES")
print("=" * 80)

train_missing = df_train.isnull().sum()
test_missing = df_test.isnull().sum()

if train_missing.sum() > 0 or test_missing.sum() > 0:
    print("\n⚠️  Missing values found:")
    print(f"   Train: {train_missing[train_missing > 0].to_dict()}")
    print(f"   Test: {test_missing[test_missing > 0].to_dict()}")
else:
    print("\n✅ No missing values in either dataset")

# -------------------------------------------------------------------
# DIAGNOSIS SUMMARY
# -------------------------------------------------------------------
print("\n" + "=" * 80)
print("🔍 DIAGNOSIS SUMMARY")
print("=" * 80)

print("\nMost likely causes of 90% val → 60% test:")
print()
print("1. ⚠️  PAIN SURVEY LEAKAGE:")
print("   If pain surveys are self-reported pain levels, they are")
print("   essentially giving away the answer during training!")
print("   → Solution: Remove pain survey features or use them differently")
print()
print("2. 🎯 DIFFERENT TEST DISTRIBUTION:")
print("   Test set may have different class balance or harder cases")
print("   → Solution: Can't fix, but be aware validation F1 is optimistic")
print()
print("3. 📊 AGGREGATION STRATEGY:")
print("   Multiple windows per sample → majority vote may not work well")
print("   → Solution: Try different aggregation (avg probability, weighted)")
print()
print("4. 🔧 PREPROCESSING MISMATCH:")
print("   If MinMaxScaler was fit on train, must use SAME scaler on test")
print("   → Solution: Save and load scaler (currently refitting on test!)")

print("\n" + "=" * 80)
print("RECOMMENDED ACTIONS:")
print("=" * 80)
print("\n1. **Remove pain survey features** from model training")
print("   These might be self-reported pain → direct label leakage")
print()
print("2. **Save and reuse MinMaxScaler** fit on training data")
print("   Currently generate_test_predictions.py refits on test!")
print()
print("3. **Try ensemble aggregation** instead of majority vote")
print("   Weight predictions by confidence (softmax probability)")
print()
print("4. **Check if test has different class distribution**")
print("   Your model is optimized for 78/14/8 split")
print("   If test is 50/30/20, performance will drop significantly")
