# Time Feature Engineering Implementation
## November 12, 2025 Clue - COMPLETED ✅

### 📜 The Clue (Yoda's Wisdom)
> "Not only what happens, but when. Time, not just an index, but a feature it is."
> 
> "Your signal, bare it is. But the 'when', rich context it gives. Is it the hour of the 
> day? The day of the week? The start of the month? These are not numbers, but cycles. 
> Transform this 'hour' into a feature. And if it is a category ('Monday'), in the 
> embedding space, its true meaning let it learn."

---

## 🎯 Implementation Summary

### What Was Done
The `time` column (sequential timesteps 0-159 per pirate) was transformed from a simple ignored index into **4 rich temporal features**:

1. **time_normalized** - Continuous [0.0, 1.0]
   - Relative position in sequence
   - Formula: `time / max(time)` per pirate
   - Captures progression through recording session
   - Use case: Model learns if pain patterns change over sequence progression

2. **time_sin** - Continuous [-1.0, 1.0]
   - Cyclical encoding (sine component)
   - Formula: `sin(2π * time / period)`
   - Period = ~53 timesteps (creates ~3 cycles per 160-step sequence)
   - Captures periodic/repeating patterns

3. **time_cos** - Continuous [-1.0, 1.0]
   - Cyclical encoding (cosine component)
   - Formula: `cos(2π * time / period)`
   - Paired with time_sin for complete cyclical representation
   - Together they encode time on a circle (no discontinuity at boundaries)

4. **time_position** - Categorical [0, 1, 2]
   - Early/Mid/Late indicator
   - 0 = Early (0-33% of sequence)
   - 1 = Mid (33-66% of sequence)
   - 2 = Late (66-100% of sequence)
   - Uses embeddings: `nn.Embedding(num_embeddings=3, embedding_dim=2)`
   - Allows model to learn context-specific meanings of "early" vs "late"

---

## 📂 Files Modified

### 1. `preprocessing.py`
**New function added:**
```python
def add_time_features(df, df_test):
    """
    Add time-based features implementing November 12 clue.
    Creates normalized, cyclical, and categorical time representations.
    """
```

**Integration points:**
- `run_preprocessing()` - Calls `add_time_features()` before other preprocessing
- `run_test_preprocessing()` - Applies same time features to test set
- `prepare_data_with_embeddings()` - Updated to include:
  - `time_position` in categorical features (for embeddings)
  - `time_normalized`, `time_sin`, `time_cos` in continuous features

**Feature organization:**
- **Categorical** (for embeddings): `pain_survey_1-4` + `time_position` (5 features)
- **Continuous**: 30 joints + 3 time features + `has_prosthetics` (34 features)

### 2. `CLUES_TRACKER.md`
- Status updated: 🔴 Not Started → 🟢 Implemented
- Added comprehensive implementation details
- Listed all 4 new features with formulas
- Documented integration into preprocessing pipeline

### 3. `data_exploration.ipynb`
**New section added: "Time Features"**
- Markdown cell explaining the November 12 clue
- Code cell: Load and create time features
- Code cell: Visualize time features for a single pirate (4 plots)
  - Normalized time progression
  - Cyclical encoding (sin/cos curves)
  - Time position category distribution
  - 2D phase space visualization
- Code cell: ANOVA analysis (time features vs pain labels)

### 4. `test_time_features.py` (New file)
Standalone test script to verify time feature creation

---

## 🔬 Technical Details

### Cyclical Encoding Rationale
**Why sin/cos instead of just normalized time?**
- Linear time (0→159) creates discontinuity at boundaries
- Cyclical encoding (sin/cos) maps time onto a circle
- Captures periodic patterns (e.g., stride cycles, arm swing phases)
- No artificial "cliff" between end and start of sequences

**Period selection:**
- Average sequence length: 159 timesteps
- Period chosen: `max(50, avg_length/3)` = 53 timesteps
- Creates ~3 complete cycles per sequence
- Allows model to learn sub-sequence periodic patterns

### Distribution Analysis
**Time position categories (Early/Mid/Late):**
- Training set:
  - Early: 35,033 samples (33.12%)
  - Mid: 34,372 samples (32.50%)
  - Late: 36,355 samples (34.38%)
- Nearly uniform distribution ✅

---

## 🧪 Testing

**Test script:** `test_time_features.py`
```bash
python test_time_features.py
```

**Results:**
- ✅ All 4 time features created successfully
- ✅ Train shape: (105760, 43) - Added 4 columns
- ✅ Test shape: (211840, 43) - Same transformation applied
- ✅ Features have expected ranges and distributions

---

## 📊 Expected Benefits

### Why This Should Improve Model Performance

1. **Temporal Context**
   - Model learns if pain patterns change over time during activity
   - Early vs late phases of movement may have different pain signatures

2. **Periodic Pattern Detection**
   - Cyclical encoding captures repeating motion patterns (gait, arm swing)
   - Pain may correlate with specific phases of periodic movements

3. **Position-Specific Learning**
   - Embeddings let model learn context: "early = warmup, late = fatigue"
   - Different pain dynamics at sequence start vs end

4. **Previously Unused Information**
   - Time column was completely ignored before
   - Now leveraging all available temporal information

---

## 🚀 Next Steps

### Immediate (Required for full integration):
1. **Update existing model training notebooks** to use new time features
   - `solution_1_cnn_improved.ipynb`
   - Other solution notebooks
2. **Include time features in `data_cols`** for sequence building
3. **Update CNN/RNN architectures** to handle increased input size:
   - Old: 31 features (30 joints + has_prosthetics)
   - New: 35 features (30 joints + 4 time + has_prosthetics)
4. **Update embedding models** to include `time_position` embedding

### Evaluation:
1. Train model with time features
2. Compare performance vs baseline (without time features)
3. Visualize learned embeddings for `time_position`
4. Analyze if time features correlate with pain labels (ANOVA in notebook)

---

## 📝 Code Example: Using Time Features

### For Standard Models (CNN/RNN):
```python
# In your training notebook
from preprocessing import run_preprocessing

# Get preprocessed data (now includes time features)
train_df, val_df, train_target, val_target = run_preprocessing()

# Define feature columns
joint_cols = ["joint_" + str(i).zfill(2) for i in range(30)]
time_cols = ['time_normalized', 'time_sin', 'time_cos']
data_cols = joint_cols + time_cols + ['has_prosthetics', 'time_position']

# Build sequences (time_position can be treated as continuous or embedded)
X_train, y_train = build_sequences(train_df, train_target, window=300, stride=150)
# Shape: (num_sequences, 300 timesteps, 35 features)
```

### For Embedding-Based Models:
```python
from preprocessing import prepare_data_with_embeddings

# Prepare data with categorical/continuous separation
df, df_test, categorical_cols, continuous_cols = prepare_data_with_embeddings(df, df_test)

# categorical_cols: ['pain_survey_1', 'pain_survey_2', 'pain_survey_3', 'pain_survey_4', 'time_position']
# continuous_cols: ['joint_00', ..., 'joint_29', 'time_normalized', 'time_sin', 'time_cos', 'has_prosthetics']

# Build sequences with embeddings
cat_seq, cont_seq, labels = build_sequences_with_embeddings(
    df, target, window=300, stride=150,
    pain_survey_cols=categorical_cols,
    continuous_cols=continuous_cols
)

# Model architecture
class CNNWithTimeEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 pain survey embeddings
        self.pain_embeds = nn.ModuleList([
            nn.Embedding(5, 3) for _ in range(4)
        ])
        # Time position embedding
        self.time_embed = nn.Embedding(3, 2)
        
        # CNN on continuous features (34 features)
        self.conv1 = nn.Conv1d(34, 64, kernel_size=5)
        # ... rest of architecture
```

---

## ✅ Completion Checklist

- [x] Create `add_time_features()` function
- [x] Integrate into `run_preprocessing()` and `run_test_preprocessing()`
- [x] Update `prepare_data_with_embeddings()` for time features
- [x] Update `CLUES_TRACKER.md` with implementation details
- [x] Add time feature visualization to `data_exploration.ipynb`
- [x] Create test script to verify functionality
- [ ] Update model training notebooks to use time features
- [ ] Train and evaluate models with time features
- [ ] Document performance comparison

---

## 🎓 Key Insights from Implementation

1. **Sequential time is now 4 complementary views:**
   - Linear progression (normalized)
   - Periodic cycles (sin/cos)
   - Categorical phases (early/mid/late)
   - Original timestep (preserved for reference)

2. **No information loss:**
   - Original `time` column retained
   - All derived features are reversible transformations

3. **Ready for both continuous and embedding-based models:**
   - 3 continuous features (normalized, sin, cos)
   - 1 categorical feature (position) for embeddings

4. **Data-driven period selection:**
   - Period automatically calculated from average sequence length
   - Adapts to dataset characteristics

---

**Implementation Date:** November 14, 2025  
**Clue Date:** November 12, 2025  
**Status:** ✅ COMPLETE  
**Ready for Model Training:** Yes (pending notebook updates)
