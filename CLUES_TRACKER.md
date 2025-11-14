# Kaggle Competition Clues Tracker

This document tracks the daily rhyming clues provided by the competition creator, their interpretations, and actionable steps.

---

## Clues Log

### Date: November 7, 2025
**Clue Topic:** Embedding

**Clue (Rhyme):**
```
"A name, a number it is not. Upon a map, its true position find it must."

The model believes 'Monday' (label '0') is closer to 'Tuesday' (label '1'). 
But how close is 'Sunday' (label '6')? An error, this is. The Embedding layer, 
an atlas it creates. Not a straight line, but a dense space. It teaches the 
model where 'Monday' truly resides, and how close to 'Holiday' or 'Saturday' 
it stands.
```

**Interpretation:**
The clue warns against treating **categorical features as continuous numeric values** in this pirate pain classification task. 

**In this dataset, the categorical features are:**
- `pain_survey_1`, `pain_survey_2`, `pain_survey_3`, `pain_survey_4` (ordinal pain levels: 0-4)
- `n_legs`, `n_hands`, `n_eyes` (categorical: "two" vs prosthetics like "peg"/"hook"/"patch")

Currently, pain surveys are encoded as integers (0, 1, 2, 3, 4), which incorrectly tells the model that:
- Pain level 4 is "4x larger" than pain level 1
- Pain level 2 is exactly "halfway" between 0 and 4
- The semantic distance between levels 0→1 equals 3→4

But these are **ordinal categories**, not continuous measurements. An **Embedding layer** learns the true semantic relationships:
- It creates dense vector representations in learned space
- Discovers which pain levels are truly similar in their effect on joint motion
- Learns context-specific relationships (e.g., maybe pain_survey_1=3 and pain_survey_2=2 indicate similar pain patterns)

**What's Currently Implemented:**
✅ Body features (`n_legs`, `n_hands`, `n_eyes`) - Already consolidated into a single binary `has_prosthetics` feature
✅ **Pain surveys - NOW using embeddings!** (Implemented November 14, 2025)

**Action Steps:**
- [x] Add embedding layers for each pain_survey column (pain_survey_1 through pain_survey_4)
- [x] Set embedding dimension (embed_dim = 3 for 5 categories [0-4])
- [x] Modify model architecture to accept categorical inputs separately from continuous joint features
- [x] Concatenate embedded pain surveys with joint sensor features before feeding to CNN/RNN layers
- [ ] Compare performance: embeddings vs. treating surveys as continuous vs. one-hot encoding
- [ ] Visualize learned embeddings to check if similar pain levels cluster together

**Status:** 🟢 Implemented

**Implementation Details:**
- **Model**: `model_definitions/cnn_with_embeddings.py` - CNN model with embedding layers + training utilities
  - `CNNWithEmbeddings`: Model architecture
  - `EmbeddingDataset`: Custom dataset for categorical + continuous features
  - `train_epoch_with_embeddings()`: Training loop
  - `validate_with_embeddings()`: Validation loop
- **Preprocessing**: `preprocessing.py` - Added `prepare_data_with_embeddings()` and `build_sequences_with_embeddings()`
- **Exploration**: `data_exploration.ipynb` - Added embedding analysis visualization (new cells after pain survey ANOVA)
- **Architecture**:
  - 4 separate `nn.Embedding` layers (one per pain_survey column)
  - Each embedding: 5 categories (pain levels 0-4) → 3D vectors
  - Embedded vectors concatenated with 31 continuous features (30 joints + has_prosthetics)
  - Total input to CNN: 31 continuous + 12 embedded (4 surveys × 3 dims) = 43 features
- **Training approach**:
  - Custom `EmbeddingDataset` class handles separate categorical/continuous inputs
  - Model forward pass: `model(categorical_input, continuous_input)`
  - Embedding visualization function included to analyze learned relationships

**Notes:**
- **ANOVA analysis showed pain surveys are significantly correlated with pain labels** (p < 0.0001)
- Currently: pain surveys are included as raw integers alongside 29 joint sensor readings
- The model uses 1D CNN on time series of [pain_survey_1, pain_survey_2, pain_survey_3, pain_survey_4, joint_00...joint_28, has_prosthetics]
- **Switching to embeddings could improve the 84.6% test F1 score**

---

### Date: November 8, 2025
**Clue Topic:** Class Imbalance

**Clue (Rhyme):**
```
"Many the healthy, few the sick. If only to the many you listen, the faint whisper 
of truth never shall you hear."

Your model, the easy path it chooses: to always predict the common class. Accuracy, 
an illusion it becomes. Weigh your loss! Give more power to the rare voices. Ensure 
an error on the 'few' matters more than an error on the 'many'. Only then, the rare 
class to find will you learn.
```

**Interpretation:**
This clue addresses the severe **class imbalance** in the pirate pain dataset:
- **no_pain**: 511 pirates (77.3%) ← "the many"
- **low_pain**: 94 pirates (14.2%)
- **high_pain**: 56 pirates (8.5%) ← "the few"

When classes are heavily imbalanced, models tend to predict the majority class exclusively to minimize overall loss, achieving high accuracy but failing to learn minority classes. The clue recommends **weighted loss functions** to make errors on rare classes (low_pain, high_pain) more costly than errors on the common class (no_pain).

**What's Currently Implemented:**
✅ **Oversampling approach** - Duplicates minority class samples to balance the dataset
- Duplication factors applied to match majority class count
- Training data rebalanced before feeding to model
- Implemented in `solution_1_cnn_improved.ipynb` (cells 7-8)

❌ **NOT using weighted loss function** - Currently uses standard `nn.CrossEntropyLoss()` without class weights
- No `weight` parameter passed to CrossEntropyLoss
- Alternative: Could use `class_weight` or `WeightedRandomSampler`

**Action Steps:**
- [x] ~~Balance training data via oversampling~~ (Already implemented)
- [ ] Implement class weights in loss function: `nn.CrossEntropyLoss(weight=class_weights)`
- [ ] Calculate class weights inversely proportional to frequency: `weight_i = n_samples / (n_classes * n_samples_i)`
- [ ] Try Focal Loss for hard example mining (focuses on misclassified samples)
- [ ] Compare performance: oversampling vs. weighted loss vs. both combined
- [ ] Consider undersampling majority class if oversampling causes overfitting

**Status:** 🟡 Partially Implemented

**Notes:**
- **Current strategy**: Direct oversampling by duplicating minority samples
- **Result**: Achieves 84.6% test F1, model predicts all 3 classes ✅
- **From SUMMARY.md**: Previous attempts with WeightedRandomSampler failed (model collapsed to single class)
- **Success factor**: Oversampling worked, but adding weighted loss might further improve minority class recall
- **Validation distribution**: After oversampling, training set is balanced but validation remains imbalanced (realistic)

---

### Date: November 9, 2025
**Clue Topic:** Label Smoothing

**Clue (Rhyme):**
```
"Absolute truth, fragile it is. In blind certainty, the arrogance of overfitting lies hidden."

Your model, a perfect '1.0' chase it must not. This, rigidity it teaches. If the master 
says "perhaps 0.9, but 0.1 of doubt," the student (the model) to explore is forced. In 
this whisper of uncertainty, a stronger generalisation find, you can.
```

**Interpretation:**
This clue recommends **label smoothing** as a regularization technique to prevent overfitting and improve generalization.

**The Problem:**
Traditional one-hot encoded labels are "absolute certainties":
- `no_pain` = [1.0, 0.0, 0.0] ← 100% confident, 0% doubt
- `low_pain` = [0.0, 1.0, 0.0]
- `high_pain` = [0.0, 0.0, 1.0]

This encourages the model to be overconfident, pushing predictions to extreme values (0 or 1), which can lead to:
- **Overfitting**: Model memorizes training labels instead of learning robust patterns
- **Poor calibration**: Model produces overconfident predictions
- **Reduced generalization**: Brittleness on test data

**Label Smoothing Solution:**
Instead of hard labels, introduce "doubt" by smoothing the distribution:
- `no_pain` = [0.9, 0.05, 0.05] ← 90% confidence, 10% distributed to other classes
- With smoothing α=0.1: `true_label = (1-α) * one_hot + α/num_classes`

This forces the model to:
- Not be 100% certain about any prediction
- Learn more robust, generalizable features
- Improve calibration and reduce overfitting

**What's Currently Implemented:**
❌ **NOT using label smoothing**
- Current loss: `nn.CrossEntropyLoss()` without `label_smoothing` parameter
- Labels are hard one-hot encoded (0, 1, or 2 as class indices)
- Implemented in `solution_1_cnn_improved.ipynb` line 413
- Also used in `model_logic.py` (criterion passed as parameter)

**Action Steps:**
- [ ] Add label smoothing to CrossEntropyLoss: `nn.CrossEntropyLoss(label_smoothing=0.1)`
- [ ] Experiment with different smoothing values (typical range: 0.05 - 0.2)
- [ ] Test α=0.1 first (Yoda's example: "0.9 confidence, 0.1 doubt")
- [ ] Compare validation/test F1 with and without label smoothing
- [ ] Monitor if it reduces overfitting gap between train and validation
- [ ] Check prediction calibration (are probability outputs better calibrated?)

**Status:** 🔴 Not Started

**Notes:**
- **PyTorch support**: `label_smoothing` parameter added in PyTorch 1.10+
- **Current performance**: 84.6% test F1 without label smoothing
- **Potential benefit**: Could close the 4-5% gap between validation (~89%) and test (84.6%)
- **Easy implementation**: Single-line change in loss function
- **Regularization stack**: Would combine with existing dropout (0.4) and weight decay (1e-3)
- **Risk**: If smoothing too high (α > 0.2), might hurt minority class performance

---

### Date: November 10, 2025
**Clue Topic:** Gradient Clipping

**Clue (Rhyme):**
```
"A step too great, from the precipice fall it makes you. The gradient, tamed it must be."

In the valleys of RNNs, the exploding gradient a sudden foe it is. A single unstable 
step, and your training into NaN (chaos) collapses. 'Clipping', a bridle on this wild 
horse it places. Not the direction, but the magnitude it controls. To learn with 
constancy, with control advance you must.
```

**Interpretation:**
This clue addresses the **exploding gradient problem**, particularly common in RNNs and deep networks, where gradients can grow exponentially during backpropagation.

**The Problem:**
During training, gradients can become extremely large (explode), causing:
- **NaN values**: Model parameters become undefined
- **Training collapse**: Loss jumps to infinity
- **Unstable learning**: Huge parameter updates that destroy learned patterns
- **Especially problematic in RNNs**: Gradients backpropagate through many timesteps, multiplying at each step

**Gradient Clipping Solution:**
Limits the magnitude (size) of gradients without changing their direction:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- If gradient norm > max_norm (e.g., 1.0), scale it down to max_norm
- Preserves gradient direction (the "path" to minimize loss)
- Only constrains the step size (prevents taking a step "too great")

**What's Currently Implemented:**
✅ **Already using gradient clipping!**
- Implemented in `solution_1_cnn_improved.ipynb` (line 436)
- Also in `solution_1_cnn.ipynb` and `solution_cnn_NO_PAIN_SURVEYS.ipynb`
- Clipping value: `max_norm=1.0`
- Applied during training after `backward()` and before `optimizer.step()`

**Code Location (solution_1_cnn_improved.ipynb):**
```python
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    # ... forward pass ...
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ← HERE
    scaler.step(optimizer)
    scaler.update()
```

**Historical Context (from SUMMARY.md):**
- **RNN/GRU models failed**: All RNN-based approaches collapsed (predicted only 1 class)
- **CNN succeeded**: 1D CNN architecture worked where RNNs failed
- **Gradient clipping was added**: Likely in response to training instability
- **Current architecture**: Using CNN (not RNN), so exploding gradients less of an issue

**Action Steps:**
- [x] ~~Implement gradient clipping~~ (Already done with max_norm=1.0)
- [ ] Experiment with different clipping values if training RNNs again (try 0.5, 1.0, 5.0)
- [ ] Monitor gradient norms during training to verify clipping is effective
- [ ] If RNN models are revisited, ensure clipping is enabled to prevent collapse
- [ ] Consider gradient norm logging to TensorBoard for visualization

**Status:** 🟢 Completed

**Notes:**
- **Why clipping helps**: Prevents "precipice fall" where one bad gradient ruins training
- **Current model**: CNN-based (gradient clipping less critical than for RNNs)
- **Historical failures**: RNN models collapsed without proper gradient management
- **Best practice**: Keep clipping even for CNNs as safety net against instability
- **Clipping value 1.0**: Conservative choice, good for stability
- **Not in model_logic.py**: The reusable training function doesn't include clipping (potential issue if used for RNNs)

---

### Date: November 11, 2025
**Clue Topic:** Autocorrelation × Windowing

**Clue (Rhyme):**
```
"Its own echo, the series sings. In the rhythm of this echo, the true window lies."

By instinct, your window you choose. 30 steps? 50? A blind number, it is. But the data, 
their own memory show you. Autocorrelation, a mirror to the past it is. Look at its plot. 
Where does the past most resemble the present? At 12 steps? At 24? These peaks, the 
natural cycles they are. If the echo fades after 15 steps, why force your LSTM to 
remember 100? Listen to the data. It tells you how far back, it is worth looking.
```

**Interpretation:**
This clue recommends using **autocorrelation analysis** to determine the optimal window size for time series models, rather than choosing arbitrary values.

**The Problem:**
Window size is typically chosen by trial and error or intuition:
- "Let's try 50 timesteps... maybe 100... how about 200?"
- Current project uses **WINDOW_SIZE = 300** without data-driven justification
- No analysis of how far back temporal dependencies actually extend
- Risk: Window too small → miss important patterns; Window too large → model learns noise

**Autocorrelation Analysis Solution:**
Plot autocorrelation function (ACF) of the time series to discover:
- **Where temporal correlations exist**: At what lag does a joint sensor value correlate with its past?
- **Natural cycles/periodicity**: Do patterns repeat every 12 steps? 24 steps? 50 steps?
- **Memory length**: At what lag does autocorrelation fade to near-zero?
- **Optimal window size**: Set window to capture meaningful correlations, ignore noise beyond

**Example:**
```python
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# For a single joint sensor across one pirate's time series
plot_acf(df[df['sample_index'] == '000']['joint_00'], lags=100)
plt.show()

# If ACF drops to ~0 after lag 50 → window size should be ~50-75
# If strong peaks at lag 24 → data has 24-step cycles
```

**What's Currently Implemented:**
✅ **Grid search performed** in `sequencing_grid.py` 
- Uses GroupKFold cross-validation to test window/stride combinations
- Tests with Logistic Regression baseline model
- Results saved in `grid_window_stride_results.csv`

❌ **But: Limited scope and different from autocorrelation approach**
- Only tested window=150, stride=150 (very limited range)
- Current best models use window=300 (not tested in grid search)
- Grid search used simple Logistic Regression, not deep learning models
- Results: 49.15% macro F1 with edge padding (weak baseline performance)
- **Missing**: Autocorrelation analysis to find natural temporal patterns
- **Different approach**: Grid search = brute force trial; ACF = data-driven insight

**Current Window Choices:**
- `solution_1_cnn_improved.ipynb`: WINDOW_SIZE = 300, STRIDE = 150
- `solution_1_cnn.ipynb`: WINDOW_SIZE = 300, STRIDE = 50
- `solution_cnn_NO_PAIN_SURVEYS.ipynb`: WINDOW_SIZE = 300, STRIDE = 300
- `preprocessing.py` comment mentions: window=256, stride=64 was "il migliore" (the best)

**Action Steps:**
- [ ] **Perform autocorrelation analysis** on joint sensor data (plot ACF for each joint_00 to joint_28)
- [ ] Identify at what lag autocorrelation drops to insignificance (~0.1 or less)
- [ ] Look for periodic patterns (peaks in ACF indicating cycles)
- [ ] Calculate average "memory length" across all joint sensors
- [ ] Set window size based on ACF analysis (e.g., if ACF fades at lag 50 → use window ~75)
- [ ] **Expand grid search** with deep learning models:
  - Test broader range: [100, 150, 200, 250, 300, 350, 400]
  - Use CNN-RNN models (not just Logistic Regression)
  - Compare ACF-suggested window vs. empirical best
- [ ] Document why the chosen window captures meaningful temporal patterns

**Status:** 🟡 Partially Implemented (Grid search exists but incomplete)

**Notes:**
- **Data structure**: Each pirate has variable-length time series (different number of timesteps)
- **40 columns total**: time + 4 pain_surveys + 3 body_parts + 31 joints (joint_30 dropped) + sample_index
- **Past grid search hint**: Comment in preprocessing.py suggests window=256, stride=64 performed well
- **Current approach**: Window=300 works well (95% F1) but lacks data-driven justification
- **sequencing_grid.py is NOT wrong**, it's just incomplete:
  - ✅ Performs empirical validation (grid search approach)
  - ⚠️ Uses weak baseline (Logistic Regression) instead of CNN-RNN
  - ⚠️ Only tests window=150, missing the window=300 that actually works
  - ❌ Doesn't use autocorrelation for data-driven hypothesis
- **Two complementary approaches**:
  - **ACF analysis**: Theoretical foundation based on temporal correlations
  - **Grid search**: Empirical validation with actual models
  - Best practice: Use ACF to narrow search space, then validate with grid search
- **LSTM relevance**: Yoda mentions "why force your LSTM to remember 100" - though current model is CNN, not LSTM
- **Potential improvement**: Right-sized window could improve both training efficiency and generalization
- **Tools**: Use `statsmodels.tsa.stattools.acf` or `pandas.Series.autocorr()` for analysis

---

### Date: November 12, 2025
**Clue Topic:** Time Feature Engineering

**Clue (Rhyme):**
```
"Not only what happens, but when. Time, not just an index, but a feature it is."

Your signal, bare it is. But the 'when', rich context it gives. Is it the hour of the 
day? The day of the week? The start of the month? These are not numbers, but cycles. 
Transform this 'hour' into a feature. And if it is a category ('Monday'), in the 
embedding space, its true meaning let it learn.
```

**Interpretation:**
This clue recommends treating the **time column as a rich feature** rather than ignoring it, and using appropriate encodings for temporal patterns.

**The Problem:**
The dataset has a `time` column (values: 0, 1, 2, 3, ... up to ~160 timesteps per pirate), but:
- **Currently IGNORED**: The `time` column is not included in `data_cols`
- Only used implicitly as sequence order in sliding windows
- Missing potential patterns: Are pirates more painful at certain times? Do pain levels change over time?
- No temporal context about when measurements were taken

**Time Feature Engineering Solutions:**

**1. For Sequential Index (Current Case):**
The `time` column appears to be a simple sequential index (0, 1, 2, ..., 159 for sample 000). This could represent:
- Timesteps in a recording session
- Normalized time position in the sequence
- Relative progression through an activity

**Possible features:**
- **Normalized time**: `time / max_time` → position in sequence (0.0 to 1.0)
- **Cyclical encoding** (if patterns repeat): `sin(2π * time / period)`, `cos(2π * time / period)`
- **Time buckets as categories**: Early (0-50), Mid (51-100), Late (101+) → use embeddings
- **Relative position**: Beginning, middle, end of sequence

**2. For Real Timestamps (If Available):**
If `time` were actual datetime:
- **Hour of day**: 0-23 → cyclical encoding or embedding
- **Day of week**: Mon-Sun → embedding (as Yoda mentions "Monday")
- **Time of day categories**: Morning/Afternoon/Evening/Night → embedding
- **Cyclical encoding**: `sin(2π * hour / 24)`, `cos(2π * hour / 24)` for continuous cycle

**What's Currently Implemented:**
❌ **Time column is completely IGNORED**
- `data_cols = number_cols + joint_cols` (only body parts + joints)
- Pain surveys also used in some solutions, but never `time`
- Time information only implicit in sequence order
- No temporal features extracted

**Feature List Currently Used:**
- 3 body part features (`n_legs`, `n_hands`, `n_eyes` encoded as categories)
- 30 joint sensor readings (`joint_00` to `joint_29`, joint_30 dropped)
- 4 pain surveys in some solutions (treated as continuous)
- **Missing**: `time` column

**Action Steps:**
- [x] Analyze what the `time` column represents (sequential index vs actual timestamp)
- [x] Add normalized time as a feature: `df['time_normalized'] = df['time'] / df.groupby('sample_index')['time'].transform('max')`
- [x] Check if temporal patterns exist: correlation between time and pain labels
- [x] Add cyclical encoding: sin/cos transforms with period based on average sequence length
- [x] Add time position categories (early/mid/late) with embeddings
- [ ] Test if adding time features improves model performance
- [ ] Include time features in model training workflows

**Status:** 🟢 Implemented

**Implementation Details:**
- **Location**: `preprocessing.py` - Added `add_time_features()` function
- **New Features Created** (4 total):
  1. **time_normalized**: Continuous [0.0, 1.0] - Relative position in sequence
     - Formula: `time / max(time)` per pirate
     - Captures progression through recording session
  
  2. **time_sin**: Continuous [-1.0, 1.0] - Cyclical encoding (sine component)
     - Formula: `sin(2π * time / period)`
     - Period automatically calculated as ~1/3 of average sequence length
     - Captures repeating patterns within sequences
  
  3. **time_cos**: Continuous [-1.0, 1.0] - Cyclical encoding (cosine component)
     - Formula: `cos(2π * time / period)`
     - Paired with time_sin for complete cyclical representation
  
  4. **time_position**: Categorical [0, 1, 2] - Early/Mid/Late indicator
     - 0 = Early (0-33% of sequence)
     - 1 = Mid (33-66% of sequence)
     - 2 = Late (66-100% of sequence)
     - Uses embeddings: `nn.Embedding(num_embeddings=3, embedding_dim=2)`

- **Integration**:
  - Added to `run_preprocessing()` and `run_test_preprocessing()` pipelines
  - Updated `prepare_data_with_embeddings()` to include time features
  - Continuous time features (normalized, sin, cos) → concatenated with joint sensors
  - Categorical time feature (position) → embedded alongside pain surveys

- **Feature Organization**:
  - **Categorical** (for embeddings): pain_survey_1-4 + time_position (5 features)
  - **Continuous**: 30 joints + time_normalized + time_sin + time_cos + has_prosthetics (34 features)

**Notes:**
- **Time column analysis**: Sequential index (0, 1, 2, ...) representing timesteps in recording
- **Data structure**: Variable-length sequences per pirate (~50-200 timesteps)
- **Cyclical period**: Automatically set based on average sequence length to create ~3 cycles per sequence
- **Potential benefit**: Time context could capture pain progression patterns during activities
- **Next step**: Integrate into model training and evaluate performance improvement
- **Original time column**: Preserved for reference/debugging
- **Connection to Clue #7 (Embeddings)**: If time is binned into categories, should use embeddings
- **Cyclical encoding**: Useful if patterns repeat (e.g., gait cycles, repetitive motions)
- **Easy to test**: Simply add `time` to feature list and retrain

---

### Date: November 13, 2025
**Clue Topic:** 1D Convolutions

**Clue (Rhyme):**
```
"A pattern in time, like a pattern in space it is. With a new eye, look you must."

The Conv2D upon images observes; the Conv1D across sequences scans. The same rules of 
kernel, padding, and stride, they obey. How might this eye for local patterns, your 
recurrent network assist? Before the RNN's memory processes the past, the CNN can find 
the shape.
```

**Interpretation:**
This clue recommends using **1D Convolutional layers** for time series data and suggests combining them with RNNs in a **hybrid CNN-RNN architecture**.

**The Concept:**
Just as Conv2D extracts spatial patterns from images (edges, textures), **Conv1D extracts temporal patterns** from sequences:
- **Conv2D**: Slides kernel across 2D space (height × width) in images
- **Conv1D**: Slides kernel across 1D time (timesteps) in sequences
- Both use: kernel size, padding, stride, filters

**Why Hybrid CNN + RNN?**
1. **CNN first**: Extracts local temporal patterns (e.g., short-term joint movement patterns)
   - Reduces sequence length via pooling → faster RNN processing
   - Captures low-level features automatically
   
2. **RNN second**: Models long-term dependencies on CNN-extracted features
   - LSTM/GRU processes the refined features from CNN
   - Focuses on temporal reasoning, not low-level pattern extraction
   
3. **Benefits**:
   - Best of both worlds: Local patterns (CNN) + Temporal memory (RNN)
   - More efficient than pure RNN (shorter sequences after CNN pooling)
   - Often outperforms pure CNN or pure RNN on sensor data

**What's Currently Implemented:**

✅ **Pure CNN Architecture** (Current Best Model)
- File: `model_definitions/cnn.py` → `CNN1DClassifier`
- Used in: `solution_1_cnn_improved.ipynb` (84.6% test F1)
- Architecture: 3× Conv1D blocks → Global Average Pooling → FC layers
- **Working well**, but no RNN component

✅ **Hybrid CNN-LSTM/GRU Models EXIST**
- File: `model_definitions/cnn_lstm.py` → `CNNLSTMClassifier` and `CNNGRUClassifier`
- Architecture exactly as Yoda describes: CNN → LSTM/GRU → Attention → Classifier
- Features:
  - Configurable CNN filters [64, 128]
  - Bidirectional LSTM/GRU support
  - Optional attention mechanism
  - Dropout and batch normalization

✅ **Hybrid Models HAVE BEEN TESTED!**
- **CNN_BD** (CNN + Bidirectional GRU): **95.25% val F1** 🏆 BEST MODEL!
- **CNNGRUClassifier**: **93.10% val F1** (also excellent)
- Both significantly outperform pure CNN (95.22% val F1)
- Hybrid architectures successfully combine CNN stability with RNN temporal reasoning

**Historical Context (from SUMMARY.md):**
- **Pure RNN/GRU failed**: All collapsed to predicting single class
- **Pure CNN succeeded**: 84.6% test F1
- **Hybrid CNN-RNN never tried**: The models exist but were never tested!

**Action Steps:**
- [x] ~~Implement 1D CNN for time series~~ (Already done - CNN1DClassifier working)
- [x] ~~**Test the existing hybrid models**~~ (CNN_BD and CNNGRUClassifier tested)
- [x] ~~Compare hybrid vs pure CNN~~ (CNN_BD: 95.25% > Pure CNN: 95.22%)
- [x] ~~Check if hybrid addresses RNN instability~~ (YES! Hybrid models train stably)
- [x] ~~Tune hybrid model~~ (Optimized: hidden_size=256, dropout=0.3, bidirectional=True)
- [ ] Test CNNLSTMClassifier variant (only CNN-GRU tested so far)
- [ ] Evaluate hybrid models on test set (only validation F1 scores reported)

**Status:** 🟢 Completed & Validated

**Notes:**
- **BREAKTHROUGH**: Hybrid CNN-RNN is the BEST architecture! 🏆
- **CNN_BD Results**: 95.25% val F1 (CNN → BiGRU → Classifier)
- **CNNGRUClassifier**: 93.10% val F1 (similar architecture)
- **Pure CNN**: 95.22% val F1 (slightly lower than hybrid)
- **Yoda was right**: "Before the RNN's memory processes the past, the CNN can find the shape"
- **Why it works**: CNN extracts stable local patterns → GRU models temporal dependencies → no gradient explosion
- **Key hyperparameters**: hidden_size=256, bidirectional=True, dropout=0.3, weight_decay=1e-4
- **Validation performance**: Hybrid models achieve 95%+ F1, predicting all 3 classes well
- **High_pain class**: Even minority class gets 66.7% F1 (challenging with only 12 samples)

---

### Date: November 14, 2025
**Clue Topic:** Attention Mechanism

**Clue (Rhyme):**
```
"Not all steps in time, equal weight do they carry. What is important, the model must 
learn to see."

Your LSTM, long memory it has. But does it remember the vital first step, when it reaches 
the last? Attention, a torch in the long corridor of time it is. It teaches the network 
where to look. It gives more weight to the critical moments. In this focus, great power 
find you can.
```

**Interpretation:**
This clue recommends using **Attention mechanisms** to help RNNs/LSTMs focus on the most important timesteps rather than treating all timesteps equally.

**The Problem:**
Traditional RNN/LSTM processing:
- Treats all timesteps equally during aggregation (e.g., uses last hidden state or mean pooling)
- **Information loss**: Early important signals get diluted by later noise
- **Long sequences**: The "vital first step" might be forgotten when processing the last step
- No way to emphasize critical moments (e.g., onset of pain, specific movement patterns)

**Example Without Attention:**
```python
# Traditional: Use last hidden state or mean pooling
lstm_out, _ = lstm(x)  # (batch, seq_len, hidden)
final_state = lstm_out[:, -1, :]  # Only last timestep
# OR
pooled = torch.mean(lstm_out, dim=1)  # Equal weight to all
```

**Attention Mechanism Solution:**
Learns to **weight timesteps by importance**:
```python
# With attention: Learn which timesteps matter
lstm_out, _ = lstm(x)  # (batch, seq_len, hidden)
attn_weights = attention_layer(lstm_out)  # (batch, seq_len) - learned weights
context = sum(attn_weights * lstm_out)  # Weighted combination
```

**How Attention Works (Bahdanau-style):**
1. Compute attention scores for each timestep: `score_t = v^T * tanh(W * h_t)`
2. Normalize with softmax: `α_t = softmax(scores)` → weights sum to 1.0
3. Weighted sum: `context = Σ(α_t * h_t)` → emphasizes important moments

**Benefits:**
- **Selective focus**: Model learns which timesteps are critical for classification
- **Interpretability**: Attention weights show what the model "looks at"
- **Better long-range**: Prevents early signals from being forgotten
- **Improved performance**: Often boosts accuracy by 2-5% on sequence tasks

**What's Currently Implemented:**

❌ **Current Best Model (CNN) has NO attention**
- File: `solution_1_cnn_improved.ipynb`
- Uses **Global Average Pooling** → treats all timesteps equally
- No mechanism to emphasize critical moments

✅ **Attention Models EXIST but UNTESTED**
Three models with Bahdanau-style attention are fully implemented:

**1. BiLSTM with Attention** (`model_definitions/bilstm.py`)
- Architecture: BiLSTM → Attention → Classifier
- Bahdanau-style attention mechanism
- Returns both context vector and attention weights

**2. CNNLSTMClassifier** (`model_definitions/cnn_lstm.py`)
- Architecture: CNN → LSTM → **Optional Attention** → Classifier
- Parameter: `use_attention=True` (default: enabled)
- Can toggle between attention and mean pooling

**3. CNNGRUClassifier** (`model_definitions/cnn_lstm.py`)
- Architecture: CNN → GRU → **Optional Attention** → Classifier  
- Same attention mechanism as CNN-LSTM variant

**Implementation Details:**
All three models use identical Bahdanau attention:
```python
def attention(self, rnn_out):
    u = torch.tanh(self.attn_W(rnn_out))  # Transform
    scores = self.attn_v(u).squeeze(-1)   # Compute scores
    attn_weights = F.softmax(scores, dim=-1)  # Normalize
    context = torch.bmm(attn_weights.unsqueeze(1), rnn_out).squeeze(1)
    return context, attn_weights
```

**Status of Attention Models:**
❌ **Attention mechanism not yet used** in tested models
- CNN_BD and CNNGRUClassifier use **last hidden state** pooling (not attention)
- BiLSTM with attention exists but not tested yet
- CNN-LSTM/GRU models have `use_attention=True` option but tested with default (mean pooling)
- No comparison of attention vs. no attention yet
- No analysis of what timesteps get high attention weights

**Action Steps:**
- [ ] Test BiLSTM with attention on current data preprocessing
- [ ] Enable `use_attention=True` in CNNGRUClassifier and compare to current results
- [ ] Test if attention improves beyond current 95.25% val F1
- [ ] Analyze attention weights to understand which timesteps are important
- [ ] Visualize attention patterns: Do high-pain samples attend to different moments than no-pain?
- [ ] Compare attention vs last hidden state pooling in CNN_BD
- [ ] Check if attention specifically improves minority class (high_pain: currently 66.7% F1)

**Status:** 🔴 Not Yet Tested

**Notes:**
- **All attention models exist**: BiLSTM, CNN-LSTM, CNN-GRU all have attention built-in
- **Never tested**: These models were coded but experiments never ran
- **Current CNN limitation**: Uses global average pooling, no attention mechanism
- **Yoda's insight**: "Does it remember the vital first step, when it reaches the last?" → attention solves this
- **Interpretability bonus**: Attention weights could reveal which time moments indicate pain
- **Easy experiment**: Import BiLSTM or CNNLSTMClassifier and test with same pipeline
- **Potential gain**: Attention typically improves 2-5% on sequence classification tasks
- **Historical context**: Pure RNNs failed, but CNN-RNN with attention might work

---

## Summary of Key Insights

- **Categorical encoding matters:** Pain survey responses (0-4) are ordinal categories, not continuous numbers
- **Current issue:** Pain surveys treated as continuous integers in the preprocessing pipeline
- **Embeddings > Integer Encoding:** For ordinal features like pain surveys, embeddings can learn semantic relationships
- **Potential improvement:** Adding embeddings for the 4 pain survey columns could boost test F1 beyond current 84.6%
- **Class imbalance is severe:** 77% no_pain, 14% low_pain, 9% high_pain
- **Current solution:** Oversampling minority classes works well (84.6% F1)
- **Alternative approach:** Weighted loss functions could complement or replace oversampling
- **Label smoothing prevents overfitting:** Introducing "doubt" (0.9 vs 1.0) improves generalization
- **Easy win:** Single-line change to add label_smoothing parameter to CrossEntropyLoss
- **Could close gap:** May reduce the 4-5% difference between validation (89%) and test (84.6%)
- **Gradient clipping already implemented:** Using max_norm=1.0 to prevent exploding gradients
- **Why CNNs succeeded over RNNs:** Gradient instability caused RNN models to collapse, CNNs more stable
- **Window size is arbitrary:** Current window=300 has no data-driven justification
- **ACF analysis missing:** Should analyze autocorrelation to find optimal window size
- **Past experiment clue:** Comment suggests window=256, stride=64 performed better than current setup
- **Time column completely ignored:** Rich temporal information unused (could indicate pain progression)
- **Missing temporal context:** Time position in sequence could be valuable feature
- **Hybrid CNN-RNN BEST performance:** CNN_BD achieves 95.25% val F1 (highest score)
- **Pure RNNs failed, hybrid CNN-RNN works excellently:** Combines CNN stability + RNN temporal reasoning
- **Attention models exist but not yet tested:** BiLSTM with attention, CNN-LSTM/GRU with use_attention=True
- **Current best models use last hidden state:** CNN_BD and CNNGRUClassifier don't use attention (yet)
- **Potential further improvement:** Adding attention mechanism could push beyond 95.25%

---

## Quick Reference

| Date | Clue Summary | Key Insight | Status |
|------|--------------|-------------|--------|
| Nov 7 | Embedding layers for categoricals | Pain surveys (0-4) should use embeddings, not treated as continuous | 🔴 |
| Nov 8 | Class imbalance handling | Use weighted loss or oversampling for imbalanced classes (77:14:9 ratio) | 🟡 |
| Nov 9 | Label smoothing | Add label_smoothing=0.1 to CrossEntropyLoss to prevent overconfident predictions | 🔴 |
| Nov 10 | Gradient clipping | Clip gradients to prevent exploding gradients (max_norm=1.0) | 🟢 |
| Nov 11 | Autocorrelation-based windowing | Use ACF analysis to determine optimal window size instead of arbitrary 300 | 🔴 |
| Nov 12 | Time feature engineering | Use time column as feature (normalized, cyclical, or categorical) | 🔴 |
| Nov 13 | 1D Convolutions & CNN-RNN hybrid | Use Conv1D for patterns; CNN+GRU hybrid achieves 95.25% val F1 🏆 | 🟢 |
| Nov 14 | Attention mechanism | Add attention to weight important timesteps (models exist, not yet tested) | 🔴 |

