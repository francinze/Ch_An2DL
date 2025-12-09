# Ch1

## Instructions

The following list extracts and organizes all step-by-step instructions and pipeline procedures detailed in the sources, grouped by context.

### I. Standard Machine Learning Pipeline (Centralized)

The traditional process for training a centralized Machine Learning model involves sequential steps:

1.  **Data Collection:** A dataset of interest is collected, usually by gathering data from each user/client into a single, large dataset for the task.
2.  **Model Training:** An ML model is trained on this consolidated data to learn how to predict the output label given specific inputs.
3.  **Inference:** Once the final model is ready, it is used in inference mode to make predictions on new data coming from a device.

### II. Machine Learning Pipeline for Time Series/Signals

The process of developing ML models from time series or signal data involves several critical steps:

1.  **Task Definition:** Defining the objective, such as **Classification** (binary or multi-label) or **Regression**.
2.  **Data Preparation:** Addressing data quality issues, which includes:
    *   Handling abnormalities (e.g., outliers or clearly incorrect values).
    *   Handling missing values (e.g., NaNs due to poor sensor positioning).
    *   Data/Signal Processing.
    *   **Feature Extraction** and Dimensionality Reduction (extracting relevant patterns from trends).
3.  **Model Training:** Selecting appropriate models (e.g., Decision Trees, Deep Learning methods like RNN, GRU, or LSTM).
    *   *Optimization:* Model configuration (e.g., determining the best hyperparameters and feature selection) can be optimized using techniques like **Optuna**.
4.  **Model Evaluation:** Splitting data (Train/Test split), using cross-validation, and measuring performance using metrics like Accuracy or Mean Squared Error.

### III. Pipelines for Handling Raw Time Series Data

When dealing with raw time series data, two main alternatives are available:

**Alternative 1: Feature Extraction (Using Classical Tabular ML Models)**

1.  **Extract Features:** Calculate summary statistics or derived features from the time series (e.g., Mean, Standard Deviation, Percentiles, or Fast Fourier Transform).
2.  **Build Tabular Dataset:** Transform the time series into a single row of tabular features associated with the label.
3.  **Apply Models:** Standard models (like Random Forest) can then be applied.

### IV. Horizontal Federated Learning (HFL) Implementations

#### A. HFL with Neural Networks (Iterative Process)

When using Neural Networks in an HFL setup, a sequence of steps is followed for training:

1.  Weights (or only gradients) are encrypted by each client and sent to the central server.
2.  The server aggregates these models using a pre-defined strategy, such as FedAvg.
3.  The resulting aggregated model is then sent back to the individual clients.
4.  The clients can evaluate the performance of this aggregated model using their own local data.
*(This process is iterative, involving multiple rounds until a final result is achieved.)*

#### B. Federated Random Forest (Single-Shot Aggregation)

This method uses the simplest aggregation strategy, performed in a single round:

1.  If the target is an $N$-tree global model, each client trains a fraction of those decision trees (e.g., 50 decision trees if targeting 200 total trees with four clients).
2.  The client serializes the trained model into bytes before sending it to the server.
3.  The client also sends the length of the training set for weighted aggregation.
4.  The server aggregates by taking the lists of decision trees from all clients and concatenating (extending) them into a single global model.

#### C. XGBoost Aggregation (Iterative Boosting)

XGBoost uses iterative boosting, where subsequent decision trees are trained to correct errors.

*   **General Iteration Cycle:**
    *   Clients receive the previous aggregated model.
    *   Clients compute the gradients of the errors.
    *   Clients train an *additional* decision tree to correct those errors (boosting).

*   **XGBoost Bagging:** Clients contribute partial models (e.g., models containing 5 DTs). Requests for new Boosting rounds are made and repeated until the final XGBoost objective is achieved.
*   **XGBoost Cyclic:** This approach repeats each boosting round using a different client until the final objective is reached. Clients send partial models, often consisting of just 1 DT.

#### D. Histogram-based Aggregation (Training Decision Trees Node-by-Node)

This method attempts to emulate centralized training by training the model node-by-node:

1.  During the training of a Decision Tree (DT), the central server requests **split histograms** from the clients for the current node (e.g., the root node).
2.  The server sums the received histograms.
3.  The server uses the result to choose the best split.
4.  This process is repeated for every node of every DT until the objective model is complete.

### V. Goal for Data Augmentation (Addressing Imbalance)

When applying data augmentation to resolve Class Imbalance and Size Imbalance:

1.  Clients should adjust their local dataset distributions to match the **overall class label distribution** of the entire federated network.

## Rules / Requirements

### I. Constraints on Wearable Data Collection and Use

*   **Goal of Wearables:** The objective of wearables is to enable **continuous and context-rich monitoring in everyday life**.
*   **Scope Limitation:** Wearable devices typically provide information focused on general well-being rather than clinical diagnosis.
*   **Data Type Constraint:** Wearable devices typically provide **time series data** for well-being purposes, while raw signals are usually accessible only from medical devices.
*   **Data Synchronization:** Data is stored in the Garmin cloud and is downloaded only once the device syncs.
*   **Wearing Requirement for HRV:** **Continuous wearing** (day and night) improves personalization, especially for Heart Rate Variability (HRV) measurements, which rely heavily on sleep time data.
*   **Stress Measurement Constraint:** Stress levels are generally **not measured during workouts**.
*   **Default Activity Goal:** The default weekly goal for physical activity monitoring is 150 minutes.

### II. Requirements and Conditions for the Federated Learning (FL) Paradigm

*   **Privacy Requirement:** Sensitive user data **cannot be shared** due to new regulations, such as GDPR.
*   **Core FL Constraint:** To maintain privacy, only the updated model weights (parameters) are exchanged and aggregated by a central server, instead of sharing raw data.
*   **Performance Condition:** Federated models rarely achieve the performance level of a centralized training approach, representing a **trade-off between privacy and model performance**.
*   **HFL Constraint:** The implementation of Horizontal Federated Learning (HFL) is appropriate when different data sources share the same features space.
*   **Divergence Condition:** Local models may diverge significantly when client data distributions are too different, which can cause aggregated contributions to cancel each other out, leading the model to become stuck or preventing effective learning.

### III. Rules and Conditions for Addressing Data Distribution Imbalance

*   **Goal for Data Augmentation:** Clients should adjust their local dataset distributions to match the **overall class label distribution** of the entire federated network.
*   **Augmentation Constraint (Noise/Transformation):** When generating "fake" samples via adding noise or applying geometric transformations, the noise or transformation must **not change the data's true class** or label value.
*   **Model Divergence Condition (Feature Imbalance):** If input features are distributed drastically differently across clients (e.g., different age groups in hospital scans), it can result in drastically different local models.

### IV. Requirements and Conditions for Model Training and Evaluation

*   **Data Preparation Requirement:** The data preparation step must include addressing data quality issues, such as handling abnormalities (outliers) and handling missing values (e.g., NaNs due to poor sensor positioning).
*   **Model Optimization Requirement:** Model configuration (e.g., determining the best hyperparameters and feature selection) can be optimized using techniques like Optuna.
*   **Regularization Requirement:** **Regularization techniques** must be used to avoid or reduce overfitting.
*   **Overfitting Condition:** Overfitting occurs when a model becomes too specialized, fitting the training data perfectly but performing poorly on new data.
*   **Overfitting Detection:** If performance on the training data continues to improve while validation performance drops, it indicates overfitting.
*   **Deep Learning Requirement:** Neural networks are complex and highly prone to overfitting, thus requiring **substantial data and careful regularization techniques**.
*   **Test Set Rule:** The Test Set (e.g., 20% of the data) must be used **only once at the end** to provide the final, unbiased performance measure of the chosen model.

### V. Specific Model Aggregation Constraints

*   **XGBoost Boosting Requirement:** XGBoost utilizes iterative boosting, where subsequent decision trees must be trained to **correct the errors** of previous aggregated models.
*   **Federated Random Forest Communication Requirement:** For single-shot aggregation, the client must send the serialized trained model into bytes and also send the length of the training set for weighted aggregation.

### VI. Project-Specific Requirements (Sleep Quality Prediction)

*   **Project Focus:** The challenge uses anonymized Garmin data, primarily focusing on **sleep data**.
*   **Expected Data Condition:** The data for the challenge is expected to be unbalanced.
*   **Task Definition:** The task is defined as **regression**, where the model predicts a continuous sleep quality indicator (0 to 1).
*   **Evaluation Metric:** The main metric used for regression evaluation is typically **Mean Square Error (MSE)**.
*   **Final Step Requirement:** The overall global model must be stored (e.g., in `.pkl` or `.pth` files) after training and then used to predict on a separate test set (X test) for evaluation.

### VII. Rules and Calculations for Wearable Data Metrics

*   **Vigorous Activity Calculation:** Vigorous minutes count **double** to reflect their higher physiological benefit.
*   **Sleep Tracking Inputs:** Sleep tracking estimates stages by combining **Heart Rate (HR), Heart Rate Variability (HRV), and body movement**.
*   **Sleep Score Calculation Components:** The Sleep Score (0–100) is calculated based on:
    1.  The balance of sleep structure (Light, Deep, REM phases).
    2.  The previous day's stress level (HRV analysis of sympathetic vs. parasympathetic activity).
    3.  The frequency and duration of awakenings.

### VIII. Advanced Requirements for Addressing Federated Learning Imbalance

*   **Transfer Learning Requirement:** To mitigate divergence caused by feature imbalance, the HFL process should start with a **pre-trained model** (trained on sufficiently large and generic datasets).
*   **Multitask Learning Requirement:** To learn more general and robust representations and address feature imbalance, the model should be trained to **optimize multiple tasks simultaneously**.
*   **Data Augmentation Instruction (Interpolation):** New data points can be created by **averaging (interpolating) couples of existing samples of the same class**.

### IX. Instructions for Model Training, Evaluation, and Regularization

*   **Learning Mechanism:** A model learns by minimizing a quantity called the **loss function** ($L$).
*   **Regularization Method (Loss Function):** A regularization technique can be implemented by changing the loss function to include a **penalty measure for weights that are too large**.
*   **Hyperparameter Selection Rule:** The **Validation Set (e.g., 10%)** must be used to choose the best hyperparameters.
*   **Test Set Standard Size:** The Test Set is typically **20%** of the data.
*   **Deep Learning Regularization (Early Stopping):** Training should be halted (early stopping) when the **validation loss or accuracy stops improving or starts worsening** across epochs.

### X. Instructions for Time Series Feature Extraction

*   **Feature Calculation Requirement:** When using Feature Extraction (Alternative 1) to transform time series into tabular data, features must be calculated using summary statistics or derived functions, such as **Mean, Standard Deviation, Percentiles, or Fast Fourier Transform**.

### XI. Instructions for Interpretability and Explainability

*   **Feature Contribution Analysis:** Use **SHAP values** to quantify precisely how much each feature contributes to the model's output deviation.
*   **Counterfactual Explanation:** Use **Diverse Counterfactual Explanations (DiCE)** to find plausible instances ($x'$) that show what input changes are required to shift the model's prediction to a desired outcome ($\hat{y}_{cf}$).
