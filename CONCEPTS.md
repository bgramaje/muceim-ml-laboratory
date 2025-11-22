# Deep Learning Laboratory Concepts

## 1. Baseline Model Architecture (128 -> 128 -> 64)
The baseline model uses a specific "funnel" architecture defined in the teacher's reference:
- **Structure**: Input (784) -> Dense(128) -> Dense(128) -> Dense(64) -> Output(10).
- **Why this choice?**:
  - **Capacity**: Keeping the second layer at 128 (instead of dropping to 64 immediately) preserves more information deeper into the network.
  - **Bottleneck Prevention**: It avoids creating an information bottleneck too early, which is helpful for Fashion MNIST as it is more complex than simple digits.
  - **Reference**: This strictly follows the `04- TF-Keras` reference notebook provided.

## 2. Baseline Configuration (Loss & Optimizer)
- **`loss='categorical_crossentropy'`**:
  - Used for **multi-class classification** when labels are **one-hot encoded** (vectors like `[0, 1, 0...]`).
  - If labels were integers (0, 1, 2...), we would use `sparse_categorical_crossentropy`.
- **`optimizer='sgd'` (Stochastic Gradient Descent)**:
  - The "standard" or vanilla optimizer.
  - **Purpose as Baseline**: It is chosen specifically because it is basic. It provides a modest performance benchmark, allowing subsequent experiments with advanced optimizers (like Adam) to demonstrate clear improvements.

## 3. Training Log Attributes
Explanation of a training log line: `Epoch 1/100 ... 6s 3ms/step - accuracy: 0.12 - loss: 2.30 - val_accuracy: 0.24 - val_loss: 2.27`

- **`Epoch 1/100`**: The current round of training. The model has seen the entire dataset once.
- **`1500/1500`**: Number of batches processed (48,000 training images / 32 batch size = 1,500 steps).
- **`6s 3ms/step`**: Total time for the epoch (6s) and average time per batch (3ms).
- **`accuracy` / `loss`**: Metrics on the **training data** (data the model is currently learning from).
- **`val_accuracy` / `val_loss`**: Metrics on the **validation data** (unseen data).
  - **Key Insight**: `val_loss` is the most critical metric for detecting overfitting. If `val_loss` rises while `loss` falls, the model is memorizing, not learning.

## 4. Early Stopping
`EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)`

- **`monitor='val_accuracy'`**: The metric to watch. We want this to go up.
- **`patience=5`**: The "tolerance." If the metric doesn't improve for 5 consecutive epochs, training stops.
- **`restore_best_weights=True`**: The "rewind" feature.
  - If training stops at Epoch 15 because it hasn't improved since Epoch 10, this ensures the model reverts to the weights from **Epoch 10**.
  - Without this, you would be left with the "overtrained" or stagnant weights from Epoch 15.

## 5. Generalization Errors
### Overfitting (High Variance)
- **Definition**: The model learns the training data *too* well, including noise and specific details that don't generalize to new data. It is "memorizing" rather than "learning."
- **Symptoms**:
  - **Training Loss**: Continues to decrease (improves).
  - **Validation Loss**: Starts to increase (worsens).
  - **Visual**: The two loss curves diverge like an opening alligator mouth.
- **Solution**: Add regularization (Dropout, L2), simplify the model, or use Early Stopping.

### Underfitting (High Bias)
- **Definition**: The model is too simple to capture the underlying patterns of the data. It fails to learn the relationship between inputs and outputs.
- **Symptoms**:
  - **Training Loss**: Remains high (poor performance).
  - **Validation Loss**: Remains high (poor performance).
  - **Visual**: Both curves flatten out at a high error rate and never go down significantly.
- **Solution**: Increase model complexity (more layers/neurons), train longer, or remove excessive regularization.

## 6. Baseline Results (Recorded)
Results obtained from the initial run of the baseline model:

- **Final Epoch (100/100)**:
  - Training Accuracy: **88.43%**
  - Training Loss: **0.3234**
  - Validation Accuracy: **87.40%**
  - Validation Loss: **0.3525**

- **Final Test Set Evaluation** (`model_baseline.evaluate`):
  - **Test Accuracy**: **86.08%**
  - **Test Loss**: **0.3936**

*Note: The slight drop from Validation Accuracy (87.4%) to Test Accuracy (86.08%) is normal, as the test set is completely unseen data.*
