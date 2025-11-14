## CNN_BD
CNN_BD Best model restored from epoch 87 with val_f1 0.9525
### Hyperparameters
WINDOW_SIZE = 300
STRIDE = 150
BATCH_SIZE = 64
lr=5e-4,           # Learning rate
    weight_decay=1e-4 
    num_classes=3, hidden_size=256, dropout=0.3, bidirectional=True, l1=0, l2=0



## CNN1DClassifier
### Epochs
Epoch   1/1500 | Train: Loss=0.9348, F1 Score=0.2405 | Val: Loss=1.1008, F1 Score=0.0152
Epoch  50/1500 | Train: Loss=0.0245, F1 Score=0.9703 | Val: Loss=0.9175, F1 Score=0.8602
Epoch 100/1500 | Train: Loss=0.0163, F1 Score=0.9764 | Val: Loss=1.2109, F1 Score=0.9303
Epoch 150/1500 | Train: Loss=0.0123, F1 Score=0.9902 | Val: Loss=1.2313, F1 Score=0.9234
Epoch 200/1500 | Train: Loss=0.0087, F1 Score=0.9922 | Val: Loss=1.5114, F1 Score=0.8398
Early stopping triggered after 230 epochs.
Best model restored from epoch 180 with val_f1 0.9522

### Results
  Initial val F1: 0.0152
  Final val F1:   0.9212
  Best val F1:    0.9522
  Improvement:    +0.9371

### Classification Report
              precision    recall  f1-score   support

     no_pain     0.9490    0.9894    0.9688        94
    low_pain     1.0000    0.9615    0.9804        26
   high_pain     0.8889    0.6667    0.7619        12

    accuracy                         0.9545       132
   macro avg     0.9460    0.8725    0.9037       132
weighted avg     0.9536    0.9545    0.9522       132

### Hyperparameters
BATCH SIZE = 64
num_filters=[64, 128, 256],
kernel_sizes=[5, 5, 3],
dropout_rate=0.4
L1 = 0
L2 = 0
weight Decay = 1e-4
LR = 5e-4

## CNNGRUClassifier
### Epochs
Epoch   1/1500 | Train: Loss=1.0284, F1 Score=0.3011 | Val: Loss=1.0951, F1 Score=0.0661
Epoch  10/1500 | Train: Loss=0.7479, F1 Score=0.2521 | Val: Loss=1.1498, F1 Score=0.0470
Epoch  20/1500 | Train: Loss=0.3183, F1 Score=0.5492 | Val: Loss=0.8620, F1 Score=0.1976
Epoch  30/1500 | Train: Loss=0.1907, F1 Score=0.6367 | Val: Loss=0.9097, F1 Score=0.6995
Epoch  40/1500 | Train: Loss=0.1696, F1 Score=0.8163 | Val: Loss=0.9605, F1 Score=0.7996
Epoch  50/1500 | Train: Loss=0.0974, F1 Score=0.8701 | Val: Loss=1.0212, F1 Score=0.8209
Epoch  60/1500 | Train: Loss=0.0649, F1 Score=0.9121 | Val: Loss=1.2103, F1 Score=0.8061
Epoch  70/1500 | Train: Loss=0.0534, F1 Score=0.9127 | Val: Loss=1.3422, F1 Score=0.7956
Epoch  80/1500 | Train: Loss=0.0515, F1 Score=0.9301 | Val: Loss=1.2663, F1 Score=0.9143
Epoch  90/1500 | Train: Loss=0.0358, F1 Score=0.9584 | Val: Loss=1.3726, F1 Score=0.9080
Epoch 100/1500 | Train: Loss=0.0641, F1 Score=0.9468 | Val: Loss=1.3905, F1 Score=0.8996
Epoch 110/1500 | Train: Loss=0.0354, F1 Score=0.9665 | Val: Loss=1.4700, F1 Score=0.8626
Epoch 120/1500 | Train: Loss=0.0515, F1 Score=0.9647 | Val: Loss=1.4984, F1 Score=0.8796
Epoch 130/1500 | Train: Loss=0.0639, F1 Score=0.9258 | Val: Loss=1.4789, F1 Score=0.7930
Epoch 140/1500 | Train: Loss=0.0277, F1 Score=0.9686 | Val: Loss=1.5369, F1 Score=0.9173
Epoch 150/1500 | Train: Loss=0.0243, F1 Score=0.9624 | Val: Loss=1.5235, F1 Score=0.8976
Epoch 160/1500 | Train: Loss=0.0153, F1 Score=0.9843 | Val: Loss=1.5737, F1 Score=0.9155
Epoch 170/1500 | Train: Loss=0.0158, F1 Score=0.9863 | Val: Loss=1.8936, F1 Score=0.9006
Epoch 180/1500 | Train: Loss=0.0262, F1 Score=0.9765 | Val: Loss=1.4567, F1 Score=0.8986
Early stopping triggered after 181 epochs.
Best model restored from epoch 131 with val_f1 0.9310

### Results
  Initial val F1: 0.0661
  Final val F1:   0.8591
  Best val F1:    0.9310
  Improvement:    +0.8648

### Classification Report
              precision    recall  f1-score   support

     no_pain     0.9588    0.9894    0.9738        94
    low_pain     0.9565    0.8462    0.8980        26
   high_pain     0.6667    0.6667    0.6667        12

    accuracy                         0.9318       132
   macro avg     0.8607    0.8341    0.8461       132
weighted avg     0.9318    0.9318    0.9310       132

### Hyperparameters
BATCH SIZE = 64
num_filters=[64, 128, 256],
kernel_sizes=[5, 5, 3],
dropout_rate=0.4
L1 = 0
L2 = 0
weight Decay = 1e-4
LR = 1e-4