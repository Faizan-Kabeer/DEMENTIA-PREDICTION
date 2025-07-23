## OUTPUT
```
--- FOLD 1/5 ---
Running GWO for hyperparameter optimization...
2025/07/22 11:19:24 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: Solving single objective optimization problem.
2025/07/22 11:19:27 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: >>>Problem: P, Epoch: 1, Current best: -0.9782608695652174, Global best: -0.9782608695652174, Runtime: 1.46061 seconds
    .
    .
    .
2025/07/22 11:22:00 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: >>>Problem: P, Epoch: 50, Current best: -0.9782608695652174, Global best: -0.9782608695652174, Runtime: 3.56184 seconds

GWO found best params: Neurons=(68, 48, 20), LR=0.0067
Training final model on full resampled data with optimal parameters...
Fold 1 Accuracy: 0.8592


--- FOLD 2/5 ---
Running GWO for hyperparameter optimization...
2025/07/22 11:22:03 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: Solving single objective optimization problem.
2025/07/22 11:22:06 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: >>>Problem: P, Epoch: 1, Current best: -0.9782608695652174, Global best: -0.9782608695652174, Runtime: 1.19169 seconds
    .
    .
    .
2025/07/22 11:22:53 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: >>>Problem: P, Epoch: 50, Current best: -0.9782608695652174, Global best: -0.9782608695652174, Runtime: 0.85880 seconds

GWO found best params: Neurons=(56, 81, 82), LR=0.0054
Training final model on full resampled data with optimal parameters...
Fold 2 Accuracy: 0.9577


--- FOLD 3/5 ---
Running GWO for hyperparameter optimization...
2025/07/22 11:22:54 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: Solving single objective optimization problem.
2025/07/22 11:22:57 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: >>>Problem: P, Epoch: 1, Current best: -0.9891304347826086, Global best: -0.9891304347826086, Runtime: 1.47915 seconds
    .
    .
    .
2025/07/22 11:24:51 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: >>>Problem: P, Epoch: 50, Current best: -0.9891304347826086, Global best: -0.9891304347826086, Runtime: 3.00940 seconds

GWO found best params: Neurons=(170, 177, 129), LR=0.0082
Training final model on full resampled data with optimal parameters...
Fold 3 Accuracy: 0.8873


--- FOLD 4/5 ---
Running GWO for hyperparameter optimization...
2025/07/22 11:24:52 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: Solving single objective optimization problem.
2025/07/22 11:24:56 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: >>>Problem: P, Epoch: 1, Current best: -0.9891304347826086, Global best: -0.9891304347826086, Runtime: 1.94670 seconds
    .
    .
    .
2025/07/22 11:26:13 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: >>>Problem: P, Epoch: 50, Current best: -0.9891304347826086, Global best: -0.9891304347826086, Runtime: 1.56630 seconds

GWO found best params: Neurons=(47, 200, 19), LR=0.0100
Training final model on full resampled data with optimal parameters...
Fold 4 Accuracy: 0.9296


--- FOLD 5/5 ---
Running GWO for hyperparameter optimization...
2025/07/22 11:26:14 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: Solving single objective optimization problem.
2025/07/22 11:26:22 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: >>>Problem: P, Epoch: 1, Current best: -0.9782608695652174, Global best: -0.9782608695652174, Runtime: 5.35971 seconds
    .
    .
    .
2025/07/22 11:30:31 PM, INFO, mealpy.swarm_based.GWO.OriginalGWO: >>>Problem: P, Epoch: 50, Current best: -0.9782608695652174, Global best: -0.9782608695652174, Runtime: 1.95242 seconds

GWO found best params: Neurons=(199, 166, 153), LR=0.0067
Training final model on full resampled data with optimal parameters...
Fold 5 Accuracy: 0.9000


--- FINAL RESULTS ---
Mean Cross-Validated Accuracy: 0.9068
Standard Deviation of Accuracy: 0.0341

Confusion Matrices for each fold:
 Fold 1:
[[ 1  2  4]
 [ 2 24  0]
 [ 1  1 36]]
--------------------
 Fold 2:
[[ 4  0  3]
 [ 0 26  0]
 [ 0  0 38]]
--------------------
 Fold 3:
[[ 4  1  3]
 [ 2 23  0]
 [ 1  1 36]]
--------------------
 Fold 4:
[[ 4  1  3]
 [ 0 25  0]
 [ 1  0 37]]
--------------------
 Fold 5:
[[ 2  2  3]
 [ 2 23  0]
 [ 0  0 38]]
--------------------
```
BEST PARAMETERS FOUND FOUND: Neurons=(56, 81, 82), LR=0.0054
