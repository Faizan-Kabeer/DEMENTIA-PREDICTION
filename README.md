# DEMENTIA-PREDICTION

### Stratified K-Fold Cross-Validation
Stratified K-Fold Cross-Validation is a validation technique designed to address class imbalance issues inherent in traditional K-Fold cross-validation methods. This enhanced approach ensures that each fold maintains the same proportion of samples from each target class as the complete dataset, providing more reliable and representative model evaluation metrics.

#### Algorithm Mechanics
The stratified sampling process follows these key steps:

1. Stratum Identification: The algorithm first identifies target classes and divides the dataset into subgroups where each subgroup corresponds to a specific class label.

2. Proportional Distribution: Within each subgroup, samples are split proportionally across all k folds.

3. Fold Composition: Each fold receives approximately the same percentage of samples from each subgroup as exists in the entire dataset.

### Gray Wolf Optimization (GWO)
Gray Wolf Optimization (GWO) is a nature-inspired metaheuristic algorithm that simulates the leadership hierarchy and hunting mechanism of gray wolves in nature. It is designed for solving complex optimization problems, especially where the search space is vast and not well defined.

