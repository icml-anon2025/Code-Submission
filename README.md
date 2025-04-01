

## Learning Accurate Models on Incomplete Data with Minimal Imputation

Missing data often exists in real-world datasets, requiring significant time and effort for imputation to learn accurate machine learning (ML) models. In this paper, we demonstrate
that imputing all missing values is not always necessary to
achieve an accurate ML model. We introduce the concept of
minimal data imputation, which ensures accurate ML models
trained over the imputed dataset. Implementing minimal imputation guarantees both minimal imputation effort and optimal ML models. We propose algorithms to find exact and approximate minimal imputation for various ML models. Our
extensive experiments indicate that our proposed algorithms
significantly reduce the time and effort required for data imputation
-----------------------------------------------------------------------------------------

Experiments are ordered first at the level of ML problem (SVM, Regression) and then at the level of dataset. Each dataset has its own file for running the experiments.

## 1. Real-World-Dataset with Inherent Missing Values

Repo includes 3 (SVM) + 2 (Linear Regression) real-world datasets with inherent missing values.
- Every dataset has its specific file for running the code.
- Additionally datasets can be found in their original source, all are cited in the paper.
- Active Clean is based on original implementation: [https://activeclean.github.io/](https://activeclean.github.io/)
- KNNImputer is implemented based on sklearn: [https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
- MeanImputer is implemented based on sklearn: [https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

-For Running Experiments for instance for water potability. python3 SVM/Real-World-Minimal-Imputation-Water.py.


