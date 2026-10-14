

\# Lab 1 — Machine Learning Model Benchmarking



\## Goal



The goal of this laboratory was to compare machine-learning models in terms of predictive performance, training time, inference latency, and resource requirements. The experiment focused on the engineering aspects of training and evaluating models in a reproducible environment.



\## Method



The Breast Cancer Wisconsin dataset included in scikit-learn was used. It contains 569 samples and 30 numerical features. No manual dataset download was required because the dataset is bundled with scikit-learn.



The data was split into training and test sets using a 70/30 ratio. Stratified sampling was used, and random seed 42 was specified to make the experiment reproducible. The training set contained 398 samples, while the test set contained 171 samples.



Two models were evaluated:



1\. Logistic Regression with StandardScaler and a maximum of 1,000 iterations.

2\. Random Forest with 100 decision trees.



Training time was measured after one warm-up run. Each model was trained five times, and the median training time was reported.



Single-sample inference latency was measured over 100 predictions after one warm-up prediction. The median latency was reported.



Accuracy and F1 score were calculated on the held-out test set. The trained models were saved using joblib. Their serialized file sizes were measured. Process memory was recorded after training, together with the change from the pre-training measurement.



The experiment was executed using Python 3.11.8 on the CPU. The installed package versions are documented in `results/versions.txt`.



\## Results



| Metric | Logistic Regression | Random Forest |

|---|---:|---:|

| Test accuracy | 0.9883 | 0.9357 |

| F1 score | 0.9907 | 0.9488 |

| Median training time | 0.006856 s | 0.119424 s |

| Median inference latency | 0.520350 ms | 3.449950 ms |

| Saved model size | 3.17 KiB | 285.01 KiB |

| Process memory after training | 157.12 MiB | 157.53 MiB |



The comparison plot is saved in `results/model\_comparison.png`. Detailed measurements are stored in the CSV files in the results directory.



\## Conclusions



1\. \*\*Predictive performance:\*\* Logistic Regression achieved an accuracy of 98.83% and an F1 score of 99.07%, outperforming Random Forest on this test split. Random Forest achieved an accuracy of 93.57% and an F1 score of 94.88%.



2\. \*\*Performance:\*\* Logistic Regression had a median training time of approximately 6.86 milliseconds and median single-sample inference latency of approximately 0.52 milliseconds. Random Forest required approximately 119.42 milliseconds for training and 3.45 milliseconds for single-sample inference. In this experiment, Logistic Regression was faster for both operations.



3\. \*\*Resource requirements:\*\* The serialized Logistic Regression model was approximately 3.17 KiB, compared with 285.01 KiB for Random Forest. Therefore, Logistic Regression had a substantially smaller saved model file. The recorded process-memory values include the Python interpreter and loaded libraries; they do not represent memory required by the model alone. Deployment suitability must be evaluated against the actual memory, model-size, and latency limits specified for each target environment.



