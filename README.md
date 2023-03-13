# JIT-defect-prediction-Android-apps
Just-In-Time defect prediction for Android app for the paper `Just-in-Time Defect Prediction for Mobile Applications: 
Using Shallow or Deep Learning?` by Raymon van Dinter, Cagatay Catal, Görkem Giray, and Bedir Tekinerdogan.

# Datasets
The datasets can be retrieved from:
`G. Catolino, D. Di Nucci, and F. Ferrucci. (2019) Cross-project just-in-time bug prediction for mobile app: An empirical assessment - online appendix https://figshare.com/s/9a075be3e1fb64f76b48.`


## How to use
1. Clone this repository and inspect [main.py](https://github.com/rvdinter/JIT-defect-prediction-Android-apps/blob/main/main.py)
2. Install scikit-learn==0.24.2, pytorch-tabnet==3.1.1, pytorch==1.9.0, and xgboost=1.4.2.
3. Run main.py

The folder `metrics/` contains the sklearn and TabNet metrics for evaluating the model. The folder `models/` contains each of the models from the paper and a function to train the model for a single fold. `DatasetLoader` is used for loading and preprocessing the dataset. `DefectDataset` is a PyTorch data object used by the MLP model.
