import torch
import pandas as pd
from os import listdir
from sklearn.model_selection import RepeatedStratifiedKFold
from datasetloader import DatasetLoader
# Sampling methods
from imblearn.over_sampling import SMOTE, RandomOverSampler, SMOTEN, SVMSMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
# Load model trainers
from models.mlp_fold import mlp_fold
from models.tabnet_fold import tabnet_fold
from models.xgboost_fold import xgboost_fold

# Get the paths of all datasets
path = 'appendix/DATASET'
csvfiles = listdir(path)

# Initialize KFold and samplers
rskf = RepeatedStratifiedKFold(n_splits=2, random_state=42, n_repeats=25)
samplers = [RandomOverSampler, RandomUnderSampler, SMOTE,
            SMOTEN, SVMSMOTE, SMOTETomek, BorderlineSMOTE, ADASYN]

# Choose which device to train the TabNet and MLP model on
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define a list of model algorithm training functions
train_fns = [mlp_fold, xgboost_fold, tabnet_fold]

# For each model algorithm, for each dataset, for each sampler, train 50 times.
for train_fn in train_fns:
    for csvfile in csvfiles:
        print(csvfile)
        dl = DatasetLoader(path, csvfile)
        log = []
        for Sampler in samplers:
            sampler = Sampler()
            for fold, (train_index, test_index) in enumerate(rskf.split(dl.X, dl.y)):
                # The TabNet model needs a validation set for early stopping
                if train_fn == tabnet_fold:
                    X_train, X_test, X_val, y_train, y_test, y_val = dl.slice_dataset(
                        train_index, test_index, val=True)
                    X_train, X_test, X_val = dl.scale_data(
                        X_train, X_test, X_val)
                    X_train_sampled, y_train_sampled = sampler.fit_resample(
                        X_train, y_train)
                    results_dict = train_fn(
                        X_train_sampled, y_train_sampled, X_test, y_test, 
                        X_val, y_val, batch_size=1024, device=device)
                else:
                    X_train, X_test, y_train, y_test = dl.slice_dataset(
                        train_index, test_index)
                    X_train, X_test = dl.scale_data(X_train, X_test)
                    X_train_sampled, y_train_sampled = sampler.fit_resample(
                        X_train, y_train)
                    results_dict = train_fn(
                        X_train_sampled, y_train_sampled, X_test, y_test, 
                        device=device)

                log.append({**{'csvfile': csvfile, 'fold': fold,
                           'sampler': str(Sampler)}, **results_dict})

        log_df = pd.DataFrame(log)
        log_df.to_excel(f'./results_{csvfile}.xlsx')
