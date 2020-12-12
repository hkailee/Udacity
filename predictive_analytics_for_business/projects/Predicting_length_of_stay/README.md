# EMR test - predicting LOS

## instruction to install packages
$ conda create -n py36_regression python=3.6 -y
$ conda activate py36_regression
$ conda install ipykernel jupyter -y; python -m ipykernel install --user --name py36_regression --display-name "py36_regression"; $ python -m pip install factor_analyzer scikit-learn torch xgboost lightgbm matplotlib seaborn tqdm statsmodels


## Summary
1. Summary performance
My score is based on a single xgboost (average on multi-seeds 20 fold).

The model score :
20-fold mean cross-validation MSE Score:  -0.7193041954946354
20-fold mean cross-validation r2 Score:  0.5952601998746865
test_mse:  0.3057348279231769 
test_r2:  0.9238808350232419


2. Feature Engineering
- Generated different derivative features based on what the vitals that i feel making sense to predict longer hospital stay and preprocessed/scaled the data accordingly.
- Included dimesional reduced principle components / varimax factors. However varimax factors were dropped eventually due to singularity detected in dataset. PCA (a more empirical dimensional reduction approach) was generated eventually and first 6 pcs used as additional features.

4. Feature selection
- Use Permutation Importance Feature selection method (https://www.kaggle.com/ogrellier/feature-selection-with-null-importances) to select top 13 features, followed by further feature selection using SHAP (SHapley Additive exPlanations) method (https://github.com/slundberg/shap)

5. Model
Only 4 features were selected eventually to be used in xgboost regressor - ['Hospital_day', 'min_SBP_ED', 'min_SpO2_ED', 'Weight_ED'], which gives rise to the best cross-validation and test scores. 

Thanks for reading!