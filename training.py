import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

### load data
train = pd.read_csv('data/train.csv', index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)

### make result data
ml_train_results = train[['data_num','Label']].copy()
ml_test_results = test[['data_num','Label']].copy()

### GridSearch parameter
grid_para = {'svm':{'C': 50.0, 'gamma': 0.3, 'kernel': 'rbf'},
             'rf' :{'max_depth': 14, 'n_estimators': 85, 'random_state': 42}, 
             'xgb':{'max_depth': 7, 'n_estimators': 110, 'random_state': 42}}

### ML data setting
feature_col = ['Week',r'$\mu^c$',r'$\beta^c$',r'$Policy^c$',r'$Delta^c$',r'$Omicron^c$',r'$Policy^p$',r'$\sigma^c$']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train[feature_col])
X_test_scaled = scaler.transform(test[feature_col])
y_train = train['Label']
y_test = test['Label']

print('=====================================================')
print('=                   Result of SVM                   =')
print('=====================================================')
grid_svm = grid_para['svm']
svm_model = SVC(C=grid_svm['C'], 
            gamma=grid_svm['gamma'], 
            kernel=grid_svm['kernel'],
            probability=True)
svm_model.fit(X_scaled, y_train)
svm_pred_train = svm_model.predict(X_scaled)
svm_pred_test = svm_model.predict(X_test_scaled)
print('train accuracy : ', accuracy_score(y_train, svm_pred_train))
print('test accuracy : ', accuracy_score(y_test, svm_pred_test))

svm_scores = cross_val_score(svm_model,                 # ML 모델
                         X_scaled,            # train data
                         y_train,             # test data
                         scoring='accuracy',  # 예측성능평가 지표
                         cv=10)                # kfold k=cv
print('10-fold cross validation mean :', np.mean(svm_scores))
print(classification_report(y_test, svm_pred_test))
print('\n')

### make svm results
proba = svm_model.predict_proba(X_test_scaled)
np.savetxt('result/svm_proba.csv',proba,delimiter=",")
ml_train_results['svm'] = svm_pred_train
ml_test_results['svm'] = svm_pred_test

print('=====================================================')
print('=                   Result of RF                    =')
print('=====================================================')
grid_rf = grid_para['rf']
rf_model = RandomForestClassifier(n_estimators=grid_rf['n_estimators'], 
                                  max_depth=grid_rf['max_depth'],
                                  random_state=grid_rf['random_state'],
                                  )
rf_model.fit(X_scaled, y_train)
rf_pred_train = rf_model.predict(X_scaled)
rf_pred_test = rf_model.predict(X_test_scaled)
print('train accuracy : ', accuracy_score(y_train, rf_pred_train))
print('test accuracy : ', accuracy_score(y_test, rf_pred_test))

rf_scores = cross_val_score(rf_model,                 # ML 모델
                         X_scaled,            # train data
                         y_train,             # test data
                         scoring='accuracy',  # 예측성능평가 지표
                         cv=10)                # kfold k=cv
print('10-fold cross validation mean :', np.mean(rf_scores))
print(classification_report(y_test, rf_pred_test))
print('\n')

### make rf results
proba = rf_model.predict_proba(X_test_scaled)
np.savetxt('result/rf_proba.csv',proba,delimiter=",")
ml_train_results['rf']=rf_pred_train
ml_test_results['rf']=rf_pred_test

print('=====================================================')
print('=                   Result of XGB                   =')
print('=====================================================')
grid_xgb = grid_para['xgb']
xgb_model = XGBClassifier(n_estimators=grid_xgb['n_estimators'], 
            max_depth=grid_xgb['max_depth'],
            random_state=grid_xgb['random_state'])
xgb_model.fit(X_scaled, y_train)
xgb_pred_train = xgb_model.predict(X_scaled)
xgb_pred_test = xgb_model.predict(X_test_scaled)
print('train accuracy : ', accuracy_score(y_train, xgb_pred_train))
print('test accuracy : ', accuracy_score(y_test, xgb_pred_test))

xgb_scores = cross_val_score(xgb_model,                 # ML 모델
                         X_scaled,            # train data
                         y_train,             # test data
                         scoring='accuracy',  # 예측성능평가 지표
                         cv=10)                # kfold k=cv
print('10-fold cross validation mean :', np.mean(xgb_scores))
print(classification_report(y_test, xgb_pred_test))
print('\n')

### make rf results
proba = xgb_model.predict_proba(X_test_scaled)
np.savetxt('result/xgb_proba.csv',proba,delimiter=",")
ml_train_results['xgb']=xgb_pred_train
ml_test_results['xgb']=xgb_pred_test

### save result
ml_train_results.to_csv('result/ml_train_results.csv')
ml_test_results.to_csv('result/ml_test_results.csv')
feature_importance_dict = {'feature':feature_col,'RF':rf_model.feature_importances_, 'XGB':xgb_model.feature_importances_}
featrue_importance_df = pd.DataFrame(feature_importance_dict)
featrue_importance_df.to_csv('result/feature_importance.csv')

