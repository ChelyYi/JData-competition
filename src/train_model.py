from feature import *
from sklearn.model_selection import train_test_split
import xgboost as xgb


def xgboost_make_submission():
    print("开始执行...")
    train_start_date = '2016-03-02'
    train_end_date = '2016-04-01'
    label_start_date = '2016-04-01'
    label_end_date = '2016-04-05'
    cate = 8

    print("Start Make Train Set...")
    user_index, training_data, label = make_train_set(train_start_date, train_end_date, label_start_date, label_end_date, cate)
    #print(user_index.info())
    print(training_data.info())
    #print(label.info())
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2, random_state=0)
    print("End make train set")

    print("Start Training...")
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate': 0.05, 'n_estimators': 1000, 'max_depth': 4,
             'min_child_weight': 5, 'gamma': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 312
    label_score = 0.06
    #param['nthread'] = 4
    #param['eval_metric'] = "auc"
    param['eval_metric'] = 'logloss'
    plst = param.items()
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train(plst, dtrain, num_round, evallist)
    print("End Training.")

    print("Start Predicting...")
    data_start_date = "2016-03-17"
    data_end_date = "2016-04-16"
    sub_user_index, sub_trainning_data = make_data_set(data_start_date, data_end_date,cate)
    sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)
    y = bst.predict(sub_trainning_data)
    sub_user_index['label'] = y

    pred = sub_user_index[sub_user_index['label'] >= label_score]

    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred.to_csv('./sub/submission.csv', index=False, index_label=False)
    print("End Predicting...")



def xgboost_cv():
    print("交叉检验：")
    cate = 8
    train_start_date = '2016-03-02'
    train_end_date = '2016-04-01'
    label_start_date = '2016-04-01'
    label_end_date = '2016-04-05'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, label_start_date, label_end_date, cate)
    # print(user_index.info())
    print(training_data.info())
    # print(label.info())
    X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    """
    print("CV:")
    param = {'learning_rate': 0.05, 'n_estimators': 1000, 'max_depth': 6,
             'min_child_weight': 5, 'gamma': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1,  'silent': 1, 'objective': 'binary:logistic'}
    e_stop = 50
    cv_result = xgb.cv(params=param, dtrain=dtrain, num_boost_round=3000,nfold=5,stratified=False,
                       metrics=['error', 'auc'],early_stopping_rounds=e_stop,verbose_eval=1, show_stdv=True, maximize=False)
    best_nrounds = int((cv_result.shape[0] - e_stop) / (1 - 1 / 5))
    print("Best iteration:"+ str(best_nrounds))
    print(cv_result.shape[0])"""

    param = {'learning_rate': 0.05, 'n_estimators': 1000, 'max_depth': 4,
             'min_child_weight': 5, 'gamma': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 312
    label_score = 0.05
    param['eval_metric'] = 'logloss'
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    bst=xgb.train( param, dtrain, num_round, evallist)

    sub_start_date = '2016-03-12'
    sub_end_date = '2016-04-11'
    sub_label_start_date = '2016-04-11'
    sub_label_end_date = '2016-04-16'
    test_user_index, test_data, test_label = make_train_set(sub_start_date, sub_end_date,
                                                                   sub_label_start_date, sub_label_end_date,cate)
    test = xgb.DMatrix(test_data)
    test_result = bst.predict(test)

    print("Test Data result：")
    pred = test_user_index.copy()
    result_true = test_user_index.copy()
    pred['label'] = test_result
    pred = pred[pred['label'] >= label_score]
    result_true['label'] = test_label
    result_true = result_true[result_true['label'] == 1]
    report(pred, result_true)


if __name__ == '__main__':
    xgboost_make_submission()
    #xgboost_cv()

