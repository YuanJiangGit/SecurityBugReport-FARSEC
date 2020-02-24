'''
author: jiangyuan
data: 2018/5/17
function: predict security bug report
'''
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
import re
import numpy as np

# previous setting： grid_search = GridSearchCV(model, param_grid, cv=2, n_jobs=1, verbose=1, scoring=score)

class SecurityClassifier():
    def __init__(self, classifier):
        classifiers = {'NB': self.naive_bayes_classifier,
                       'NBCV':self.nb_cross_validation,
                       'KNN': self.knn_classifier,
                       'KNNCV':self.knn_cross_validation,
                       'LR': self.logistic_regression_classifier,
                       'LRCV':self.lr_cross_validation,
                       'RF': self.random_forest_classifier,
                       'RFCV':self.rf_cross_validation,
                       'DT': self.decision_tree_classifier,
                       'SVM': self.svm_classifier,
                       'SVMCV': self.svm_cross_validation,
                       'GBDT': self.gradient_boosting_classifier,
                       'MLP': self.MLP_Classifier,
                       'ADB': self.ADB_classifier,
                       'MLPCV': self.mlp_cross_validation
                       }
        if classifier not in classifiers:
            raise ValueError(classifier)
        else:
            self.classifier = classifiers[classifier]
        self.model = None

    # train model
    def train(self, train_x, train_y):
        self.model = self.classifier(train_x, train_y)

    # predict binary
    def predict_b(self, X_test):
        return self.model.predict(X_test)

    # predict probability
    def predict_p(self, X_test):
        return self.model.predict_proba(X_test)

    # evaluate binary
    def evaluate_b(self, y, y_pred):
        report_text = classification_report(y, y_pred, target_names=['nsbr', 'sbr'])
        print(report_text)
        report_list = re.sub(r'[\n\s]{1,}', ' ', report_text).strip().split(' ')
        # print(report_list)
        print('Confusion Matrix:')
        conf_matrix = confusion_matrix(y, y_pred)
        print(conf_matrix)
        TN = conf_matrix.item((0, 0))
        FN = conf_matrix.item((1, 0))
        TP = conf_matrix.item((1, 1))
        FP = conf_matrix.item((0, 1))
        pd = 100 * TP / (TP + FN)
        pf = 100 * FP / (FP + TN)
        g_measure = 2 * pd * (100 - pf) / (pd + 100 - pf)
        print('g-measure:%s' % g_measure)
        print('precision:%s' % precision_score(y, y_pred, average='binary'))
        print('recall:%s' % recall_score(y, y_pred, average='binary'))
        print('f-measure:%s' % f1_score(y, y_pred, average='binary'))
        prec = 100 * precision_score(y, y_pred, average='binary')
        recall = 100 * recall_score(y, y_pred, average='binary')
        f_measure = 100 * f1_score(y, y_pred, average='binary')

        accuracy = 100 * accuracy_score(y, y_pred)
        # dict = {'TN': TN, 'FN': FN, 'TP': TP, 'FP': FP, 'pd': pd, 'pf': pf, 'prec': prec,
        #         'recall': recall, 'f-measure': f_measure, 'g-measure': g_measure, 'accuracy': accuracy}
        result=[TN,TP,FN,FP,pd,pf,prec,f_measure,g_measure]

        return result


    # Multinomial Naive Bayes Classifier
    def naive_bayes_classifier(self, train_x, train_y):
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB(alpha=0.01)
        model.fit(train_x, train_y)
        return model

    def nb_cross_validation(self,train_x,train_y):
        from sklearn.model_selection import GridSearchCV
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import make_scorer
        from pandas import DataFrame
        import pandas as pd
        score = make_scorer(self.my_custom_loss_func, greater_is_better=True)
        model = MultinomialNB(alpha=0.01)

        param_grid = dict(alpha=[0.1,0.5,1.0],binarize=[0.0,0.2,0.4,0.6,0.8])
        grid_search = GridSearchCV(model, param_grid, cv=2, n_jobs=3, verbose=1, scoring=score)
        grid_search.fit(train_x, train_y)
        # df=DataFrame(data=grid_search.cv_results_)
        # df.to_csv('../resources/temp_grid.csv', encoding='utf-8')
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in best_parameters.items():
            print(para, val)
        # model = KNeighborsClassifier(n_neighbors =best_parameters['n_neighbors'], weights=best_parameters['weights'],
        #                              algorithm = best_parameters['algorithm'], leaf_size=best_parameters['leaf_size'], probability=True)
        # model.fit(train_x, train_y)
        # permits to return grid_search object(refit=true default)
        return grid_search


    # KNN Classifier
    def knn_classifier(self, train_x, train_y):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
        model.fit(train_x, train_y)
        return model

    # knn Classifier using cross validation
    def knn_cross_validation(self, train_x, train_y):
        from sklearn.model_selection import GridSearchCV
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import make_scorer
        from pandas import DataFrame
        import pandas as pd
        score = make_scorer(self.my_custom_loss_func, greater_is_better=True)
        model = KNeighborsClassifier()
        k_range = list(range(1, 10))
        leaf_range = list(range(1, 2))
        weight_options = ['uniform', 'distance']
        algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
        param_grid = dict(n_neighbors=k_range, weights=weight_options, algorithm=algorithm_options,
                             leaf_size=leaf_range)
        grid_search = GridSearchCV(model, param_grid, cv=2, n_jobs=3, verbose=1, scoring=score)
        grid_search.fit(train_x, train_y)
        # df=DataFrame(data=grid_search.cv_results_)
        # df.to_csv('../resources/temp_grid.csv', encoding='utf-8')
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in best_parameters.items():
            print(para, val)
        # model = KNeighborsClassifier(n_neighbors =best_parameters['n_neighbors'], weights=best_parameters['weights'],
        #                              algorithm = best_parameters['algorithm'], leaf_size=best_parameters['leaf_size'], probability=True)
        # model.fit(train_x, train_y)
        # permits to return grid_search object(refit=true default)
        return grid_search
    # Logistic Regression Classifier
    def logistic_regression_classifier(self, train_x, train_y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty='l2')
        model.fit(train_x, train_y)
        return model

    # lr Classifier using cross validation
    def lr_cross_validation(self, train_x, train_y):
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import make_scorer
        score = make_scorer(self.my_custom_loss_func, greater_is_better=True)
        model = LogisticRegression(tol=1e-6)
        param_grid = [{'penalty':['l1','l2'],'solver':['liblinear'],'multi_class':['ovr'],'class_weight':['balanced',None]},
                      {'penalty':['l2'], 'C':[0.01,0.05,0.1,0.5,1,5,10,50], 'solver':['lbfgs'],'multi_class':['ovr','multinomial']}]
        grid_search = GridSearchCV(model, param_grid, cv=2, n_jobs=3, verbose=1, scoring=score)
        grid_search.fit(train_x, train_y)
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in best_parameters.items():
            print(para, val)
        # model = SVC(kernel='linear', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
        # model.fit(train_x, train_y)
        # permits to return grid_search object(refit=true default)
        return grid_search


    # Random Forest Classifier
    def random_forest_classifier(self, train_x, train_y):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5)
        model.fit(train_x, train_y)
        return model

    # rf Classifier using cross validation
    def rf_cross_validation(self, train_x, train_y):
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import make_scorer
        from pandas import DataFrame
        import pandas as pd
        score = make_scorer(self.my_custom_loss_func, greater_is_better=True)
        model = RandomForestClassifier(n_estimators=20)
        param_grid = {"max_depth": [3, None],
                      "max_features": [1, 3, 10],
                      "min_samples_split": [2, 3, 10],
                      "min_samples_leaf": [1, 3, 10],
                      "bootstrap": [True, False],
                      'n_estimators':list(range(5,15,2)),
                      "criterion": ["gini", "entropy"]}
        grid_search = GridSearchCV(model, param_grid, cv=2, n_jobs=3, verbose=1, scoring=score)
        grid_search.fit(train_x, train_y)
        # df=DataFrame(data=grid_search.cv_results_)
        # df.to_csv('../resources/temp_grid.csv', encoding='utf-8')
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in best_parameters.items():
            print(para, val)
        # permits to return grid_search object(refit=true default)
        return grid_search

    # Decision Tree Classifier
    def decision_tree_classifier(self, train_x, train_y):
        from sklearn import tree
        model = tree.DecisionTreeClassifier()
        model.fit(train_x, train_y)
        return model

    # GBDT(Gradient Boosting Decision Tree) Classifier
    def gradient_boosting_classifier(self, train_x, train_y):
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200)
        model.fit(train_x, train_y)
        return model

    # SVM Classifier
    def svm_classifier(self, train_x, train_y):
        from sklearn.svm import SVC
        model = SVC(kernel='linear', C=1, gamma=0.001, probability=True)
        model.fit(train_x, train_y)
        return model

    # SVM Classifier using cross validation
    def svm_cross_validation(self, train_x, train_y):
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        from sklearn.metrics import make_scorer
        from pandas import DataFrame
        import pandas as pd
        score = make_scorer(self.my_custom_loss_func, greater_is_better=True)
        model = SVC(kernel='linear', probability=True)
        param_grid = {'C': [0.01, 0.05, 0.1, 1, 5, 10, 20, 100], 'gamma': [0.001, 0.0001]}
        grid_search = GridSearchCV(model, param_grid, cv=2, n_jobs=3, verbose=1, scoring=score)
        grid_search.fit(train_x, train_y)
        # df=DataFrame(data=grid_search.cv_results_)
        # df.to_csv('../resources/temp_grid.csv', encoding='utf-8')
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in best_parameters.items():
            print(para, val)
        model = SVC(kernel='linear', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
        model.fit(train_x, train_y)
        # permits to return grid_search object(refit=true default)
        return model

    def MLP_Classifier(self, train_x, train_y):
        from sklearn.neural_network import MLPClassifier
        # (30,30)对derby效果最好，达到74.459%，增加20%
        mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), activation='tanh', max_iter=300)
        mlp.fit(train_x, train_y)
        return mlp

    def my_custom_loss_func(self, ground_truth, predictions):
        # print('Confusion Matrix:')
        conf_matrix = confusion_matrix(ground_truth, predictions)
        # print(conf_matrix)
        TN = conf_matrix.item((0, 0))
        FN = conf_matrix.item((1, 0))
        TP = conf_matrix.item((1, 1))
        FP = conf_matrix.item((0, 1))
        pd = 100 * TP / (TP + FN)
        pf = 100 * FP / (FP + TN)
        if pd + 100 - pf==0:
            g_measure = 0
        else:
            g_measure = 2 * pd * (100 - pf) / (pd + 100 - pf)
        # print(g_measure)
        return g_measure

    # MLP Classifier using cross validation
    def mlp_cross_validation(self, train_x, train_y):
        from sklearn.model_selection import GridSearchCV
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import make_scorer
        score = make_scorer(self.my_custom_loss_func, greater_is_better=True)
        model = MLPClassifier(activation='tanh', max_iter=300)
        hidden_layer_list = []
        for i in range(10, 50, 5):
            for j in range(10, 50, 5):
                hidden_layer_list.append((i, j, 30))
        param_grid = {'hidden_layer_sizes': hidden_layer_list}
        grid_search = GridSearchCV(model, param_grid, cv=2, n_jobs=3, verbose=1, scoring=score)
        grid_search.fit(train_x, train_y)
        # best_parameters = grid_search.best_estimator_.get_params()
        # for para, val in best_parameters.items():
        #     print(para, val)
        # model = MLPClassifier(activation='tanh', hidden_layer_sizes=best_parameters['hidden_layer_sizes'],
        #                       max_iter=300)
        # model.fit(train_x, train_y)
        return grid_search

    def ADB_classifier(self, train_x, train_y):
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=100)
        model.fit(train_x, train_y)
        return model

