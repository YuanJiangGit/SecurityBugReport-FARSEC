from SBRDetection.SecurityBRClassifier import SecurityClassifier
import pandas as pd
from sklearn.utils import shuffle
import pickle
import os


class SBRDetection:
    def __init__(self):
        self.root = '../resources/'
        self.result_save_path=None

    def read_data(self, project, filter, source_project=None):
        if source_project == None:
            train_path = os.path.join(self.root, project, project + '-' + filter + '.csv')
            test_path =os.path.join(self.root, project, project + '-' + 'test.csv')
        else:
            train_path = os.path.join(self.root, source_project, source_project+ '-' + filter + '.csv')
            test_path = os.path.join(self.root, project, project + '-' + source_project + '-' + 'tpp-test.csv')
        df_train = pd.read_csv(train_path)
        df_train = shuffle(df_train)

        df_test = pd.read_csv(test_path)
        return df_train, df_test

    def wpp_sbr_predict(self, df_train, df_test, classifier="MLP"):
        df_train = df_train.dropna()
        df_train = df_train.astype('int64')
        df_test = df_test.astype('int64')

        X_train = df_train.iloc[:, 1:-1]
        y_train = df_train.Security

        X_test = df_test.iloc[:, 1:-1]
        y_test = df_test.Security
        print(len(df_test))

        # X_train, X_test, y_train, y_test = train_test_split(content, label, test_size=0.4, random_state=42)
        sc = SecurityClassifier(classifier)
        sc.train(X_train, y_train)

        y_pred = sc.predict_b(X_test)
        result = sc.evaluate_b(y_test, y_pred)
        # print(result)
        y_score = sc.predict_p(X_test)
        # sc.evaluate_p(y_test,y_score)
        data = {'result': result, 'y_pred': y_pred, 'y_score': y_score, 'y': y_test}
        # print(data)
        # 移除已经存在的数据
        for file in os.listdir(os.path.dirname(self.result_save_path)):
            if file.startswith(os.path.basename(self.result_save_path)+'_'):
                os.remove(os.path.join(os.path.dirname(self.result_save_path),file))

        with open(self.result_save_path+'_'+str(data['result'][-1]), 'wb') as f:
            pickle.dump(data, f)

def entry():
    sbr_detection = SBRDetection()
    #WPP ('ambari','two','RF'), ('wicket','two','LR'), ('camel','two','LRCV') ('derby', 'two', 'SVMCV') ('chromium','clnisq','MLP' )
    #TPP ('ambari','chromium','sq','MLP') ('wicket','camel','train','NB')  ('camel','derby','two','NB')  ('derby', 'chromium','clnisq', 'NB') ('chromium', 'ambari','clnisq', 'RF')
    projects = ['chromium_large','mozilla_large']
    source_projects = ['ambari', 'derby', 'camel', 'wicket', 'chromium']
    # 'clni', ['train', 'two', 'sq', 'none','clninone','clnisq','clnitwo'] ,
    key_farsecs = ['train', 'two', 'sq','clni', 'none','clninone','clnisq','clnitwo']
    # ['LR', 'LRCV','MLP', 'MLPCV','SVM','SVMCV', 'NB', 'RF','RFCV', 'KNN' ,'KNNCV']
    classifiers=['LR', 'LRCV','MLP', 'MLPCV','SVM','SVMCV', 'NB', 'RF','RFCV', 'KNN' ,'KNNCV']
    TPP = False
    WPP=True
    for project in projects:
        for key_farsec in key_farsecs:
            for classifier in classifiers:
                if WPP:
                    sbr_detection.result_save_path=os.path.join(sbr_detection.root,'results',project,'_'.join(['WPP',project,key_farsec,classifier]))
                    df_train, df_test = sbr_detection.read_data(project, key_farsec)
                    sbr_detection.wpp_sbr_predict(df_train, df_test, classifier=classifier)
                if TPP:
                    for source_project in source_projects:
                        if source_project!=project:
                            sbr_detection.result_save_path = os.path.join(sbr_detection.root, 'results', project,
                                                                          '_'.join(
                                                                              ['TPP', project,source_project, key_farsec, classifier]))
                            df_train, df_test = sbr_detection.read_data(project, key_farsec, source_project)
                            sbr_detection.wpp_sbr_predict(df_train, df_test, classifier=classifier)



if __name__ == '__main__':
    entry()
    # ('derby', 'two', 'SVM')
    # sbr_detection = SBRDetection()
    # source_project='ambari'
    # project='camel'
    # key_farsec = 'train'
    # classifier = 'LRCV'
    # TPP=False
    # if TPP:
    #     sbr_detection.result_save_path = os.path.join(sbr_detection.root, 'results', project,
    #                                                   '_'.join(
    #                                                       ['TPP', project, source_project, key_farsec, classifier]))
    #     df_train, df_test = sbr_detection.read_data(project, key_farsec, source_project)
    # else:
    #     sbr_detection.result_save_path = os.path.join(sbr_detection.root, 'results', project,
    #                                                   '_'.join(['WPP', project, key_farsec, classifier]))
    #     df_train, df_test = sbr_detection.read_data(project, key_farsec)
    #
    # sbr_detection.wpp_sbr_predict(df_train, df_test, classifier=classifier)