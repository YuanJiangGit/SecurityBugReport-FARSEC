from Score.ScoreKeywords import ScoreKeywords
from Score.ScoreReport import ScoreReport
from CrossWords.SecurityWords import SecurityWords
from utils.data_process import read_data_from_csv
from FilterReport import FilterReport
import pandas as pd
import numpy as np
import os


class TPPFilterReport(FilterReport):
    def __init__(self):
        super(TPPFilterReport, self).__init__()
        self.df_all=None
        self.top_words=None


    def gen_test_data(self,source_project,target_project):
        '''
        生成测试数据集
        :param project:
        :return:
        '''
        df_all_target = read_data_from_csv(target_project)
        # the last 1/2 as training data
        df_test_target = pd.DataFrame(df_all_target, index=range(int(len(df_all_target) / 2),len(df_all_target)))

        if self.top_words:
            top_words=self.top_words
        else:
            sw = SecurityWords(source_project)
            df_all_source = read_data_from_csv(source_project)
            # the form 1/2 as training data
            df_train_source = pd.DataFrame(df_all_source, index=range(int(len(df_all_source) / 2)))
            corpus = sw.load_sbr_corpus(df_train_source)
            # security related words and the score
            sbr_words = sw.top_security_words(corpus)
            # security related words
            top_words = [tuple[0] for tuple in sbr_words]

        df_test_data = self.make_data_by_topwords([df_test_target], top_words)
        df_test_data['issue_id'] = df_test_data['issue_id'].astype('int64')
        save_path = os.path.join('../resources/', target_project,target_project+ '-' + source_project + '-' + 'tpp-test.csv')
        df_test_data.to_csv(save_path,index=None)


if __name__ == '__main__':
    FR = TPPFilterReport()
    # projects=['ambari','derby','camel','wicket','chromium']
    source_projects = ['ambari', 'derby', 'camel', 'wicket', 'chromium']
    target_projects=source_projects.copy()
    for target_project in target_projects:
        for source_project in source_projects:
            if source_project==target_project:
                continue
            FR.gen_test_data(source_project,target_project)
