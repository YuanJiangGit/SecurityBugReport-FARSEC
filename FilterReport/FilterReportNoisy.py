from FilterReport import FilterReport
from Score.ScoreKeywords import ScoreKeywords
from Score.ScoreReport import ScoreReport
from CrossWords.SecurityWords import SecurityWords
from utils.data_process import read_data_from_csv
import pandas as pd
import numpy as np
import math
import os


class FilterReportNoisy(FilterReport):
    def __init__(self):
        super(FilterReportNoisy, self).__init__()

    def make_noise(self, df_train, percent):
        '''
        make noise by change the security label to non-security label in accordance with a certain proportion
        :param df_train:
        :param percent:
        :return:
        '''
        df = df_train.copy(deep=True)
        # randomly selected n% SBRS and artificially change their labels from security to non-security
        df_sbr = df[df.Security == 1]
        # df.Security.apply(lambda x: 0 if x.security==1 else 0,axis=1)
        print('the number of sbr before the change is %s' % len(df_sbr))
        noise_issue_id = []
        # the number of noise data
        noise_num = math.ceil(len(df_sbr) * percent)
        # 向上取整
        for i in range(noise_num):
            issue_id = df_sbr.iloc[i, 0]
            noise_issue_id.append(issue_id)
        # update the value of label(change SBR to NSBR)
        df.Security = df.apply(lambda x: 0 if x[0] in noise_issue_id else x.Security, axis=1)

        df_sbr = df[df.Security == 1]
        # add FNs to the golden set
        print('the length of SBR after the change is %s' % len(df_sbr))
        return df, noise_issue_id

    def gen_train_data(self, project, percent, FARSEC, key_farsec=' '):
        '''
        obtain the filtered training data of project
        :param project:
        :return:
        '''
        if FARSEC:
            if key_farsec not in ['two', 'sq', 'none', 'train']:
                return None
        df_all = read_data_from_csv(project)
        # the form 1/2 as training data
        df_train = pd.DataFrame(df_all, index=range(int(len(df_all) / 2)))

        # make noisy for df_train
        df_train, noise_issue_id = self.make_noise(df_train, percent)

        sw = SecurityWords(project)
        corpus = sw.load_sbr_corpus(df_train)
        # security related words and the score
        sbr_words = sw.top_security_words(corpus)
        # security related words
        top_words = [tuple[0] for tuple in sbr_words]
        # used in save unfiltered data

        filtered_train_data = self.farsec_filtered(df_train, key_farsec, top_words)
        filter_issue_id = filtered_train_data.iloc[:,0]
        train_noise_issue = set(noise_issue_id).intersection(set(filter_issue_id))
        return train_noise_issue

    # how resistant is FARSEC to mislabelled bug reports
    def resistant_cap(self):
        # projects = ['chromium','wicket','ambari','camel','derby']
        projects = ['chromium']
        columns = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        FARSEC = True
        key_farsecs = ['two', 'sq', 'none']
        for project in projects:
            df_noise = pd.DataFrame(columns=columns, index=key_farsecs)
            for key_farsec in key_farsecs:
                dict_noise = {}
                for percent in columns:
                    train_noise_issue = self.gen_train_data(project, percent, FARSEC, key_farsec)
                    num_noise_issue = len(train_noise_issue)
                    print('the number of noise issue report is %s' % num_noise_issue)
                    dict_noise[percent] = num_noise_issue
                df_noise.loc[key_farsec] = pd.Series(dict_noise)
            result_dir = '../resources/results/resistant_results'
            df_noise.to_csv(os.path.join(result_dir, project + '_farsec_noise.csv'))

def main():
    frn = FilterReportNoisy()
    frn.resistant_cap()

if __name__ == '__main__':
    main()