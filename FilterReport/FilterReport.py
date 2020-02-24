from Score.ScoreKeywords import ScoreKeywords
from Score.ScoreReport import ScoreReport
from CrossWords.SecurityWords import SecurityWords
from utils.data_process import read_data_from_csv
import pandas as pd
import numpy as np
import os


class FilterReport:
    def __init__(self):
        self.df_all=None
        self.top_words=None

    def one_report(self, report, top_words):
        '''
        convert report into the binary form
        :param report:
        :param top_words:
        :return:
        '''
        dict = {term: 0 for term in top_words}
        # 0 就是issue_id
        dict['issue_id'] = report[0]
        text = report['summary'] + report['description']
        for term in text:
            if term in top_words:
                dict[term] += 1
        dict['Security'] = report['Security']
        return dict

    def make_data_by_topwords(self, pdList, top_words):
        '''
        combine dataset in pdList and convert each instance according to top_words
        :param pdList:
        :param top_words:
        :return:
        '''
        columns = ['issue_id'] + top_words + ['Security']
        data = pd.DataFrame(columns=columns)
        for corpus in pdList:
            for i in range(len(corpus)):
                report_var = self.one_report(corpus.iloc[i], top_words)
                data = data.append(report_var, ignore_index=True)
        return data

    def farsec_filtered(self, df_train, key_farsec, top_words):
        '''
        farsec 过滤nsbr
        :param df_train:  数据集
        :param key_farsec: support function type, which includes {two, sq, none}
        :param top_words:  security related keywords
        :return:  The output is a data matrix with the frequency of each top-word in each bug report, The first column is the id and the last is
        the label: 1 or 0 to indicate SBR or NSBR respectively.
        '''
        SK = ScoreKeywords()
        # score feature sets（or keywords）
        scored_words = SK.score_words(df_train, key_farsec, top_words)

        sbr_corpus = df_train[df_train['Security'] == 1]
        nsbr_corpus = df_train[df_train['Security'] == 0]

        # filter nsbr
        SR = ScoreReport()
        nsbr_corpus['score'] = nsbr_corpus.apply(lambda x: SR.score_report(scored_words, x.summary + x.description),
                                                 axis=1)
        filtered_nsbr_corpus = nsbr_corpus[nsbr_corpus['score'] < 0.75]
        # combine sbr_corpus and filtered_nsbr_corpus
        filtered_train_data = self.make_data_by_topwords(pdList=[sbr_corpus, filtered_nsbr_corpus], top_words=top_words)
        return filtered_train_data

    def euclidean_distance(self, x, y):
        x_vec = np.array(x)
        y_vec = np.array(y)
        dist = np.linalg.norm(x_vec - y_vec)
        return dist

    def is_noisy_report(self, one, n, threshold, df_all):
        '''
        determine the NSBR whether it is a noisy
        :param one:
        :param n:
        :param threshold:
        :param df_all:
        :return:
        '''
        # compute the distance between one and df_all
        df_all['score'] = df_all.apply(lambda br: self.euclidean_distance(br[1:-2], one[1:-2]), axis=1)
        df_all.sort_values('score', ascending=True, inplace=True)  # 升序
        # df_all是所有数据，所以one有可能和自己计算距离，距离是0排在最前
        if int(one['issue_id']) in list(df_all.head(n)['issue_id']):
            df_top_n = df_all.head(n + 1)
        else:
            df_top_n = df_all.head(n)
        df_diff_one = df_top_n[df_top_n['Security'] != one['Security']]
        theta = len(df_diff_one) * 1.0 / n
        if theta >= threshold:
            return True
        else:
            return False

    def clnix_filtered(self, df_train, n, ep, top_words):
        '''
        From the paper, Dealing with Noise in Defect Prediction, Closest List Noise Identification (CLNI), is adapted and compared with FARSEC filtering.
        CLNI uses top-nearest neighbors to determine if an instance is noisy. Hence it very slow on larger data-sets like Chromium. time-period-hrs is used to stop the algorithm after a set number of hours.
        To speed up the algorithm, select the 100 nearest nsbrs to each sbr, and only use these as potential noisy nsbrs.
        :param df_train:
        :param n: n neighbours
        :param ep: prev/curr>ep  stop
        :param top_words:
        :return:
        '''
        sbr_corpus = df_train[df_train['Security'] == 1]
        nsbr_corpus = df_train[df_train['Security'] == 0]
        threshold = len(sbr_corpus) * 1.0 / len(df_train)

        sbr_data = self.make_data_by_topwords([sbr_corpus], top_words)
        sbr_data['issue_id'] = sbr_data['issue_id'].astype('int64')
        sbr_data['score'] = 0

        nsbr_data = self.make_data_by_topwords([nsbr_corpus], top_words)
        nsbr_data['issue_id'] = nsbr_data['issue_id'].astype('int64')
        nsbr_data['score'] = 0
        # initial_noisy_nsbrs
        initial_noisy_nsbrs_id = set()
        for i in range(len(sbr_data)):
            sbr = sbr_data.iloc[i]
            nsbr_data['score'] = nsbr_data.apply(lambda nsbr: self.euclidean_distance(nsbr[1:-2], sbr[1:-2]), axis=1)
            nsbr_data.sort_values('score', ascending=True, inplace=True)  # 升序
            temp_noise_nsbr = nsbr_data.head(100)
            initial_noisy_nsbrs_id = initial_noisy_nsbrs_id.union(temp_noise_nsbr['issue_id'])

        initial_noisy_nsbrs = nsbr_data[nsbr_data['issue_id'].isin(initial_noisy_nsbrs_id)]
        # initial_noisy_nsbrs中is_noisy_report结果为true的行保留
        # curr = initial_noisy_nsbrs[self.is_noisy_report(initial_noisy_nsbrs, n, threshold, df_train)]
        df_train = pd.concat([sbr_data, nsbr_data])
        curr = initial_noisy_nsbrs[
            initial_noisy_nsbrs.apply(lambda x: self.is_noisy_report(x, n, threshold, df_train), axis=1)]
        prev = set()
        for i in range(100):
            # curr的长度为0，代表一个都没去掉
            if len(curr)==0:
                return df_train
            if len(prev) / len(curr) >= ep:
                break
            # initial_noisy_nsbrs不在curr中的nsbr
            new_nsbr = initial_noisy_nsbrs[~initial_noisy_nsbrs['issue_id'].isin(list(curr['issue_id']))]
            new_data = pd.concat([new_nsbr, sbr_data])
            new_noisy = new_nsbr[new_nsbr.apply(lambda x: self.is_noisy_report(x, n, threshold, new_data), axis=1)]
            prev = curr.copy()
            curr = pd.concat([curr, new_noisy])

        # 去掉噪音的nsbr
        filtered_nsbr_data=nsbr_data[~nsbr_data['issue_id'].isin(list(curr['issue_id']))]
        filtered_train_data = pd.concat([filtered_nsbr_data, sbr_data])
        return filtered_train_data,filtered_train_data['issue_id']

    def get_batch(self,dataset, idx, bs):
        '''
        obtain a batch size of NSBRs
        :param dataset:
        :param idx:
        :param bs:
        :return:
        '''
        tmp = dataset.iloc[idx: idx + bs]
        return tmp

    def clni_filtered(self, train_data, n, ep):
        '''
        more fast CLNI filtered implementation by means of matrix computation
        :param df_train:
        :param n:
        :param ep:
        :param _
        :return:
        '''
        train_data=train_data.dropna()
        sbr_data = train_data[train_data['Security'] == 1]
        sbr_data['issue_id'] = sbr_data['issue_id'].astype('int64')
        # sbr_data['score'] = 0

        nsbr_data = train_data[train_data['Security'] == 0]
        nsbr_data['issue_id'] = nsbr_data['issue_id'].astype('int64')
        # nsbr_data['score'] = 0
        NSBR = nsbr_data
        SBR = sbr_data

        del_count = 0
        e = 0
        batch_size=2000
        # ep 0.99
        while e < ep:
            i = 0
            index = set()
            while i < len(NSBR):
                print('process the %s-th NSBRs'%i)
                batch = self.get_batch(NSBR, i, batch_size)
                i += batch_size
                disNN = self.EuclideanDistances(batch.ix[:,1:-1], NSBR.ix[:,1:-1])
                disNN.sort()
                disNS = self.EuclideanDistances(batch.ix[:,1:-1], SBR.ix[:,1:-1])
                disNS.sort()
                # 排序后，当一份NSBR第五(n+1)近的NSBR比最近的SBR远的话，那么它一定是噪音（top5的不同标签的百分比>0.8%--9%,一个进前五就能有20%)
                Noise = disNN[:, n+1] - disNS[:, 0]

                for j in range(Noise.shape[0]):
                    if Noise[j] > 0:
                        index.add(batch.iloc[j]['issue_id'])

            # 统计本次删除的，一共删除的,计算e，并从语料库中移除对应的NSBR
            delect = len(index)
            if del_count + delect == 0:
                e = 1
            else:
                e = del_count / (del_count + delect)
            del_count = del_count + delect
            # NSBR = np.delete(NSBR, index, axis=0)
            NSBR=NSBR[~NSBR['issue_id'].isin(list(index))]

        filtered_train_data = pd.concat([NSBR, SBR])
        print('CLNI delect : ', del_count)
        return filtered_train_data, filtered_train_data['issue_id']

    def EuclideanDistances(self, A, B):
        # .values 转化为np array
        A=A.values
        B=B.values
        AB = A.dot(B.T)
        Asq = np.array([(A ** 2).sum(axis=1)]).T  # A行1列
        Bsq = (B ** 2).sum(axis=1)  # 1行B列
        # 结果是欧氏距离的平方(未开方），已经足够比较距离了
        distance = -2 * AB + Asq + Bsq
        return distance


    def gen_train_data(self, project,FARSEC,CLNI,key_farsec=' '):
        '''
        obtain the filtered training data of project
        :param project:
        :return:
        '''
        if FARSEC:
            if key_farsec not in ['two','sq','none','train']:
                return None
        df_all = read_data_from_csv(project)
        # the form 1/2 as training data
        df_train = pd.DataFrame(df_all, index=range(int(len(df_all) / 2)))

        sw = SecurityWords(project)
        corpus = sw.load_sbr_corpus(df_train)
        # security related words and the score
        sbr_words = sw.top_security_words(corpus)
        # security related words
        top_words = [tuple[0] for tuple in sbr_words]
        # used in save unfiltered data
        if key_farsec=='train':
            if FARSEC and CLNI:
                train_data=self.make_data_by_topwords([df_train],top_words)
                save_path = os.path.join('../resources/', project, project + '-' + key_farsec + '.csv')
                train_data.to_csv(save_path, index=None)
                return None
            else:
                return None

        train_data_path=os.path.join('../resources/', project, project + '-' + 'train' + '.csv')
        if os.path.exists(train_data_path):
            train_data=pd.read_csv(train_data_path)
        else:
            train_data=self.make_data_by_topwords([df_train],top_words)

        retain_ids=None
        if CLNI:
            save_path = os.path.join('../resources/', project, project + '-' + 'clni.csv')
            if os.path.exists(save_path):
                clni_filtered_df=pd.read_csv(save_path)
                retain_ids=clni_filtered_df['id']
            else:
                filtered_train_data,retain_ids= self.clni_filtered(train_data, n=5, ep=0.99)
                filtered_train_data.rename(columns={'issue_id':'id'},inplace=True)
                filtered_train_data.to_csv(save_path, index=None)

        if FARSEC:
            if CLNI==True:
                df_train=df_train[df_train.iloc[:,0].isin(retain_ids)]

            filtered_train_data = self.farsec_filtered(df_train, key_farsec, top_words)

            _key_farsec= 'clni' + key_farsec if CLNI else key_farsec
            save_path = os.path.join('../resources/', project, project + '-' + _key_farsec + '.csv')
            filtered_train_data.to_csv(save_path, index=None)


    def gen_test_data(self,project):
        '''
        生成测试数据集
        :param project:
        :return:
        '''
        if self.df_all:
            df_all=self.df_all
        else:
            df_all = read_data_from_csv(project)
        # the last 1/2 as training data
        df_test = pd.DataFrame(df_all, index=range(int(len(df_all) / 2),len(df_all)))

        if self.top_words:
            top_words=self.top_words
        else:
            sw = SecurityWords(project)
            corpus = sw.load_sbr_corpus(df_all)
            # security related words and the score
            sbr_words = sw.top_security_words(corpus)
            # security related words
            top_words = [tuple[0] for tuple in sbr_words]

        df_test_data = self.make_data_by_topwords([df_test], top_words)
        df_test_data['issue_id'] = df_test_data['issue_id'].astype('int64')
        save_path = os.path.join('../resources/', project, project + '-' + 'test.csv')
        df_test_data.to_csv(save_path,index=None)


def entry():
    FR = FilterReport()
    # ,'chromium_large','mozilla_large'
    projects = ['derby', 'camel', 'wicket', 'chromium']
    # projects=['chromium_large','mozilla_large']
    # projects = ['mozilla_large']
    FARSEC = True
    # key_farsecs = ['train', 'two', 'sq', 'none']
    key_farsecs = ['two', 'sq', 'none']
    CLNIs = [True]
    for project in projects:
        # FR.gen_test_data(project)
        for key_farsec in key_farsecs:
            for CLNI in CLNIs:
                print(project)
                FR.gen_train_data(project, FARSEC, CLNI, key_farsec)


def testCLNI():
    FR = FilterReport()
    FARSEC = False
    CLNI=True
    project = 'ambari'
    FR.gen_train_data(project, FARSEC, CLNI)
if __name__ == '__main__':
    entry()
    # testCLNI()


