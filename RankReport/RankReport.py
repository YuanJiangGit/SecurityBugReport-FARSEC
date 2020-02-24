import pickle
import operator
import pandas as pd
import numpy as np
import os


class RankReport:
    def __init__(self,exp,project):
        self.exp=exp
        wpath=None
        tpath=None
        self.project=project
        self.root='../resources/'
        self.ffilters=["train","sq","two","none","clni"]
        self.sfilters=["clnisq","clnitwo","clninone"]

    def best_filter_results(self):
        '''
        the best result for each filter way by channing classifiers
        :return:
        '''
        path=os.path.join(self.root,'results',self.project)
        filters=self.ffilters+self.sfilters
        df = pd.DataFrame()
        for filter in filters:
            temp = pd.DataFrame()
            for file in os.listdir(path):
                _f_split = file.split('_')
                # exp='WPP or 'TPP'
                filter_index=2 if self.exp=='WPP' else 3
                if _f_split[0] == self.exp and _f_split[filter_index] == filter:
                    # load results
                    _f_path=os.path.join(path, file)
                    data = pickle.load(open(_f_path, 'rb'))
                    _result = data['result']

                    if self.exp == 'WPP':
                        dict = {'TN': _result[0], 'TP': _result[1],
                                'FN': _result[2],
                                'FP': _result[3], 'pd': _result[4],
                                'pf': _result[5], 'prec': _result[6],
                                'f-measure': _result[7], 'g-measure': _result[8],
                                'Classifier': _f_split[3], 'Filter way': _f_split[filter_index],'File':_f_path}
                    else:
                        dict = {'TN': _result[0], 'TP': _result[1],
                                'FN': _result[2],
                                'FP': _result[3], 'pd': _result[4],
                                'pf': _result[5], 'prec': _result[6],
                                'f-measure': _result[7], 'g-measure': _result[8],
                                'Classifier': _f_split[4], 'Filter way': _f_split[filter_index],
                                'Source project': _f_split[2],'File':_f_path}

                    # measure_ = _result['f-measure']
                    temp =temp.append(pd.Series(dict),ignore_index=True)
            temp.sort_values(by='f-measure', ascending=False,inplace=True)
            df=df.append(temp.iloc[0],ignore_index=True)
        return df

    def boost_by_filter(self,target_filter,candidate_filters,results):
        '''
        Return best candidate filter based on the number of predicted sbrs.
        Return true of false if boosting is useful.
        :param target_filter:
        :param candidate_filters:
        :param results:  每个过滤器在不同分类器上的最好结果
        :return:
        '''
        _can_filter_results = results[results['Filter way'].isin(candidate_filters)]
        _can_filter_results['predicted_p_num']=_can_filter_results['TP']+_can_filter_results['FP']
        _can_filter_results.sort_values(by='predicted_p_num', ascending=True,inplace=True)
        best_candidate=_can_filter_results.iloc[0]

        _target_filter_results = results[results['Filter way']==target_filter]
        target_result=_target_filter_results.iloc[0]

        predicted_p_num_can=best_candidate['TP']+best_candidate['FP']
        predicted_p_num_target = target_result['TP'] + target_result['FP']
        if target_filter not in ['train','clni'] and predicted_p_num_can<predicted_p_num_target:
            boost_flag=True
        else:
            boost_flag=False
        return (best_candidate['Filter way'],boost_flag)


    def boost_result_by_map(self,target_filter,candidate_filters):
        '''
        以target_filter为过滤器，计算的mean average precision
        :param target_filter:
        :param candidate_filters:
        :return:
        '''
        best_results=self.best_filter_results()
        best_candidate=self.boost_by_filter(target_filter,candidate_filters,best_results) # (best filter way, boost_flag)

        train = best_results[best_results['Filter way'] == best_candidate[0]]
        # 把train对应的文件，加载进来
        _f_path= train.iloc[0]['File']
        data = pickle.load(open(_f_path, 'rb'))
        _train_pred = np.c_[data['y'], data['y_pred']]

        far=best_results[best_results['Filter way'] == target_filter]
        _f_path = far.iloc[0]['File']
        data = pickle.load(open(_f_path, 'rb'))
        # 保存预测的结果，总共有三列: 第一列是 test, 第二列是best_candidate所对应的过滤器的预测结果，第三列是target filter对应的预测结果
        # 是否进行boost，根据的是y_pred
        _train_far_pred = np.c_[_train_pred, data['y_pred']]

        mean_aveg = self.evaluate(_train_far_pred, best_candidate[1])
        return mean_aveg

    def boost_results_by_map(self,target_filters,candidate_filters):
        '''
        计算所有的target_filters所对应的mean average precision
        :param target_filters:
        :param candidate_filters:
        :return:
        '''
        columns = list(range(0, 10))
        df_results=pd.DataFrame(index=target_filters,columns=columns)
        for target_filter in target_filters:
            mean_aveg=self.boost_result_by_map(target_filter,candidate_filters)
            df_results.loc[target_filter]=pd.Series(mean_aveg)
        return df_results

    # [ground truth, class, probability]
    def evaluate(self,predict_result,boost):
        top_n = self.decile_vals(10, predict_result)
        precision_at_n = [0.0 for _ in predict_result]
        mean_precision_at = [0.0 for _ in top_n]
        if boost:
            predict_result = sorted(predict_result, key=operator.itemgetter(1), reverse=True)  # best predict result
        predict_result = sorted(predict_result, key=operator.itemgetter(2), reverse=True)  # best predict result
        # predict_result=list(predict_result)  # baseline
        sorted_ground_truth = [x[0] for x in predict_result]

        for i in range(len(predict_result)):
            if i == 0:
                precision_at_n[i] = 0
            else:
                precision_at_n[i] = float(sum(sorted_ground_truth[:i]) / i) * 100
        # Iterating over top n
        for k, rank in enumerate(top_n):
            # the total sbr in the top rank
            mean_precision_at[k] = float(sum(precision_at_n[:rank])) / rank

        return mean_precision_at

    # the boundary of each deci when splitting x_col into deci
    def decile_vals(self,deci, x_col):
        n = (int)(len(x_col) / deci)
        result = [0 for _ in range(deci)]
        for i in range(deci):
            result[i] = n * (i + 1) if n * (i + 1) <= len(x_col) else len(x_col)
        return result
def _test_best_filter_result():
    report = RankReport('WPP','ambari')
    report.best_filter_results()

def _test_boost_results_by_map():
    report = RankReport('TPP', 'camel')
    target_filters=['train','clni','sq','two','none','clnisq','clnitwo','clninone']
    candidate_filters=['train','sq','none']
    results=report.boost_results_by_map(target_filters,candidate_filters)
    print(results)

def main():
    exps=['WPP','TPP']
    #'chromium',
    projects=['chromium','wicket','ambari', 'camel', 'derby']
    target_filters = ['train', 'clni', 'sq', 'two', 'none', 'clnisq', 'clnitwo', 'clninone']
    candidate_filters = ['train', 'sq', 'none']

    for exp in exps:
        for project in projects:
            report = RankReport(exp, project)
            project_result_path = os.path.join(report.root, 'results', 'rank_results',exp+'_'+project+'.csv')
            _results = report.boost_results_by_map(target_filters, candidate_filters)
            # 每个项目的结果单独保存
            _results.to_csv(project_result_path)

if __name__ == '__main__':
    # _test_best_filter_result()
    # _test_boost_results_by_map()
    main()