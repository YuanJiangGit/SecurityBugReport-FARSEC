import os
import pickle
from pandas import Series, DataFrame


class ResultStatistic:
    def __init__(self):
        self.root = '../resources/results/'

    def sing_project_results(self, project, type='WPP'):
        path = os.path.join(self.root, project)
        files = os.listdir(path)
        _g_measure = 0
        _best2file = None
        _result = None
        for file in files:
            _f_split = file.split('_')
            # type='WPP or 'TPP'
            if _f_split[0] == type and float(_f_split[-1]) > _g_measure:
                _g_measure = float(_f_split[-1])
                _best2file = _f_split
                # load results
                data = pickle.load(open(os.path.join(path, file), 'rb'))
                _result = data['result']
                # [TN,TP,FN,FP,pd,pf,prec,f_measure,g_measure]
        if type == 'WPP':
            dict = {'TN': _result[0], 'TP': _result[1],
                    'FN': _result[2],
                    'FP': _result[3], 'pd': _result[4],
                    'pf': _result[5], 'prec': _result[6],
                    'f-measure': _result[7], 'g-measure': _result[8],
                    'Classifier': _best2file[4], 'Filter way': _best2file[3]}
        else:
            dict = {'TN': _result[0], 'TP': _result[1],
                    'FN': _result[2],
                    'FP': _result[3], 'pd': _result[4],
                    'pf': _result[5], 'prec': _result[6],
                    'f-measure': _result[7], 'g-measure': _result[8],
                    'Classifier': _best2file[4], 'Filter way': _best2file[3], 'Source project': _best2file[2]}

        print(_result)
        return Series(dict)

    def statistic(self):
        projects = ['chromium_large', 'mozilla_large']
        columns = ['Classifier', 'Filter way', 'TN', 'TP', 'FN', 'FP', 'pd', 'pf', 'prec', 'f-measure', 'g-measure']
        df_wpp = DataFrame(columns=columns, index=projects)
        for project in projects:
            _r_wpp = self.sing_project_results(project, 'WPP')
            df_wpp.loc[project] = _r_wpp

        df_wpp.to_csv(os.path.join(self.root, 'WPP_Best_Large_Project_FARSEC.csv'))


if __name__ == '__main__':
    _rs = ResultStatistic()
    _rs.statistic()
