# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics
import pandas

###############################
class OutputData:
    def __init__(self, num_of_label, trial_times):
        self.trial_times = trial_times
        self.params = ('AA', 'OA', 'Kappa', 'train_time', 'predict_time')
        self.params_class = tuple(np.arange(1, num_of_label + 1))
        self.args = ('dataset', 'method', 'patch', 'samples_per_class')
        self.chart = pandas.DataFrame(np.zeros((trial_times + 2, len(self.params + self.params_class))),
                                      columns=self.params_class+self.params, index=np.arange(1, trial_times + 3))
        self.chart = self.chart.rename(index={trial_times + 1: 'average'})

    def set_data(self, param_name, current_trail_turn, data):
        self.chart[param_name][current_trail_turn + 1] = data

    def output_data(self, path, xlsxname):
        for i in self.params_class:
            average_class = "{:.2f}".format(self.chart[i][0:self.trial_times].mean()) + " ± " + "{:.2f}".format(self.chart[i][0:self.trial_times].std())
            self.chart.loc['average', i] = average_class
            
        for param_name in self.params:
            average_name = "{:.2f}".format(self.chart[param_name][0:self.trial_times].mean()) + " ± " + "{:.2f}".format(self.chart[param_name][0:self.trial_times].std())
            self.chart.loc['average', param_name] = average_name
        self.chart.to_excel(path, sheet_name=xlsxname)