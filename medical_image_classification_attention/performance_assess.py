import numpy as np
import xlsxwriter

class Performance_recorder_txt():
    def __init__(self, save_directory_txt='', save_directory_excel='', object_id='', if_sort = True, **kwargs):
        '''
        :param save_directory_txt: directory to save text file which is to record performance
        :param save_directory_excel: directory to save excel file which is to record performance
        :param object_id: String that describe the object. E.g.) 'Fold_1_Validation_Tumor_T2'
        :param kwargs: Expect dictionary with String type key and corresponding empty list.
        '''
        self._save_directory_txt = save_directory_txt
        self._save_directory_excel = save_directory_excel
        self._object_id = object_id
        self._one_epoch_list_dict = kwargs
        self._tendency_list_dict = {}
        for k, v in self._one_epoch_list_dict.items():
            self._tendency_list_dict[k] = v.copy()
        self._if_sort = if_sort
        self._index_list = []

    def _save_txt(self, _list, _add_str):
        '''
        save list as text file at 'self._save_directory_txt' utilizing 'self._object_id' followed by '_add_str)
        :param _list:
        :param _add_str:
        :return:
        '''
        with open(self._save_directory_txt + "/" + self._object_id + "_" + _add_str + ".txt", "w") as output:
            output.write(str(_list))

    def create_item_one_epoch_list_dict(self, **kwargs):
        '''
        :param kwargs: such as 'path_list', etc. 'index_list' shall not be here. ''index_list' shall be saved in
        'self._index_list'
        :return:
        '''
        self._one_epoch_list_dict.update(kwargs)

    def create_item_tendency_list_dict(self, **kwargs):
        '''
        If needed.
        :param kwargs:
        :return:
        '''
        self._tendency_list_dict.update(kwargs)

    def reset_one_epoch_list(self):
        '''
        Reset EVERY values (lists) in 'self._one_epoch_list_dict'
        :return:
        '''
        self._index_list=[]
        for k, _ in self._one_epoch_list_dict.items():
            self._one_epoch_list_dict[k] = []

    def get_item_one_epoch_list(self, key):
        return self._one_epoch_list_dict[key]

    def get_item_tendency_list(self, key):
        return self._tendency_list_dict[key]

    def update_one_epoch_list(self, **kwargs):
        for key, value_list in kwargs.items():
            self._one_epoch_list_dict[key].extend(value_list)

    def update_tendency_list(self, **kwargs):
        for key, value_list in kwargs.items():
            self._tendency_list_dict[key].extend(value_list)

    def update_tendency_list_by_averaging(self, *args):
        '''
        Averages specified lists in dictionary and reset ALL the lists in a dictionary
        :param args: String type key to average list
        :return:
        '''
        for arg in args.__iter__():
            if len(self._one_epoch_list_dict[arg]) > 0:
                # try:
                self._tendency_list_dict[arg].append(sum(self._one_epoch_list_dict[arg]) /
                                                         float(len(self._one_epoch_list_dict[arg])))
                # except:
                #     print(f'arg: {arg}, length: {len(self._one_epoch_list_dict[arg])}')
                #     print(f'# of patients: {len(self._one_epoch_list_dict["patient_number"])}, patient number: {self._one_epoch_list_dict["patient_number"]}')
            else:
                print("Empty list for <" + arg + "> !!!")

    def update_tendency_list_by_specific_number(self, keys, denominators):
        '''
        Averages specified lists in dictionary and reset ALL the lists in a dictionary
        :param args: String type key to average list
        :return:
        '''
        for idx in range(len(keys)):
            if len(self._one_epoch_list_dict[keys[idx]]) > 0:
                    self._tendency_list_dict[keys[idx]].append(sum(self._one_epoch_list_dict[keys[idx]]) /
                                                         float(denominators[idx]))
            else:
                print("Empty list for <" + keys[idx] + "> !!!")

    def update_index_list(self, _list):
        self._index_list.extend(_list)

    def save_tendency_list_as_txt(self, *args):
        for arg in args.__iter__():
            self._save_txt(self._tendency_list_dict[arg], arg)

    def _sort_one_epoch_list_dict_all(self):
        '''
        Sort EVERY 'value's (list) of 'self._one_epoch_list_dict' by 'self._index_list'
        :return:
        '''
        if len(self._index_list) < 1:
            raise ValueError("Error at <_sort_list_dict_all> !!! Empty 'self._index_list'!!!")
        for key, value in self._one_epoch_list_dict.items():
            self._one_epoch_list_dict[key] = list([x for _, x in sorted(zip(self._index_list, value))])

    def _record_last_performance_of_tendency_list_dict(self, *args):
        _strg = ""
        for arg in args.__iter__():
            _strg = _strg + str(arg) + ": " + str(self._tendency_list_dict[arg][-1]) + ", "
        with open(self._save_directory_txt + "/" + self._object_id + "_RESULT.txt", "w") as output:
            output.write(_strg)

    def make_excel_from_one_epoch_list_dict(self, _epoch, *args):
        '''
        :param _epoch: number not string
        :param args: Better start with path_list
        :return:
        '''
        if self._if_sort:
            self._sort_one_epoch_list_dict_all()

        _workbook = xlsxwriter.Workbook(
            self._save_directory_excel + "/" + self._object_id + "_epoch_" + str(_epoch) + ".xlsx")

        _worksheet = _workbook.add_worksheet('INFO')
        _evaluation_metric = _workbook.add_format({'num_format': '0.0000'})
        _bold = _workbook.add_format({'bold': True})

        i = 0
        for arg in args.__iter__():
            _worksheet.write(0, i, arg, _bold)

            for row in range(len(self._index_list)):
                # try:
                _worksheet.write((row + 1), i, (self._one_epoch_list_dict[arg])[row])
                # except:
                #     print(f'row: {row}, arg: {arg}')
                #     print(f'Content: {(self._one_epoch_list_dict[arg])[row]}')
            i = i + 1

        _workbook.close()