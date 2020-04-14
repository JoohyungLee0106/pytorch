import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import data_provider
import networks
import performance_assess
import os
import loss

class Training:
    def __init__(self, path_csv_before_fold_num_tr, path_csv_before_fold_num_val, result_save_directory, save_folder_name,
                 my_device, mini_batch_size, network, params_network, output_num, num_workers, image_dimension=2,
                 fold_size=10, max_epoch=500, learning_rate_threshold=0.000009, optimizer=optim.SGD, lr_scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                 lr_division_factor=0.3, lr_patience=6, mask_ratio=4.0, dataset=data_provider.CustomDataset_2D_with_mask,
                 dataloader=DataLoader, oversampler=torch.utils.data.sampler.WeightedRandomSampler,
                 perf_recorder=performance_assess.Performance_recorder_txt):
        print("Initializing...")
        self.path_csv_before_fold_num_tr=path_csv_before_fold_num_tr
        self.path_csv_before_fold_num_val=path_csv_before_fold_num_val
        self.result_save_directory= os.path.join(result_save_directory, save_folder_name,'fold_')
        self.root_directory = os.path.join(result_save_directory, save_folder_name)
        # if os.path.isdir(self.root_directory):
        # if False:
        #     raise ValueError("<Class Training> Root directory already exists!!")
        # else:
        try:
            os.mkdir(self.root_directory)
        except:
            raise ValueError('Save folder already exists!')

        self.my_device = torch.device(my_device)
        self.mini_batch_size=mini_batch_size
        self.network = network
        self.output_num = output_num
        self.image_dimension = image_dimension
        self.fold_size=fold_size
        self.max_epoch=max_epoch
        self.learning_rate_threshold=learning_rate_threshold
        self.optimizer=optimizer
        self.lr_scheduler=lr_scheduler
        self.mask_ratio=mask_ratio

        self.dataset = dataset
        self.dataloader=dataloader
        self.oversampler=oversampler
        self.perf_recorder=perf_recorder
        self.loss_list = []
        for i in range(1, self.output_num+1):
            self.loss_list.append('loss_'+str(i))
        self.acc_list = []
        for i in range(1, self.output_num + 1):
            self.acc_list.append('accuracy_' + str(i))

        self.what_to_average = ['loss_overall']
        self.what_to_average.extend(self.acc_list)
        self.what_to_average.extend(self.loss_list)
        self.what_to_save = list(self.what_to_average)
        self.what_to_save.extend(['sensitivity_1', 'specificity_1'])
        self.what_for_excel = ['category', 'patient_number', 'prediction_1',
                          'sensitivity_1', 'specificity_1', 'loss_1']
        self.what_for_excel.extend(self.acc_list)
        self.what_for_excel.extend(self.loss_list)


        self.__mode_set()

        self.params_optimizer = {'lr':0.01, 'momentum':0.9, 'weight_decay':1e-2, 'nesterov':False}
        self.params_lr_scheduler = {'mode':'min', 'factor':lr_division_factor, 'patience':lr_patience}
        self.params_dataloader = {'batch_size':mini_batch_size, 'shuffle':False, 'num_workers':num_workers}

        self.reset_params_perf_recorder()

        self.params_network = params_network

        # self.loss_classes = [nn.BCEWithLogitsLoss(reduction='none'), loss.FocalLoss(gamma=3), loss.FocalLoss(gamma=3, if_custom_mean=True)]
        # self.loss_classes = [nn.BCEWithLogitsLoss(reduction='none'), loss.DiceLoss(),
        #                      loss.DiceLoss()]
        self.loss_classes = [loss.FocalLoss(gamma=2, if_logit=True), loss.DiceLoss(),
                             loss.DiceLoss()]


        self.zero_grad = self.zero_grad_tr

        self.acc_functions = [self.acc, self.dice, self.dice]
        self.to_preds_dc = [self.logits_to_pred_dc, self.probs_to_pred_dc, self.probs_to_pred_dc]
        # print(f'output_num: {self.output_num}')
        if self.output_num == 1:
            # print('output_num 1 triggered!!')
            self.make_label = self.make_label_1
            self.preds_to_list = self.single_pred_to_list
        elif self.output_num == 2:
            self.preds_to_list = self.multiple_preds_to_list
            self.make_label = self.make_label_2
        elif self.output_num == 3:
            self.preds_to_list = self.multiple_preds_to_list
            if self.params_network['dilations']:
                self.make_label = self.make_label_3_cancer_dilated
            else:
                self.make_label = self.make_label_3_cancer_undilated
        else:
            raise ValueError("<Class Training> Invalid Argument: output_num")
    def reset_params_perf_recorder(self):
        # self.params_performance_recorder = {'accuracy':[], 'sensitivity':[], 'specificity':[], 'loss_category':[],
        #                                     'probability':[], 'prediction':[], 'category':[], 'patient_number':[]}
        params_performance_recorder = {'sensitivity_1': [], 'specificity_1': [],
                                            'probability_1': [], 'prediction_1': [], 'category': [],
                                            'patient_number': []}
        for i in range(1, self.output_num + 1):
            params_performance_recorder.update({'accuracy_' + str(i): []})
            params_performance_recorder.update({'loss_' + str(i): []})
        params_performance_recorder.update({'loss_overall': []})
        return params_performance_recorder

    def logits_to_pred_dc(self, logits):
        _temp = logits.clone()
        return F.relu(_temp.sign())


    def probs_to_pred_dc(self, probs):
        _temp = probs.clone()
        _temp[_temp<0.5]=0.0
        _temp[_temp >= 0.5] = 1.0
        return _temp

    def zero_grad_tr(self, optimizer):
        optimizer.zero_grad()
    def zero_grad_val(self, optimizer):
        pass
    def acc(self, pred, label):
        '''
        :param pred: Torch.tensor
        :param label: Torch.tensor
        :return:
        '''
        return (pred == label).tolist()

    def dice(self, pred, label):
        '''
        :param pred: NHW, Torch.tensor
        :param label: NHW, Torch.tensor
        :return: N
        '''
        # print(f'pred.size(): {pred.size()}, label.size(): {label.size()}')
        # print(f'tuple(range(1,self.image_dimension+1)): {tuple(range(1,self.image_dimension+1))}')
        intersection = torch.sum(pred*label, dim=tuple(range(1,self.image_dimension+1)), keepdim=False)+0.0000001
        denominator = torch.sum(pred+label, dim=tuple(range(1,self.image_dimension+1)), keepdim=False)+0.0000001
        # print(f'intersection.size(): {intersection.size()}, denominator.size(): {denominator.size()}')
        return (intersection*2.0/denominator).tolist()

    def __mode_set(self):
        torch.autograd.set_detect_anomaly(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.fastest = True
    def single_pred_to_list(self, x):
        return [x]

    def multiple_preds_to_list(self, x):
        return list(x)

    def set_sampler(self, __fold):
        csv_file=pd.read_csv(self.path_csv_before_fold_num_tr + str(__fold) + '.csv')
        stage = csv_file['t-stage']

        label, counts = np.unique(stage, return_counts=True)
        weights = 1.0 / torch.tensor(counts, dtype=torch.float)
        weights = weights.double()

        temp = stage.to_numpy(dtype='double')
        temp2 = stage.to_numpy(dtype='double')
        ww = weights.numpy()

        temp2[np.where(temp == label[0])] = ww[0]
        temp2[np.where(temp == label[1])] = ww[1]

        sample_weights = torch.from_numpy(temp2)
        return torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))

    def get_last_perf(self):
        _temp= {'sensitivity_1':[], 'specificity_1':[]}
        for i in range(1, self.output_num+1):
            _temp.update({ 'accuracy_'+str(i): [] })
        return _temp

    def train(self, __fold):
        csv_tr = self.path_csv_before_fold_num_tr + str(__fold) + '.csv'
        csv_val = self.path_csv_before_fold_num_val + str(__fold) + '.csv'

        self.model_record = {'best_model_epoch': [], 'reduced_lr_epoch': []}

        model = self.network(**self.params_network).to(self.my_device)
        __optimizer = self.optimizer(model.parameters(), **self.params_optimizer)
        __lr_scheduler = self.lr_scheduler(__optimizer, **self.params_lr_scheduler)

        __dataset_tr = self.dataset(csv_tr, transform=True, mask_ratio = self.mask_ratio)
        __dataset_val = self.dataset(csv_val, transform=False, mask_ratio=self.mask_ratio)

        __sampler = self.set_sampler(__fold)
        __dataloader_tr = self.dataloader(__dataset_tr, sampler=__sampler, drop_last=True, **self.params_dataloader)
        __dataloader_val = self.dataloader(__dataset_val, **self.params_dataloader)

        performance_recorder_tr = self.perf_recorder(save_directory_txt = self.directory_fold,
                                                          save_directory_excel = self.directory_fold,
                                                          object_id='TrainingSet', if_sort = False, learning_rate=[],
                                                          **(self.reset_params_perf_recorder()))
        performance_recorder_val = self.perf_recorder(save_directory_txt=self.directory_fold,
                                                          save_directory_excel=self.directory_fold,
                                                          object_id='ValidationSet', **(self.reset_params_perf_recorder()))

        best_epoch = 0
        lr_reduction_num = 0
        for epoch in range(self.max_epoch):
            # bf_epoch=time.time()
            torch.set_grad_enabled(True)
            model.train()
            # self.plist=[]
            for i, data in enumerate(__dataloader_tr, 0):
                __optimizer.zero_grad()
                loss_overall = self.feed_forward(data, performance_recorder_tr, model, 'Training')
                (torch.mean(loss_overall)).backward()
                __optimizer.step()
            torch.set_grad_enabled(False)
            model.eval()
            for i, data in enumerate(__dataloader_val, 0):
                _ = self.feed_forward(data, performance_recorder_val, model, 'Validation')
            learning_rate_old = __optimizer.param_groups[-1]['lr']
            performance_recorder_tr.update_tendency_list(learning_rate=[learning_rate_old])

            self.update_performance_recorder(performance_recorder_tr, epoch)
            self.update_performance_recorder(performance_recorder_val, epoch)

            performance_recorder_tr.save_tendency_list_as_txt('learning_rate')
            __lr_scheduler.step(list(performance_recorder_tr.get_item_tendency_list('loss_1'))[-1])

            if (list(performance_recorder_tr.get_item_tendency_list('loss_1'))[-1] == __lr_scheduler.best):
                torch.save(model.state_dict(), self.directory_fold + '/model_params.pth')
                self.model_record['best_model_epoch'].append(epoch)
                best_epoch = epoch
                # print(f'best_epoch: {best_epoch} !!!')
            #     if learning rate is updated
            elif learning_rate_old != __optimizer.param_groups[-1]['lr']:
                self.model_record['reduced_lr_epoch'].append(epoch)
                model.load_state_dict(torch.load(self.directory_fold + '/model_params.pth'))
                lr_reduction_num += 1
                if lr_reduction_num == 3:
                    print(f'BEST(3) epoch: {best_epoch}, FOLD: {__fold}')
                    torch.save(model, self.directory_fold + '/model_3.pth')
                    for key in self.last_perf_tr_3.keys():
                        self.last_perf_tr_3[key].append(
                            performance_recorder_tr.get_item_tendency_list(key)[best_epoch])
                    for key in self.last_perf_val_3.keys():
                        self.last_perf_val_3[key].append(
                            performance_recorder_val.get_item_tendency_list(key)[best_epoch])

                elif lr_reduction_num == 4:
                    print(f'BEST(4) epoch: {best_epoch}, FOLD: {__fold}')
                    torch.save(model, self.directory_fold + '/model_4.pth')
                    for key in self.last_perf_tr_4.keys():
                        self.last_perf_tr_4[key].append(
                            performance_recorder_tr.get_item_tendency_list(key)[best_epoch])
                    for key in self.last_perf_val_4.keys():
                        self.last_perf_val_4[key].append(
                            performance_recorder_val.get_item_tendency_list(key)[best_epoch])

                elif lr_reduction_num == 5:
                    print(f'BEST(5) epoch: {best_epoch}, FOLD: {__fold}')
                    torch.save(model, self.directory_fold + '/model_5.pth')
                    for key in self.last_perf_tr_5.keys():
                        self.last_perf_tr_5[key].append(
                            performance_recorder_tr.get_item_tendency_list(key)[best_epoch])
                    for key in self.last_perf_val_5.keys():
                        self.last_perf_val_5[key].append(
                            performance_recorder_val.get_item_tendency_list(key)[best_epoch])

            performance_recorder_tr.reset_one_epoch_list()
            performance_recorder_val.reset_one_epoch_list()

            print("Epoch: " + str(epoch))

            if (__optimizer.param_groups[-1]['lr'] < self.learning_rate_threshold):
                print(f'Early stopped after finishing epoch {epoch}')
                del performance_recorder_tr
                del performance_recorder_val
                break


    def update_performance_recorder(self, performance_recorder, epoch):

        performance_recorder.update_tendency_list_by_averaging(*self.what_to_average)
        performance_recorder.update_tendency_list_by_specific_number(['sensitivity_1', 'specificity_1'],
                                         [sum(performance_recorder.get_item_one_epoch_list('category')),
                                          len(performance_recorder.get_item_one_epoch_list('category')) -
                                          sum(performance_recorder.get_item_one_epoch_list('category'))])

        performance_recorder.save_tendency_list_as_txt(*self.what_to_save)
        performance_recorder.make_excel_from_one_epoch_list_dict(epoch, *self.what_for_excel)


    def update_probs_preds(self, logits, list_prob, list_pred):
        list_prob.append(torch.sigmoid(logits))
        list_pred.append(probs_temp_tr_tensor.round())

    # output length 별로 다른 function (label_mask, label_clasf -> labels) : interpolate 안해도 되게
    # 0은 그냥 하고 1 부터 range로 돌리기?

    def make_label_1(self, label_clasf, label_mask):
        # print('make_label_1 triggered!!')
        return [label_clasf.to(self.my_device)]

    def make_label_2(self, label_clasf, label_mask):
        # print('make_label_2 triggered!!')
        return [label_clasf.to(self.my_device), (label_mask[:,1,:,:]).to(self.my_device)]

    def make_label_3_cancer_dilated(self, label_clasf, label_mask):
        # print('make_label_3 triggered!!')
        return [label_clasf.to(self.my_device), (label_mask[:,1,:,:]).to(self.my_device),\
               (label_mask[:, 2, :, :]).to(self.my_device)]

    def make_label_3_cancer_undilated(self, label_clasf, label_mask):
        return [label_clasf.to(self.my_device), (label_mask[:,1,:,:]).to(self.my_device),\
                ((F.interpolate(label_mask[:, 2:3, :, :],\
                   int(label_mask.size(3) / 2.0),\
                   mode='bicubic', align_corners=True)).view(
        label_mask.size(0), int(label_mask.size(2)/ 2.0),\
                   int(label_mask.size(3)/ 2.0))).to(self.my_device)]


    def feed_forward(self, data, performance_recorder, model, id):
        # print(f'id: {id}')

        loss_overall=0
        list_pred = []
        performannce_update_dict={}

        image, label_mask, label_clasf, patient_number = data

        image = image.to(self.my_device)
        patient_number = list(patient_number)
        labels = self.make_label(label_clasf, label_mask)
        list_output = self.preds_to_list(model(image))
        print(f'<ATTENTION> mean: {torch.mean(list_output[1])}, min: {torch.min(list_output[1])}, max: {torch.max(list_output[1])}')
        print(f'<CANCER> mean: {torch.mean(list_output[2])}, min: {torch.min(list_output[2])}, max: {torch.max(list_output[2])}')

        # self.loss_classes[2].set_division_factor = torch.sum( (model.attention_mask_seg).sign(), dim=(1,2,3), keepdim=False)
        for i in range(self.output_num):
            loss = self.loss_classes[i]( list_output[i], labels[i] )
            list_pred.append( self.to_preds_dc[i](list_output[i]) )
            loss_overall = loss_overall + loss
            performannce_update_dict.update({'loss_' + str(i + 1): loss.tolist()})

            performannce_update_dict.update( {'accuracy_'+str(i+1): self.acc_functions[i](list_pred[i], labels[i])} )
        performannce_update_dict.update( {'sensitivity_1': (list_pred[0] * labels[0]).tolist(),
                                          'specificity_1': ( (1-list_pred[0]) * (1-labels[0]) ).tolist(),
                                          'probability_1': (list_output[0]).tolist(),
                                          'prediction_1': (list_pred[0]).tolist(), 'category': labels[0].tolist(),
                                           'patient_number': patient_number, 'loss_overall': loss_overall.tolist()})
        # print(f'loss_overall.tolist(): {loss_overall.tolist()}')
        performance_recorder.update_one_epoch_list(**performannce_update_dict)
        performance_recorder.update_index_list(patient_number)

        return loss_overall


    def __call__(self):
        print("called")
        self.last_perf_tr_3 = self.get_last_perf()
        self.last_perf_val_3 = self.get_last_perf()
        self.last_perf_tr_4 = self.get_last_perf()
        self.last_perf_val_4 = self.get_last_perf()
        self.last_perf_tr_5 = self.get_last_perf()
        self.last_perf_val_5 = self.get_last_perf()

        for fold in range(1, self.fold_size+1):
            self.fold=fold
            print(f'FOLD: {fold}')
            self.directory_fold = self.result_save_directory + str(fold)
            # if os.path.isdir(self.directory_fold):
            # if False:
            #     raise ValueError(f"<Class Training> Fold {fold} directory already exists!!")
            # else:
            try:
                os.mkdir(self.directory_fold)
            except:
                raise ValueError(f'Fold {fold} directory already exists')

            self.train(fold)

            with open(os.path.join(self.root_directory, 'model_record_'+ str(fold) + '.txt'), 'w') as f:
                print(self.model_record, file=f)
            with open(os.path.join(self.root_directory, 'record_tr3.txt'), 'w') as f:
                print(self.last_perf_tr_3, file=f)
            with open(os.path.join(self.root_directory, 'record_val3.txt'), 'w') as f:
                print(self.last_perf_val_3, file=f)
            with open(os.path.join(self.root_directory, 'record_tr4.txt'), 'w') as f:
                print(self.last_perf_tr_4, file=f)
            with open(os.path.join(self.root_directory, 'record_val4.txt'), 'w') as f:
                print(self.last_perf_val_4, file=f)
            with open(os.path.join(self.root_directory, 'record_tr5.txt'), 'w') as f:
                print(self.last_perf_tr_5, file=f)
            with open(os.path.join(self.root_directory, 'record_val5.txt'), 'w') as f:
                print(self.last_perf_val_5, file=f)

            # self.reset_params_perf_recorder()