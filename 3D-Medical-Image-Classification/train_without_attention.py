import torch
import random
import numpy as np
from torchvision import utils
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import data_provider
import torch.optim as optim
import network
import torch.nn.functional as F
import performance_assess
from torchvision import transforms
from operator import add
import time
import losses as custom_loss
from PIL import Image

torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
# https://keep-steady.tistory.com/14
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# training set: 511
# validation set: 56

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(99)
# random.seed(99)

csv_tr = 'D:/Rectum_exp/Data/data_path_3d/training_fold_1_ex.csv'
csv_val = 'D:/Rectum_exp/Data/data_path_3d/validation_fold_1.csv'
result_save_directory = 'D:/Rectum_exp/Data/result_3d/experiment1/NO-attention-maxpool'
my_cuda_device_number = 0
my_cuda_device = torch.device("cuda:"+str(my_cuda_device_number))
max_epoch=500
my_batch_size = 32

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
net = network.Net_res3D_max().to(my_cuda_device)

# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.AdamW(net.parameters(), lr=0.002)
# optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay = 0.003)
optimizer = optim.AdamW(net.parameters(), lr=0.001)
# dataiter = iter(dataloader_tr)
# images, _, labels, patient_number = dataiter.next()

ds_tr = data_provider.CustomDataset_without_mask(csv_tr, transform=True, mask_resize_ratio=8)
ds_val = data_provider.CustomDataset_without_mask(csv_val, transform=False, mask_resize_ratio=8)
# dataloader = DataLoader(ds_tr, batch_size=4, shuffle=True, num_workers=1, collate_fn=collate_fn)
dataloader_tr = DataLoader(ds_tr, batch_size=my_batch_size, shuffle=True)
dataloader_val = DataLoader(ds_val, batch_size=my_batch_size, shuffle=False)

performance_recorder_validationset = performance_assess.Performance_recorder_txt(
            save_directory_txt=result_save_directory,
            save_directory_excel=result_save_directory, object_id='ValidationSet',
            accuracy=[], sensitivity=[], specificity=[], loss_category=[], probability=[], prediction=[], category=[], patient_number=[])

performance_recorder_trainingset = performance_assess.Performance_recorder_txt(
            save_directory_txt=result_save_directory,
            save_directory_excel=result_save_directory, object_id='TrainingSet',
            accuracy=[], sensitivity=[], specificity=[], loss_category=[], probability=[], prediction=[], category=[], patient_number=[])

writer = SummaryWriter(result_save_directory)

step_number = 0


for epoch in range(max_epoch):
    # bf_epoch=time.time()
    for i, data_tr in enumerate(dataloader_tr, 0):
        # print("one epoch: " + str(time.time() - bf_epoch))
        # before_datagen = time.time()
        inputs_tr_tensor, labels_temp_tr_tensor, patient_number_temp_tr_tensor = data_tr

        inputs_tr_tensor= inputs_tr_tensor.cuda(my_cuda_device)
        labels_temp_tr_tensor = labels_temp_tr_tensor.cuda(my_cuda_device)
        optimizer.zero_grad()
        # print("inputs_tr_tensor: "+str(inputs_tr_tensor.size()))
        logits_category_tr = net(inputs_tr_tensor)
        # print("ff: "+str(time.time()-bfff))
        # torch.nn.Conv3d : (N, C, D(3rd dim), H, W)
        # bfcompute = time.time()
        logits_category_tr = ((logits_category_tr).double()).view(inputs_tr_tensor.size()[0])

        loss_BCE_category_tr = F.binary_cross_entropy_with_logits(logits_category_tr, labels_temp_tr_tensor, reduction='none')
        loss_focal_category_tr = (((1-torch.exp(-loss_BCE_category_tr))**4)*loss_BCE_category_tr)*(1.0+((1.0-labels_temp_tr_tensor)*208.0/151.0))

        probs_temp_tr_tensor = torch.sigmoid(logits_category_tr)
        preds_temp_tr_tensor = probs_temp_tr_tensor.round()

        # print("computing: " + str(time.time() - bfcompute))
        # print("dice: "+str(dice_score_tr.tolist()))
        # bfbp = time.time()
        # with torch.cuda.device(0):
        (torch.mean(loss_focal_category_tr)).backward()
        optimizer.step()
        # print("bp: " + str(time.time() - bfbp))

        # bfrecording = time.time()

        performance_recorder_trainingset.update_one_epoch_list(probability=probs_temp_tr_tensor.tolist(),
                                   prediction=preds_temp_tr_tensor.tolist(),
                                   accuracy=(preds_temp_tr_tensor == labels_temp_tr_tensor).tolist(),
                                   sensitivity=(preds_temp_tr_tensor * labels_temp_tr_tensor).tolist(),
                                   specificity=((1 - preds_temp_tr_tensor) * (1 - labels_temp_tr_tensor)).tolist(),
                                   category=labels_temp_tr_tensor.tolist(),
                                   patient_number=patient_number_temp_tr_tensor.tolist(),
                                   loss_category=loss_focal_category_tr.tolist())


        performance_recorder_trainingset.update_index_list(patient_number_temp_tr_tensor.tolist())
        # print("b recording: " + str(time.time() - bfrecording))
        step_number += 1

    for i_val, data_val in enumerate(dataloader_val, 0):
        # print("one epoch: " + str(time.time() - bf_epoch))
        # before_datagen = time.time()
        inputs_val_tensor, labels_temp_val_tensor, patient_number_temp_val_tensor = data_val

        inputs_val_tensor= inputs_val_tensor.cuda(my_cuda_device)
        labels_temp_val_tensor = labels_temp_val_tensor.cuda(my_cuda_device)
        logits_category_val = net(inputs_val_tensor)
        # print("ff: "+str(time.time()-bfff))
        # torch.nn.Conv3d : (N, C, D(3rd dim), H, W)
        # bfcompute = time.time()
        logits_category_val = ((logits_category_val).double()).view(inputs_val_tensor.size()[0])

        loss_BCE_category_val = F.binary_cross_entropy_with_logits(logits_category_val, labels_temp_val_tensor, reduction='none')
        loss_focal_category_val = (((1-torch.exp(-loss_BCE_category_val))**4)*loss_BCE_category_val)*(1.0+((1.0-labels_temp_val_tensor)*208.0/151.0))

        probs_temp_val_tensor = torch.sigmoid(logits_category_val)
        preds_temp_val_tensor = probs_temp_val_tensor.round()

        performance_recorder_validationset.update_one_epoch_list(probability=probs_temp_val_tensor.tolist(),
                                   prediction=preds_temp_val_tensor.tolist(),
                                   accuracy=(preds_temp_val_tensor == labels_temp_val_tensor).tolist(),
                                   sensitivity=(preds_temp_val_tensor * labels_temp_val_tensor).tolist(),
                                   specificity=((1 - preds_temp_val_tensor) * (1 - labels_temp_val_tensor)).tolist(),
                                   category=labels_temp_val_tensor.tolist(),
                                   patient_number=patient_number_temp_val_tensor.tolist(),
                                   loss_category=loss_focal_category_val.tolist())


        performance_recorder_validationset.update_index_list(patient_number_temp_val_tensor.tolist())

    # print("inputs_val_tensor: "+str(inputs_val_tensor.size()))
    # inputs_val_tensor: torch.Size([3 or 4, 1, 7, 256, 256])
    writer.add_graph(net, inputs_val_tensor)
    performance_recorder_trainingset.update_tendency_list_by_averaging('accuracy', 'loss_category')
    performance_recorder_validationset.update_tendency_list_by_averaging('accuracy', 'loss_category')
    performance_recorder_trainingset.update_tendency_list_by_specific_number(['sensitivity', 'specificity'],
                                         [sum(performance_recorder_trainingset.get_item_one_epoch_list('category')),
                                          len(performance_recorder_trainingset.get_item_one_epoch_list('category'))-
                                              sum(performance_recorder_trainingset.get_item_one_epoch_list('category'))])

    performance_recorder_validationset.update_tendency_list_by_specific_number(['sensitivity', 'specificity'],
                                         [sum(performance_recorder_validationset.get_item_one_epoch_list('category')),
                                          len(performance_recorder_validationset.get_item_one_epoch_list('category')) -
                                          sum(performance_recorder_validationset.get_item_one_epoch_list('category'))])

    performance_recorder_trainingset.save_tendency_list_as_txt('accuracy', 'sensitivity', 'specificity', 'loss_category')
    performance_recorder_validationset.save_tendency_list_as_txt('accuracy', 'sensitivity', 'specificity', 'loss_category')

    performance_recorder_trainingset.make_excel_from_one_epoch_list_dict(epoch, 'category',
                                                                         'patient_number', 'probability',
                                                                         'prediction', 'accuracy',
                                                                         'sensitivity', 'specificity', 'loss_category')
    performance_recorder_validationset.make_excel_from_one_epoch_list_dict(epoch, 'category',
                                                                         'patient_number', 'probability',
                                                                         'prediction', 'accuracy',
                                                                         'sensitivity', 'specificity', 'loss_category')


    performance_recorder_trainingset.reset_one_epoch_list()
    performance_recorder_validationset.reset_one_epoch_list()

    print("Epoch: "+str(epoch))
    print("step_number: "+str(step_number))
    # print("labels_overall_tr ("+str(len(labels_tr))+"): "+str(labels_tr))
    # print("labels_overall_val (" + str(len(labels_val)) + "): " + str(labels_val))
    # print("acc_tr (" + str(len(acc_tr)) + "): " + str(acc_tr))
    # print("acc_val (" + str(len(acc_val)) + "): " + str(acc_val))
    # print("patient_number_tr (" + str(len(patient_number_tr)) + "): " + str(patient_number_tr))
    # print("patient_number_val (" + str(len(patient_number_val)) + "): " + str(patient_number_val))

    # acc_overall_tr.append(float(sum(acc_tr)) / float(len(acc_tr)))
    # acc_overall_val.append(float(sum(acc_val)) / float(len(acc_val)))
    # sens_overall_tr.append(float(sum(sens_tr)) / float(sum(labels_tr)))
    # sens_overall_val.append(float(sum(sens_val)) / float(sum(labels_val)))
    # spec_overall_tr.append(float(sum(spec_tr)) / float(len(labels_tr)-sum(labels_tr)))
    # spec_overall_val.append(float(sum(spec_val)) / float(len(labels_val)-sum(labels_val)))
    # loss_overall_tr.append(float(sum(loss_tr)) / float(len(loss_tr)))
    # loss_overall_val.append(float(sum(loss_val)) / float(len(loss_val)))
    #
    # writer.add_scalar('training focal loss', loss_overall_tr[-1], epoch)
    # writer.add_scalar('validation focal loss', loss_overall_val[-1], epoch)
    # writer.add_scalar('training accuracy', acc_overall_tr[-1], epoch)
    # writer.add_scalar('validation accuracy', acc_overall_val[-1], epoch)
    # writer.add_scalar('training sensitivity', sens_overall_tr[-1], epoch)
    # writer.add_scalar('validation sensitivity', sens_overall_val[-1],  epoch)
    # writer.add_scalar('training specificity', spec_overall_tr[-1], epoch)
    # writer.add_scalar('validation specificity', spec_overall_val[-1], epoch)
    #
    # with open(result_save_directory + "/training focal loss.txt", "w") as output:
    #     output.write(str(loss_overall_tr))
    # with open(result_save_directory + "/validation focal loss.txt", "w") as output:
    #     output.write(str(loss_overall_val))
    # with open(result_save_directory + "/training acc.txt", "w") as output:
    #     output.write(str(acc_overall_tr))
    # with open(result_save_directory + "/validation acc.txt", "w") as output:
    #     output.write(str(acc_overall_val))
    # with open(result_save_directory + "/training sens.txt", "w") as output:
    #     output.write(str(sens_overall_tr))
    # with open(result_save_directory + "/validation sens.txt", "w") as output:
    #     output.write(str(sens_overall_val))
    # with open(result_save_directory + "/training spec.txt", "w") as output:
    #     output.write(str(spec_overall_tr))
    # with open(result_save_directory + "/validation spec.txt", "w") as output:
    #     output.write(str(spec_overall_val))
    #
    # with open(result_save_directory + "/latest epoch training patient.txt", "w") as output:
    #     output.write(str(patient_number_tr))
    # with open(result_save_directory + "/latest epoch validation patient.txt", "w") as output:
    #     output.write(str(patient_number_val))
    # with open(result_save_directory + "/latest epoch training prob.txt", "w") as output:
    #     output.write(str(probs_tr))
    # with open(result_save_directory + "/latest epoch validation prob.txt", "w") as output:
    #     output.write(str(probs_val))
    # with open(result_save_directory + "/latest epoch training focal loss.txt", "w") as output:
    #     output.write(str(loss_tr))
    # with open(result_save_directory + "/latest epoch validation focal loss.txt", "w") as output:
    #     output.write(str(loss_val))
    # with open(result_save_directory + "/latest epoch training label.txt", "w") as output:
    #     output.write(str(labels_tr))
    # with open(result_save_directory + "/latest epoch validation label.txt", "w") as output:
    #     output.write(str(labels_val))

print('Training Finished')
# writer.close()
