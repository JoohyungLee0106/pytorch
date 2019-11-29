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

csv_tr = 'D:/Rectum_exp/Data/data_path_3d/training_fold_1.csv'
csv_val = 'D:/Rectum_exp/Data/data_path_3d/validation_fold_1.csv'
ds_tr = data_provider.CustomDataset(csv_tr, transform=True, mask_resize_ratio=8)
ds_val = data_provider.CustomDataset(csv_val, transform=False, mask_resize_ratio=8)
result_save_directory = 'D:/Rectum_exp/Data/result_3d/experiment1'
writer = SummaryWriter(result_save_directory)
net = network.Net_res3D_attention()
# dataloader = DataLoader(ds_tr, batch_size=4, shuffle=True, num_workers=1, collate_fn=collate_fn)
dataloader_tr = DataLoader(ds_tr, batch_size=4, shuffle=True)
dataloader_val = DataLoader(ds_val, batch_size=4, shuffle=False)


# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.AdamW(net.parameters(), lr=0.002)
# optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay = 0.003)
optimizer = optim.AdamW(net.parameters(), lr=0.001)
# dataiter = iter(dataloader_tr)
# images, _, labels, patient_number = dataiter.next()

step_number = 0

performance_recorder_validationset = performance_assess.Performance_recorder_txt(
            save_directory_txt=result_save_directory,
            save_directory_excel=result_save_directory, object_id='ValidationSet',
            accuracy=[], sensitivity=[], specificity=[], loss_category=[], loss_attention=[], dice_score=[],
            loss_overall=[],probability=[], prediction=[], category=[], patient_number=[])

performance_recorder_trainingset = performance_assess.Performance_recorder_txt(
            save_directory_txt=result_save_directory,
            save_directory_excel=result_save_directory, object_id='TrainingSet',
            accuracy=[], sensitivity=[], specificity=[], loss_category=[], loss_attention=[], dice_score=[],
            loss_overall=[], probability=[], prediction=[], category=[], patient_number=[])


acc_overall_tr = []
acc_overall_val = []
sens_overall_tr = []
sens_overall_val = []
spec_overall_tr = []
spec_overall_val = []
loss_overall_tr = []
loss_overall_val = []

for epoch in range(100):
    acc_tr = []
    acc_val = []
    sens_tr = []
    sens_val = []
    spec_tr = []
    spec_val = []

    loss_tr = []
    loss_val = []

    labels_tr = []
    labels_val = []
    patient_number_tr = []
    patient_number_val = []
    probs_tr = []
    probs_val = []
    preds_tr = []
    preds_val = []

    for i, data_tr in enumerate(dataloader_tr, 0):
        inputs_tr_tensor, mask_tr_tensor, labels_temp_tr_tensor, patient_number_temp_tr_tensor = data_tr
        # print("label unique: "+ str(torch.unique(mask_tr_tensor)))
        # torch.nn.Conv3d : (N, C, D(3rd dim), H, W)
        # print(str(patient_number_temp_tr_tensor.size()))
        # fig = plt.figure()
        # fig.add_subplot(2, 2, 1)
        # plt.imshow(inputs_tr_tensor[1,0,0,:,:], cmap='Greys_r')
        # fig.add_subplot(2, 2, 2)
        # plt.imshow(inputs_tr_tensor[1, 0, 1, :, :], cmap='Greys_r')
        # fig.add_subplot(2, 2, 3)
        # plt.imshow(inputs_tr_tensor[1, 0, 2, :, :], cmap='Greys_r')
        # fig.add_subplot(2, 2, 4)
        # plt.imshow(inputs_tr_tensor[1, 0, 3, :, :], cmap='Greys_r')
        # plt.title(str(patient_number_temp_tr_tensor[1]))
        # plt.show()
        # zero the parameter gradients
        optimizer.zero_grad()
# torch.nn.Conv3d : (N, C, D(3rd dim), H, W)

        # bfff=time.time()
        # with torch.cuda.device(0):
        logits_category_tr, logits_attention_tr = net(inputs_tr_tensor)
        # print("ff: "+str(time.time()-bfff))
        # torch.nn.Conv3d : (N, C, D(3rd dim), H, W)
        logits_category_tr = ((logits_category_tr).double()).view(inputs_tr_tensor.size()[0])
        logits_attention_tr = ((logits_attention_tr).double()).view(
            logits_attention_tr.size()[0], logits_attention_tr.size()[2], logits_attention_tr.size()[3], logits_attention_tr.size()[4])

        loss_BCE_category_tr = F.binary_cross_entropy_with_logits(logits_category_tr, labels_temp_tr_tensor, reduction='none')
        loss_focal_category_tr = ((1-torch.exp(-loss_BCE_category_tr))**4)*loss_BCE_category_tr


        # torch.nn.Conv3d : (N, C, D(3rd dim), H, W)
        loss_BCE_mask_tr = F.binary_cross_entropy_with_logits(logits_attention_tr, mask_tr_tensor[:,1,:,:,:].double(), reduction='none')
        loss_focal_mask_tr = torch.mean(((1 - torch.exp(-loss_BCE_mask_tr)) ** 4) * loss_BCE_mask_tr, dim = [1,2,3])
        loss_mtl_tr = loss_focal_category_tr+(loss_focal_mask_tr)

        preds_mask_tr = F.relu(logits_attention_tr).sign()
        dice_score_tr = 2.0 * torch.sum((preds_mask_tr*mask_tr_tensor[:,1,:,:,:].double()), dim=(1,2,3), keepdim=False)/\
                        (torch.sum(preds_mask_tr, dim=(1,2,3), keepdim=False)+
                         torch.sum(mask_tr_tensor[:,1,:,:,:].double(), dim=(1,2,3), keepdim=False))
        print("dice: "+str(dice_score_tr.tolist()))
        # bfbp = time.time()
        # with torch.cuda.device(0):
        (torch.mean(loss_mtl_tr)).backward()
        optimizer.step()
        # print("bp: " + str(time.time() - bfbp))
        probs_temp_tr_tensor = torch.sigmoid(logits_category_tr)
        preds_temp_tr_tensor = probs_temp_tr_tensor.round()

        performance_recorder_trainingset.update_one_epoch_list(probability=probs_temp_tr_tensor.tolist(),
                                   prediction=preds_temp_tr_tensor.tolist(),
                                   accuracy=(preds_temp_tr_tensor == labels_temp_tr_tensor).tolist(),
                                   sensitivity=(preds_temp_tr_tensor * labels_temp_tr_tensor).tolist(),
                                   specificity=((1 - preds_temp_tr_tensor) * (1 - labels_temp_tr_tensor)).tolist(),
                                   category=labels_temp_tr_tensor.tolist(), dice_score = dice_score_tr.tolist(),
                                   patient_number=patient_number_temp_tr_tensor.tolist(),
                                   loss_category=loss_focal_category_tr.tolist(), loss_attention=loss_focal_mask_tr.tolist(),
                                   loss_overall=loss_mtl_tr.tolist())

        performance_recorder_trainingset.update_index_list(patient_number_temp_tr_tensor.tolist())

        step_number += 1

    for i_val, data_val in enumerate(dataloader_val, 0):
        inputs_val_tensor, mask_val_tensor, labels_temp_val_tensor, patient_number_temp_val_tensor = data_val

        # torch.nn.Conv3d : (N, C, D(3rd dim), H, W)torch max
        logits_category_val, logits_attention_val = net(inputs_val_tensor)

        # torch.nn.Conv3d : (N, C, D(3rd dim), H, W)
        logits_category_val = ((logits_category_val).double()).view(inputs_val_tensor.size()[0])
        logits_attention_val = ((logits_attention_val).double()).view(
            logits_attention_val.size()[0], logits_attention_val.size()[2], logits_attention_val.size()[3], logits_attention_val.size()[4])

        loss_BCE_category_val = F.binary_cross_entropy_with_logits(logits_category_val, labels_temp_val_tensor, reduction='none')
        loss_focal_category_val = ((1 - torch.exp(-loss_BCE_category_val)) ** 4) * loss_BCE_category_val

        # torch.nn.Conv3d : (N, C, D(3rd dim), H, W)
        loss_BCE_mask_val = F.binary_cross_entropy_with_logits(logits_attention_val, mask_val_tensor[:, 1, :, :, :].double(), reduction='none')
        loss_focal_mask_val = torch.mean(((1 - torch.exp(-loss_BCE_mask_val)) ** 4) * loss_BCE_mask_val, dim=[1, 2, 3])
        loss_mtl_val = loss_focal_category_val + loss_focal_mask_val

        preds_mask_val = F.relu(logits_attention_val).sign()
        dice_score_val = 2.0 * torch.sum((preds_mask_val * mask_val_tensor[:, 1, :, :, :].double()), dim=(1, 2, 3), keepdim=False) / \
                        (torch.sum(preds_mask_val, dim=(1, 2, 3), keepdim=False) +
                         torch.sum(mask_val_tensor[:, 1, :, :, :].double(), dim=(1, 2, 3), keepdim=False))


        probs_temp_val_tensor = torch.sigmoid(logits_category_val)
        preds_temp_val_tensor = probs_temp_val_tensor.round()

        performance_recorder_validationset.update_one_epoch_list(probability=probs_temp_val_tensor.tolist(),
                                       prediction=preds_temp_val_tensor.tolist(),
                                       accuracy=(preds_temp_val_tensor == labels_temp_val_tensor).tolist(),
                                       sensitivity=(preds_temp_val_tensor * labels_temp_val_tensor).tolist(),
                                       specificity=((1 - preds_temp_val_tensor) * (1 - labels_temp_val_tensor)).tolist(),
                                       category=labels_temp_val_tensor.tolist(), dice_score = dice_score_val.tolist(),
                                       patient_number=patient_number_temp_val_tensor.tolist(),
                                       loss_category=loss_focal_category_val.tolist(),
                                       loss_attention=loss_focal_mask_val.tolist(),
                                       loss_overall=loss_mtl_val.tolist())

        performance_recorder_validationset.update_index_list(patient_number_temp_val_tensor.tolist())

    print("inputs_val_tensor: "+str(inputs_val_tensor.size()))
    writer.add_graph(net, inputs_val_tensor)
    performance_recorder_trainingset.update_tendency_list_by_averaging('accuracy', 'sensitivity', 'specificity', 'dice_score',
                                                                       'loss_category', 'loss_attention', 'loss_overall')
    performance_recorder_validationset.update_tendency_list_by_averaging('accuracy', 'sensitivity', 'specificity',
                                                                       'loss_category', 'loss_attention', 'loss_overall')

    performance_recorder_trainingset.save_tendency_list_as_txt('accuracy', 'sensitivity', 'specificity', 'dice_score',
                                                                       'loss_category', 'loss_attention', 'loss_overall')
    performance_recorder_validationset.save_tendency_list_as_txt('accuracy', 'sensitivity', 'specificity',
                                                                       'loss_category', 'loss_attention', 'loss_overall')

    performance_recorder_trainingset.make_excel_from_one_epoch_list_dict(epoch, 'category',
                                                                         'patient_number', 'probability',
                                                                         'prediction', 'accuracy',
                                                                         'sensitivity', 'specificity', 'dice_score',
                                                                         'loss_category', 'loss_attention',
                                                                         'loss_overall')
    performance_recorder_validationset.make_excel_from_one_epoch_list_dict(epoch, 'category',
                                                                         'patient_number', 'probability',
                                                                         'prediction', 'accuracy',
                                                                         'sensitivity', 'specificity', 'dice_score',
                                                                         'loss_category', 'loss_attention',
                                                                         'loss_overall')


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