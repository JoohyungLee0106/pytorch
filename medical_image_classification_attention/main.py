import training
import networks
import os
import torch

CSV_TR = 'D:/Rectum_exp/Data/data_path_3d_new/2D_training_fold_'
CSV_VAL = 'D:/Rectum_exp/Data/data_path_3d_new/2D_validation_fold_'


# Everything will be saved under: RESULT_SAVE_DIRECTORY+'/'+SAVE_FOLDER_NAME
RESULT_SAVE_DIRECTORY = 'D:/Rectum_exp/Data/20200408_att_cancer'
try:
    os.mkdir(RESULT_SAVE_DIRECTORY)
except:
    print('RESULT_SAVE_DIRECTORY already exists!!!')

MY_DEVICE = "cuda:0"
MAX_EPOCH=500
NUM_WORKERS=3
FOLD_SIZE=10
OUTPUT_NUM = 3
# resnet18, resnet34, resnet50, resnext50, attention_dilated_2D, attention_dilated_with_cancer_2D
NETWORK = networks.attention_dilated_with_cancer_2D
SAVE_FOLDER_NAME = 'avp_16'
FEATURE_NUM=16
MINI_BATCH_SIZE = 16
PARAMS_NETWORK = {'pretrained':False,'dilations':True, 'global_pooling': 'avp','arch':'resnet34', 'block':networks.BasicBlock,
                  'base_width':FEATURE_NUM,'starting_filter_num':FEATURE_NUM, 'lated': False, 'my_device': torch.device(MY_DEVICE)}

if __name__ == '__main__':
    res34_train = training.Training(path_csv_before_fold_num_tr=CSV_TR, path_csv_before_fold_num_val=CSV_VAL,
                           result_save_directory=RESULT_SAVE_DIRECTORY, save_folder_name=SAVE_FOLDER_NAME,
                           my_device=MY_DEVICE, mini_batch_size=MINI_BATCH_SIZE, network=NETWORK,
                           params_network= PARAMS_NETWORK, output_num=OUTPUT_NUM, num_workers= NUM_WORKERS,
                            max_epoch=MAX_EPOCH, fold_size=FOLD_SIZE)

    res34_train()
