## By Joohyung Lee, 2018-08-30, 14:55 p.m.
# Initial creation
## By Joohyung Lee, 2018-09-14
# Added a functionality that can edit the root directory.
import os
import csv
import random

# _path_until_patient_list must end with '/'
# _path_until_patient_list = "D:/Rectum_exp/Data/total_data_20181018/T3/"
_path_until_patient_list = "D:/Rectum_exp/bmp/T3/"

# _target_sub_path_after_patient = "image_3ch_crop"
_image_folder_name = "image_prep"
_label_folder_name = "Label_wo_gel"
# _label_folder_name = "Label_w_gel"
_category_name = "T3"
_save_path = "D:/Rectum_exp/Data/data_path_3d/"
_kfold = 10

# True 면 환자번호기준으로 shuffle, False 면 환자번호기준으로 sort 함
_if_random_shuffle = True

# etc. 라벨 말고 영상으로 하고, 두번 돌린다 (T2, T3)
def make_image_list( path_until_patient_list, image_folder_name, label_folder_name, save_path, kfold, category_name, if_random_shuffle ):
    # 0. 총 환자 list를 오리지널과 카피 두개 가지고 있는다, 기준은 누적으로!!
    # 1. 총 영상 갯수를 가져오고 kfold로 나눈다
    # 2. 환자 별 영상 갯수를 가져온다
    # 3. 총 영상 갯수/kfold와 가장 맞는 숫자로 환자를 자르고 fold별 첫 환자 index를 리스트로 저장한다
    # 4. 4. training set path list와 validation set path list 등을 저장한다

    ###
    # path_until_patient_list 는 /로 끝난다
    # image_suffix = something like *.bmp or *_roi.bmp
################################################
    if category_name == 'T2':
        t_stage_label = 0
    elif category_name == 'T3':
        t_stage_label = 1
    else:
        raise ValueError('Invalid Input param: category_name !!!')

# 0. 총 환자 list를 오리지널과 카피 두개 가지고 있는다, 기준은 누적으로!!
    # shuffle 이 True면 셔플, off면 sort
    ## Output: 오리지널 환자리스트, 환자리스트 카피본
    _patient_list_original = os.listdir(path_until_patient_list)
    if if_random_shuffle:
        random.shuffle(_patient_list_original)
    else:
        _patient_list_original.sort()


# 1. 총 환자 수를 가져오고 kfold로 나눈다
## Output: 총 영상 갯수/kfold
    patient_number_per_fold = float(len(_patient_list_original)) / float(kfold)

# 2. 환자 별 영상 갯수를 가져온다
# 3. 총 영상 갯수/kfold와 가장 맞는 숫자로 환자를 자르고 fold별 첫 환자 index를 리스트로 저장한다
# patient_starting_index_list[-1] 이 len(patient_list_copy) 와 같은지 확인한다. 11번째 starting index는 없는 index이다 (+1이 되므로)

    ## Output: 환자 별 영상 갯수, fold로 자를 환자 index, fold 별 영상 갯수
    patient_starting_index_list = [0]
    patient_number_per_fold_cumulative_list = []
    patient_number_per_fold_list = []

    patient_index_start = 0

    for fold_num in range(1, (kfold+1)):
        patient_list_copy = []
        patient_list_copy = list(_patient_list_original)
        patient_number_per_fold_cumulative_list.append(round(float(patient_number_per_fold) * float(fold_num)))
        print("patient_number_per_fold_cumulative_list: " + str(patient_number_per_fold_cumulative_list))
        patient_index_end = patient_number_per_fold_cumulative_list[-1]

        patient_list_val = list(patient_list_copy[patient_index_start : patient_index_end])
        del( patient_list_copy[patient_index_start : patient_index_end] )

        image_path_list_val = [os.path.join(path_until_patient_list, _patient_number, image_folder_name)
                            for _patient_number in patient_list_val]

        mask_path_list_val = [os.path.join(path_until_patient_list, _patient_number, label_folder_name)
                            for _patient_number in patient_list_val]

        image_path_list_tr = [os.path.join(path_until_patient_list, _patient_number, image_folder_name)
                            for _patient_number in patient_list_copy]

        mask_path_list_tr = [os.path.join(path_until_patient_list, _patient_number, label_folder_name)
                                           for _patient_number in patient_list_copy]

        patient_index_start = patient_number_per_fold_cumulative_list[-1]

        # with open(save_path + category_name + "_path_list_val_fold_" + str(fold_num) + ".txt", "w") as w:
        #     w.write(str(patient_path_val))
        # with open(save_path + category_name + "_path_list_tr_fold_" + str(fold_num) + ".txt", "w") as w:
        #     w.write(str(patient_path_tr))

        with open(save_path + category_name + "_path_list_tr_fold_" + str(fold_num) + ".csv", 'w', newline='') as csvfile:
            fieldnames = ['t-stage', 'patient_number', 'images_path_tr', 'mask_path_tr']
            writer = csv.DictWriter(csvfile, fieldnames = fieldnames)

            writer.writeheader()
            for row_num in range(len(image_path_list_tr)):
                writer.writerow({'t-stage': t_stage_label, 'patient_number': patient_list_copy[row_num],
                                 'images_path_tr': image_path_list_tr[row_num], 'mask_path_tr': mask_path_list_tr[row_num]})
                print('row_num: '+str(row_num))

        with open(save_path + category_name + "_path_list_val_fold_" + str(fold_num) + ".csv", 'w', newline='') as csvfile:
            fieldnames = ['t-stage', 'patient_number', 'images_path_val', 'mask_path_val']
            writer = csv.DictWriter(csvfile, fieldnames = fieldnames)

            writer.writeheader()
            for row_num in range(len(image_path_list_val)):
                writer.writerow({'t-stage': t_stage_label, 'patient_number': patient_list_val[row_num],
                                 'images_path_val': image_path_list_val[row_num], 'mask_path_val': mask_path_list_val[row_num]})

        del (image_path_list_val)
        del (mask_path_list_val)
        del (image_path_list_tr)
        del (mask_path_list_tr)
        del (patient_list_copy)
        del (patient_list_val)

# 실행 !!!
make_image_list( _path_until_patient_list, _image_folder_name, _label_folder_name, _save_path, _kfold, _category_name, _if_random_shuffle )