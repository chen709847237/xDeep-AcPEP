# coding:utf-8
import os
import numpy as np
import pandas as pd
import torch
import warnings
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, kendalltau
from data_layer import ACP_MT_dataset, ACP_dataset
from argparse import ArgumentParser
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


def st_model_prediction(model, val_dataloader):
    model.eval()
    val_epoch_batch_count = 0
    val_predict_list = []
    val_target_list = []
    for index, (val_x, val_y) in enumerate(val_dataloader):
        val_fea = torch.unsqueeze(val_x, dim=1).float()
        val_target = val_y.float()
        with torch.no_grad():
            val_predict = model(val_fea)
        val_epoch_batch_count += 1
        val_predict_list.extend(val_predict.data)
        val_target_list.extend(val_target.data)
    val_predict_list = np.array([i.item() for i in val_predict_list])
    val_target_list = np.array([i.item() for i in val_target_list])
    val_mse = mean_squared_error(val_target_list, val_predict_list)
    val_pcc = pearsonr(val_target_list, val_predict_list)[0]
    val_ktc = kendalltau(val_target_list, val_predict_list)[0]

    return val_mse, val_pcc, val_ktc

def mt_model_prediction(model, tissue_type, val_dataloader):
    model.eval()
    val_epoch_batch_count = 0
    val_predict_list = []
    val_target_list = []
    map_dict = {'breast': 1, 'cervix': 3, 'lung': 4, 'skin': 5, 'prostate': 6, 'colon': 11}
    for index, (val_x, val_y, _) in enumerate(val_dataloader):
        val_fea = torch.unsqueeze(val_x, dim=1).float()
        val_target = val_y.float()
        with torch.no_grad():
            val_predict = model(val_fea)
        val_epoch_batch_count += 1
        val_predict_list.extend(val_predict.data.numpy())
        val_target_list.extend(val_target.data.numpy())
    val_task_pre_list = np.array(val_predict_list)[:, map_dict[tissue_type]]
    val_task_tar_list = np.array(val_target_list)
    val_mse = mean_squared_error(val_task_tar_list, val_task_pre_list)
    val_pcc = pearsonr(val_task_tar_list, val_task_pre_list)[0]
    val_ktc = kendalltau(val_task_tar_list, val_task_pre_list)[0]
    return val_mse, val_pcc, val_ktc

if __name__ == '__main__':

    parser = ArgumentParser('Model 5-folds CV Result Reproduction')
    parser.add_argument('-mo', '--model_path', type=str, default='./model/', help='Model root Path')
    parser.add_argument('-da', '--data_path', type=str, default='./data/', help='Data root path')
    parser.add_argument('-o', '--output_path', type=str, default='./', help='Result save path')
    args = parser.parse_args().__dict__
    warnings.filterwarnings("ignore")

    model_root_path = args['model_path']
    data_root_path = args['data_path']
    output_root_path = args['output_path']
    tissue_type_list = ['breast', 'cervix', 'colon', 'lung', 'prostate', 'skin']
    tissue_result_list, model_name_result_list, pcc_result_list, mse_result_list, ktc_result_list = [], [], [], [], []
    print('Model 5-folds CV Result Reproduction Start !!!')
    for tissue_type in tissue_type_list:
        tissue_model_root_path = model_root_path+tissue_type+'/'
        tissue_data_root_path = data_root_path+tissue_type+'/'
        model_list = [f for f in os.listdir(tissue_model_root_path) if not f.startswith('.')]
        data_list = [f for f in os.listdir(tissue_data_root_path) if not f.startswith('.')]
        print(tissue_type)
        for model_name in model_list:
            single_model_root_path = tissue_model_root_path + model_name + '/'
            ad_ratio = model_name.split('_')[1]
            if model_name.split('_')[-2] == 'ST' or model_name.split('_')[-3] == 'ST': model_type = 'ST'
            else: model_type = 'MT'
            if model_name.split('_')[-2] == 'NT': pad_type = 'NT'
            else: pad_type = 'CT'
            print(model_name+'...')
            # find & load data
            data_file_name = tissue_type + '_' + model_type + '_' + ad_ratio
            if model_type == 'ST' and pad_type == 'NT': data_file_name = data_file_name+'_NT'
            elif model_type == 'ST' and pad_type == 'CT': data_file_name = data_file_name+'_CT'
            index = [i for i, x in enumerate(data_list) if x.find(data_file_name+'_data.npy') != -1]
            data_file_path = tissue_data_root_path + data_list[index[0]]
            val_data_file = np.load(data_file_path, allow_pickle=True).item()
            X = val_data_file['X']
            y = val_data_file['y']
            # find & load model
            val_mse_list, val_pcc_list, val_ktc_list = [], [], []
            for k_fold in range(5):
                fold_folder_path = single_model_root_path + str(k_fold) + '_fold/'
                val_idx_list = np.load(fold_folder_path + 'val_data_idx_list.npy')
                if model_type == 'ST':
                    val_X= X[val_idx_list]
                    val_y = y[val_idx_list]
                    val_data = ACP_dataset(val_X, val_y)
                else:
                    val_X = X[val_idx_list]
                    val_y = y[val_idx_list]
                    val_msk = val_data_file['mask'][val_idx_list]
                    val_data = ACP_MT_dataset(val_X, val_y, val_msk)
                dataloader = DataLoader(val_data, batch_size=val_X.shape[0])
                model = torch.load(fold_folder_path + 'best_mse_cnn_train_model.pth', map_location=torch.device('cpu'))
                if model_type == 'ST':
                    val_mse, val_pcc, val_ktc = st_model_prediction(model, dataloader)
                else:
                    val_mse, val_pcc, val_ktc = mt_model_prediction(model, tissue_type, dataloader)
                val_mse_list.append(val_mse)
                val_pcc_list.append(val_pcc)
                val_ktc_list.append(val_ktc)
            tissue_result_list.append(tissue_type)
            model_name_result_list.append(model_name)
            pcc_result_list.append(np.array(val_pcc_list).mean())
            mse_result_list.append(np.array(val_mse_list).mean())
            ktc_result_list.append(np.array(val_ktc_list).mean())
        print('---------------------------')
    result_df = pd.DataFrame({'tissue': tissue_result_list, 'model': model_name_result_list,
                              'mse': mse_result_list, 'pcc': pcc_result_list, 'ktc': ktc_result_list})
    result_df = result_df[['tissue', 'model', 'mse', 'pcc', 'ktc']]
    result_df.sort_values('model', inplace=True)
    wb = Workbook()
    ws = wb.active
    for r in dataframe_to_rows(result_df, index=False, header=True):
        ws.append(r)
    wb.save(output_root_path + 'model_cv_result.xlsx')



