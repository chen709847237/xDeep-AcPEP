# coding=utf-8
import os
import time
import warnings
import torch
import numpy as np
import pandas as pd
import math
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist
from data_layer import *
from feature_generator import *


def prediction(tissue_type, ad_param, temp_fea_file, model_path, result_root_folder, data_file_name):
        begin = time.time()
        warnings.filterwarnings("ignore")
        torch.manual_seed(0)
        map_dict = {'breast': 1, 'cervix': 3, 'lung': 4, 'skin': 5, 'prostate': 6, 'colon': 11}
        pre_fea, pre_id = matrix_generator(temp_fea_file, pad_len=38)
        pre_sim_score = []
        for i in range(pre_fea.shape[0]):
            pre_sim_score.extend(pdist(np.vstack([pre_fea[i].flatten(), ad_param['ad_centroid']]))/1000)
        pre_ad_filter = (pre_sim_score <= ad_param['ad_mean'] + ad_param['ad_ratio'] * ad_param['ad_std'])
        print('Total number of submited samples')
        print('in AD: ', pre_ad_filter.sum())
        print('out AD: ', (pre_ad_filter == 0).sum())
        if pre_ad_filter.sum() != 0:
            pre_fea, pre_id = pre_fea[pre_ad_filter], pre_id[pre_ad_filter]
            pre_dataloader = DataLoader(ACP_dataset(pre_fea, pre_id), batch_size=pre_fea.shape[0], shuffle=False)
            result_df = pd.DataFrame()
            q_pre_list, q_id_list = [], []
            print('----------------------------------------------------------------')
            print('PREDICTION START!')
            model_name_list = [f for f in os.listdir(model_path) if not f.startswith('.')]
            index = [i for i, x in enumerate(model_name_list) if x.find(tissue_type) != -1]
            target_model_path = model_path + np.array(model_name_list)[index][0]
            model = torch.load(target_model_path)
            model.eval()
            for index, (q_x, q_id) in enumerate(pre_dataloader):
                q_fea = torch.unsqueeze(q_x, dim=1).float()
                q_id = list(q_id)
                with torch.no_grad():
                    q_predict = model(q_fea)
                q_pre_list.extend(q_predict.data.numpy())
                q_id_list.extend(q_id)
            result_df[tissue_type+'_idx'] = np.array(q_id_list)
            #result_df[tissue_type+'_pre'] = np.array(q_pre_list)[:, map_dict[tissue_type]]
            result_ori = np.array([math.pow(10, -i) * 1000 * 1000 for i in np.array(q_pre_list)[:, map_dict[tissue_type]]])
            result_df[tissue_type + '_pre'] = result_ori
        
        result_df.to_csv(result_root_folder + 'result_'+data_file_name + '.csv', index=False)
            print('PREDICTION OVER!')
            print('result save at ', result_root_folder)
            print('Total time: %.2f' % (time.time() - begin) + 's')
            print('----------------------------------------------------------------')
        else:
            print('Zero samples in the AD range!')
            print('Program End !')
        os.remove(temp_fea_file)

if __name__ == '__main__':

    parser = ArgumentParser('xDeep-AcPEP Model Prediction')
    parser.add_argument('-t', '--tissue', type=str, default='breast', help='Define model type')
    parser.add_argument('-m', '--model_path', type=str, default='./model/', help='Data root path')
    parser.add_argument('-d', '--data_path', type=str, help='Input peptide fasta file path')
    parser.add_argument('-o', '--output_path', type=str, default='./', help='Result save path')
    args = parser.parse_args().__dict__
    warnings.filterwarnings("ignore")
    tissue_type = args['tissue']
    model_path = args['model_path']
    result_root_folder = args['output_path']
    fasta_data_file = args['data_path']
    temp_fea_file = result_root_folder + '.temp_fea_' + fasta_data_file.split('/')[-1]
    ad_define_file = model_path + 'ad_define.npy'
    ad_data = np.load(ad_define_file, allow_pickle=True).item()
    temp_fea_file = feature_generator(fasta_data_file, temp_fea_file, verbose=50)
    prediction(tissue_type, ad_data, temp_fea_file, model_path, result_root_folder, fasta_data_file.split('/')[-1])
    print('prediction all over!!!')




