"""Load the cropped single-beat ventricular MAP dictionary from disk.

Faithful copy of the original ``retrieve_ventMAP.py`` (data loading only; the
numerical logic is unchanged). The caller passes ``root_path`` -- in this repo
that is :func:`cardiac_map_diffusion.paths.ventmap_root` -- so no cluster path
is hard-coded here.
"""

import numpy as np
import pandas as pd
import datetime
import os

# %% Define function to acquire ventricular MAPs from raw data files

def retrieve_ventMAP(root_path):
    MAP_VT_voltages_path = os.path.join(root_path,
                                        r'Step_1_RemoveUnderOverShoots',
                           r'single_MAP_dict_BP_mean_3SD_cropped_20191111.npz')
    MAP_VT_voltages = np.load(MAP_VT_voltages_path, allow_pickle = True)
    MAP_VT_voltages = MAP_VT_voltages['arr_0']
    MAP_VT_voltages = MAP_VT_voltages.item()

    return MAP_VT_voltages

def getdate():
    date = datetime.datetime.now()
    day = str(date.day)
    month = str(date.month)
    year = str(date.year)
    if len(day) <= 1:
        day = '0' + day
    if len(month) <= 1:
        month = '0' + month
    date = year+month+day
    return date

def retrieve_ventMAP_and_metadata(root_path):
    date = getdate()
    vtvf3yr_split_path = os.path.join(root_path, r'Step_2_KCrossValidationSplits/VTVF 3 year/2019 1108 VTVF Split.xlsx')
    death3yr_split_path = os.path.join(root_path, r'Step_2_KCrossValidationSplits/Death 3 year/2019 1108 death 3 year split.xlsx')
    MAP_VT_voltages_path = os.path.join(root_path, r'Step_1_RemoveUnderOverShoots/single_MAP_dict_BP_mean_3SD_cropped_20191111.npz')


    for i in range(2):
        results_Train = pd.DataFrame()
        results_Val = pd.DataFrame()
        if i == 0:
            Split_PD = pd.ExcelFile(vtvf3yr_split_path)
            output_label = 'VT_VF in 3Y'
            writer = pd.ExcelWriter('DNN_Norm_'+date+'_3SD_singleBeat_'+'_'+output_label+'.xlsx', engine='xlsxwriter')

        elif i ==1:
            Split_PD = pd.ExcelFile(death3yr_split_path)
            output_label = 'Death in 3Y'
            writer = pd.ExcelWriter('DNN_Norm_'+date+'_3SD_singleBeat_'+'_'+output_label+'.xlsx', engine='xlsxwriter')


        MAP_VT_voltages = np.load(MAP_VT_voltages_path, allow_pickle = True)
        MAP_VT_voltages = MAP_VT_voltages['arr_0']
        MAP_VT_voltages = MAP_VT_voltages.item()

        all_auc = []
        for sheet in ['CV9']: #,'CV2','CV3','CV4','CV5','CV6','CV7','CV8','CV9','CV10']:
            sheet_name = sheet
            outcome_sheet = pd.read_excel(Split_PD, sheet_name = sheet)

            Train_dict = {}
            Val_dict = {}

            for index, row in outcome_sheet.iterrows():
                if index >= 0:
                    split_num = row['split_num']
                    if split_num == 0:
                        Train_dict[str(row['patient'])] = 1
                    elif split_num == 1:
                        Val_dict[str(row['patient'])] = 3

            X_Train = np.reshape(np.asarray([]),(0,370))
            Y_Train = np.reshape(np.asarray([]),(0,1))
            X_Val = np.reshape(np.asarray([]),(0,370))
            Y_Val = np.reshape(np.asarray([]),(0,1))

            count = 0
            for key in MAP_VT_voltages.keys():
    #            print(key)
                patient_voltages = MAP_VT_voltages[key]
                if key.endswith('_B'):
                    patient_row = outcome_sheet['patient']==key
                else:
                    patient_row = outcome_sheet['patient']==int(key)

                patient_row = outcome_sheet[patient_row]
                MAP_label = patient_row[output_label].values
                if str(key) in Train_dict.keys():
                    X_Train = np.append(X_Train, patient_voltages,axis = 0)
                    Y_Train = np.append(Y_Train, [MAP_label]*patient_voltages.shape[0])

                elif str(key) in Val_dict.keys():
                    X_Val = np.append(X_Val, patient_voltages, axis = 0)
                    Y_Val = np.append(Y_Val, [MAP_label]*patient_voltages.shape[0])
                count += 1

            X_mean = np.mean(X_Train)
            X_STD = np.std(X_Train)

            X_Train = (X_Train - X_mean)/X_STD
            X_Val = (X_Val - X_mean)/X_STD

            X_Train = np.expand_dims(X_Train,axis = -1)
            X_Val = np.expand_dims(X_Val,axis = -1)

    X = np.concatenate((X_Train, X_Val), axis=0)
    Y = np.concatenate((Y_Train, Y_Val), axis=0)

    X = np.squeeze(X)
    Y = np.squeeze(Y)
    patient_ID = list(MAP_VT_voltages.keys())

    return X, Y, patient_ID
