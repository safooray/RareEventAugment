import pandas as pd
import numpy as np


DATA_PATH = 'data/processminer-rare-event-detection-data-augmentation.xlsx'
SHEET_NAME = 'data-(a)-raw-data'
LABEL_NAME = 'y'


def prepare_data(data_path=DATA_PATH, sheet_name=SHEET_NAME, label_name=LABEL_NAME):
    data_file = pd.ExcelFile(data_path)
    data = pd.read_excel(data_file, sheet_name)

    data = data.drop(['time', 'x28', 'x61'], axis=1)
    data = label_shift(data, -2, label_name)

    input_X = data.loc[:, data.columns != label_name].values  # converts the df to a numpy array
    input_y = data[label_name].values
    return input_X, input_y


def label_shift(df, shift_by, labelcol='y'):
    '''
    This function will shift the binary labels in a dataframe.
    The curve shift will be with respect to the 1s. 
    For example, if shift is -2, the following process
    will happen: if row n is labeled as 1, then
    - Make row (n+shift_by):(n+shift_by-1) = 1.
    - Remove row n.
    i.e. the labels will be shifted up to 2 rows up.
    
    Inputs:
    df       A pandas dataframe with a binary labeled column. 
             This labeled column should be named as 'y'.
    shift_by An integer denoting the number of rows to shift.
    
    Output
    df       A dataframe with the binary labels shifted by shift.
    '''
    sign = lambda x: (1, -1)[x < 0]
    vector = df[labelcol].copy()
    for s in range(abs(shift_by)):
        tmp = vector.shift(sign(shift_by))
        tmp = tmp.fillna(0)
        vector += tmp
    # Add vector to the df
    df.insert(loc=0, column=labelcol+'tmp', value=vector)
    # Remove the rows with labelcol == 1.
    df = df.drop(df[df[labelcol] == 1].index)
    # Drop labelcol and rename the tmp col as labelcol
    df = df.drop(labelcol, axis=1)
    df = df.rename(columns={labelcol+'tmp': labelcol})
    # Make the labelcol binary
    df.loc[df[labelcol] > 0, labelcol] = 1

    return df
