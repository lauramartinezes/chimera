import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from utils import *


def get_test_plates(df):
    test_plates = train_test_split(df.platename.unique(
    ), random_state=2022, test_size=0.2, shuffle=True)[1].tolist()
    testseries = pd.Series(test_plates)
    return testseries.tolist()


def get_test_plates_final_weeks(df, start_interval_final_weeks=None, end_interval_final_weeks=-1):
    weeks = df.date.unique().tolist()
    weeks.sort()

    if start_interval_final_weeks is None:
        interval_final_weeks = [weeks[end_interval_final_weeks]]
    else:
        interval_final_weeks = weeks[start_interval_final_weeks:end_interval_final_weeks]

    df_final_week = df[df['date'].isin(interval_final_weeks)]
    test_plates = df_final_week.platename.unique().tolist()
    testseries = pd.Series(test_plates)
    return testseries.tolist()


def split_dataset_by_plate(df):
    test_plates = get_test_plates(df)

    df_trainval = df[~df.platename.isin(test_plates)]
    df_test = df[df.platename.isin(test_plates)]

    topclasses = df['label'].value_counts().head(12).index.tolist()

    df = df[df['label'].isin(topclasses)]
    df_trainval = df_trainval[df_trainval['label'].isin(topclasses)]
    df_test = df_test[df_test['label'].isin(topclasses)]

    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(
                                                          df['label'].tolist()),
                                                      y=df['label'].tolist())

    class_weights = {np.unique(df['label'])[i]: class_weights[i]
                     for i in range(len(class_weights))}
    df['weights'] = df['label'].map(class_weights)

    df_train, df_val = train_test_split(
        df_trainval, test_size=0.18, random_state=42, shuffle=True)

    return df_train, df_val, df_test


def split_dataset_simplified(df):
    # this function is a simplification of split_dataset() from the stickybugs_ai_repo where we do not
    # divide the dataset into train-validation-test but only train-validation, since our test set will be
    # composed of only images from the last week
    validation_plates = get_test_plates(df)

    df_train = df[~df.platename.isin(validation_plates)]
    df_val = df[df.platename.isin(validation_plates)]

    topclasses = df['label'].value_counts().head(12).index.tolist()

    df = df[df['label'].isin(topclasses)]
    df_train = df_train[df_train['label'].isin(topclasses)]
    df_val = df_val[df_val['label'].isin(topclasses)]

    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(
                                                          df['label'].tolist()),
                                                      y=df['label'].tolist())

    class_weights = {np.unique(df['label'])[i]: class_weights[i]
                     for i in range(len(class_weights))}
    df['weights'] = df['label'].map(class_weights)

    return df_train, df_val
