#!/usr/bin/env python

import pandas as pd



def ini():
    '''Initialisierung

    Args:


    Returns:

    '''
    pass


def save(picture_name, picture_ratio, output_folder):
    '''CSV File erstellen

    Args:


    Returns:

    '''
    raw_data = {'Picture': picture_name, 'Ratio': picture_ratio}
    df = pd.DataFrame(raw_data, columns = ['Picture', 'Ratio'])
    df.to_csv(output_folder)
