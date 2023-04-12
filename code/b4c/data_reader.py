import cPickle  # for working with pickled data
import numpy # otherwise opening the pickle file fails
import numpy as np  # for working with arrays
import pandas as pd  # for working with dataframes
from glob import glob  # for searching files by pattern


def b4c_parse_train_data(file_path):
    """Parses the training data pickle file and returns a Pandas dataframe.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        pd.DataFrame: A dataframe containing the parsed data.
    """
    # Initialize empty dataframe with column names
    col_names = ["id", "t", "label",
                 "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12",  # face-tracking features
                 "x13", "x14", "x15", "x16", ]  # road features
    df = pd.DataFrame(columns=col_names)

    # Load data from pickle file
    train_data = cPickle.load(open(file_path))
    Y_tr = train_data['labels']
    X_tr = train_data['features']

    # Iterate over data and add to dataframe
    for i in range(len(X_tr)):
        train_samples = X_tr[:, i:i+1, :]

        for j, k in enumerate(train_samples):
            # add an index number, timestep, and label
            label = Y_tr[j][i]
            id_t = np.array([int(i), int(j), int(label)]) # id, time, label
            row = np.concatenate((id_t, k[0]))
            # add as row to dataframe
            df = df.append(pd.DataFrame(row.reshape(1,-1), columns=col_names), ignore_index=True)

    return df


def main():
    file_path = './original_data/train_data_*.pik'
    filepath_list = glob(file_path)
    for i, filepath in enumerate(filepath_list):
        # Example run of the python 2.7 pickle parse script
        df = b4c_parse_train_data(filepath)
        df.to_pickle('b4c_train_data_{}.p'.format(str(i)))
    
if __name__ == "__main__":
    main()
