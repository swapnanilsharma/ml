# create_folds.py
import config
import pandas as pd
from sklearn import model_selection


def create_folds(no_of_fold, target):
    # read the training data
    df = pd.read_csv(config.RAW_FILE)

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch targets
    y = df[target].values

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=no_of_fold)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = fold

    # save the new csv with kfold column
    df.to_csv(config.TRAINING_FILE, index=False)


if __name__ == "__main__":
    create_folds(no_of_fold=5, target="label")