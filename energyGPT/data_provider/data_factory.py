from data_provider.data_loader import  Dataset_ASU_hour
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import os
from sklearn.preprocessing import StandardScaler
import pickle 
import pandas as pd
data_dict = {
    'ASU':Dataset_ASU_hour
}


def load_all_train_data(root_path, data_path):
    df_train = pd.read_csv(os.path.join(root_path, data_path))
    return df_train


def data_provider(args, flag, is_shuffle=None):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        if is_shuffle is None:
            shuffle_flag = False
        else:
            shuffle_flag = is_shuffle
        drop_last = True
        batch_size = 1  
        freq = args.freq
    else:
        if is_shuffle is None:
            shuffle_flag = True
        else:
            shuffle_flag = is_shuffle
        drop_last = True
        batch_size = args.batch_size  
        freq = args.freq
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=0,
            percent=percent,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
        )
        batch_size = args.batch_size
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
