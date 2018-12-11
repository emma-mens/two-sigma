import os
import pickle
import sys
import time
import torch
import torch.nn as nn

import models
import options
from dataloader import FinanceDataset
from torch.utils.data import DataLoader
from scripts import train, validate

if __name__ == "__main__":
    
    args = options.get_options()
    
    output_dim = 1
    
    if args.type == 'test':
        test_loader = DataLoader(FinanceDataset(args.data_file, args.augmented_data,
                                             data_type='test', single_df=args.single_df,
                                             returns_direction=args.direction,
                                             max_interval=str(args.max_interval) + ' days'),
                              batch_size=args.batch_size, shuffle=False, 
                              num_workers=1, pin_memory=True, drop_last=False)
        n_features = test_loader.dataset[0][0].shape[1]
        model = models.FinanceModel(input_dim=n_features, output_dim=output_dim,  binary=args.direction)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load("%s/models/best_model.pth" % (args.exp_dir)))

        validate(model, test_loader, args)
        
    else:
        train_loader = DataLoader(FinanceDataset(args.data_file,args.augmented_data,
                                                 data_type='train', single_df=args.single_df,
                                                 returns_direction=args.direction,
                                                 max_interval=str(args.max_interval) + ' days'),
                                  batch_size=args.batch_size, shuffle=True, 
                                  num_workers=1, pin_memory=True, drop_last=True)

        val_loader = DataLoader(FinanceDataset(args.data_file, args.augmented_data,
                                               data_type='val', single_df=args.single_df,
                                               returns_direction=args.direction,
                                               max_interval=str(args.max_interval) + ' days'),
                                batch_size=args.batch_size, shuffle=False, 
                                num_workers=1, pin_memory=True, drop_last=True)

        resume = args.resume

        if args.resume:
            assert(bool(args.exp_dir))
            if args.reuse_old:
                with open("%s/args.pkl" % args.exp_dir, "rb") as f:
                    args = pickle.load(f)
        args.resume = resume

        print(args)

        # get number of features of input
        n_features = train_loader.dataset[0][0].shape[1]
        model = models.FinanceModel(input_dim=n_features, output_dim=output_dim,
                                    dropout_p=args.dropout, binary=args.direction)

        if not bool(args.exp_dir):
            print("exp_dir not specified, automatically creating one...")
            args.exp_dir = "exp/Data-%s/Optim-%s_LR-%s_Epochs-%s" % (
                os.path.basename(args.data_file), args.optim, args.lr, args.n_epochs)

        if not args.resume:
            print("\nexp_dir: %s" % args.exp_dir)
            os.makedirs("%s/models" % args.exp_dir)
            with open("%s/args.pkl" % args.exp_dir, "wb") as f:
                pickle.dump(args, f)

        train(model, train_loader, val_loader, args)