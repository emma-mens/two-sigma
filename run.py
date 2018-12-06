import os
import pickle
import sys
import time
import torch


import dataloaders
import models
import options


if __name__ == "__main__":
    
    args = options.get_options()
    
    train_loader = torch.utils.data.DataLoader(FinanceDataset(args.data_file, data_type='train'),
                        batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    val_loader = train_loader = torch.utils.data.DataLoader(FinanceDataset(args.data_file, data_type='val'),
                    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    
    model = models.FinanceModel()
    
    
    if not bool(args.exp_dir):
        print("exp_dir not specified, automatically creating one...")
        args.exp_dir = "exp/Data-%s/AudioModel-%s_ImageModel-%s_Optim-%s_LR-%s_Epochs-%s" % (
            os.path.basename(args.data_train), args.audio_model, args.image_model, args.optim,
            args.lr, args.n_epochs)
        
    if not args.resume:
        print("\nexp_dir: %s" % args.exp_dir)
        os.makedirs("%s/models" % args.exp_dir)
        with open("%s/args.pkl" % args.exp_dir, "wb") as f:
            pickle.dump(args, f)

    train(model, train_loader, val_loader, args)