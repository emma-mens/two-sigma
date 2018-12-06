import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
from .utils import *


def train(audio_model, image_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    
    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)
            
    # create/load exp
    if args.resume:
        progress_pkl = "%s/progress.pkl" % exp_dir
        progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
        print("\nResume training from:")
        print("  epoch = %s" % epoch)
        print("  global_step = %s" % global_step)
        print("  best_epoch = %s" % best_epoch)
        print("  best_acc = %.4f" % best_acc)
        
        
    if not isinstance(model, torch.nn.DataParallel):
        model = nn.DataParallel(model)
        
    if epoch != 0:
        model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)
        
    model = model.to(device)
    
    # Set up the optimizer
    trainables = [p for p in model.parameters() if p.requires_grad]
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(trainables, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay,
                                betas=(0.95, 0.999))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)
    
    criterion = nn.MSELoss()
        
    if epoch != 0:
        optimizer.load_state_dict(torch.load("%s/models/optim_state.%d.pth" % (exp_dir, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)

    epoch += 1
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    model.train()
    while True:
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        end_time = time.time()
        model.train()
        for i, (input, label) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end_time)

            input = input.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            output = model(input)

            loss = criterion(output, loss)

            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)

            if global_step % args.n_print_steps == 0 and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        val_loss = validate(model, test_loader, criterion, args)
        
        torch.save(audio_model.state_dict(),
                "%s/models/model.%d.pth" % (exp_dir, epoch))
        torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))
        
        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            shutil.copyfile("%s/models/model.%d.pth" % (exp_dir, epoch), 
                "%s/models/best_model.pth" % (exp_dir))
        _save_progress()
        epoch += 1
        
    def validate(model, val_loader, criterion, args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_time = AverageMeter()
        if not isinstance(model, torch.nn.DataParallel):
            model = nn.DataParallel(model)

        model = model.to(device)
        # switch to evaluate mode
        model.eval()

        end = time.time()
        N_examples = val_loader.dataset.__len__()
        I_embeddings = [] 
        A_embeddings = [] 
        frame_counts = []
        total_loss = 0
        with torch.no_grad():
            for i, (input, label) in enumerate(val_loader):
                input = input.to(device)

                output = model(input)

                output = output.to('cpu').detach()

                loss = criterion(output, label)
                
                batch_time.update(time.time() - end)
                end = time.time()
                total_loss += loss.item()
        loss = total_loss/len(val_loader)
                
        print('validation loss'.format(loss), flush=True)

        return loss