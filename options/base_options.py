import argparse

def get_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-file", type=str, default='',
            help="training data file")
    parser.add_argument("--type", type=str, default='train',
            help="{train|test}")
    parser.add_argument("--exp-dir", type=str, default="",
            help="directory to dump experiments")
    parser.add_argument("--resume", action="store_true", dest="resume",
        help="load from exp_dir if True")
    parser.add_argument("--optim", type=str, default="sgd",
            help="training optimizer", choices=["sgd", "adam"])
    parser.add_argument('-b', '--batch-size', default=256, type=int,
        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-decay', default=40, type=int, metavar='LRDECAY',
        help='Divide the learning rate by 10 every lr_decay epochs')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float,
        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument("--n_epochs", type=int, default=100,
            help="number of maximum training epochs")
    parser.add_argument("--n_print_steps", type=int, default=100,
            help="number of steps to print statistics")
    
    return parser.parse_args()