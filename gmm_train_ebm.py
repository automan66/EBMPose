import os
import glob
from re import split
import torch
import random
import logging
import datetime
import numpy as np
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
from common.arguments import opts as parse_args
from common.utils import *
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset
import time
from model_nce import *
from model.model_IGANet import *

parser = parse_args()
parser.add_argument(
    "--gmm_stds",
    type=float,
    nargs="+",
    default=[0.05, 0.1, 0.2],
    help="Standard deviations for Gaussian noise."
)
args = parser.parse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
exec('from model.' + 'model_IGANet' + ' import Model as IGANet')

def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)

def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)

def step(split, args, actions, dataLoader, model, optimizer=None, epoch=None):

    loss_all = {'loss': AccumLoss()}

    action_error_sum = define_error_list(actions)
    
    model_3d = model['IGANet']
    if split == 'train':
        model_3d.train()
    else:
        model_3d.eval()

    for _, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])

        if split =='train':
            output_3D, xfeat = model_3d(input_2D) 
        else:
            input_2D, output_3D, xfeat = input_augmentation(input_2D, model_3d)

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0

        if split == 'train':
            loss_p1 = mpjpe_cal(output_3D, out_target.clone())
            N = input_2D.size(0)
            loss_all['loss'].update(loss_p1.detach().cpu().numpy() * N, N)

            loss = loss_p1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif split == 'test':
            output_3D = output_3D[:, args.pad].unsqueeze(1) 
            output_3D[:, :, 0, :] = 0
            test_p1 = mpjpe_cal(output_3D, gt_3D)
            action_error_sum = test_calculation(output_3D, out_target, action, action_error_sum, args.dataset, subject)

    if split == 'train':
        return loss_all['loss'].avg

    elif split == 'test':
        mpjpe_p1, p2 = print_error(args.dataset, action_error_sum, args.train)
        return mpjpe_p1, p2

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == '__main__':
    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logtime = time.strftime('%y%m%d_%H%M_%S')
    args.create_time = logtime
     
    if args.create_file:
        # create backup folder
        if args.debug:
            args.checkpoint = './debug/' + logtime
        else:
            args.checkpoint = './checkpoint/' + logtime
    
        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)

        # backup files
        import shutil
        file_name = os.path.basename(__file__)
        shutil.copyfile(src=file_name, dst = os.path.join( args.checkpoint, args.create_time + "_" + file_name))
        shutil.copyfile(src="common/arguments.py", dst = os.path.join(args.checkpoint, args.create_time + "_arguments.py"))
        shutil.copyfile(src="model/model_IGANet.py", dst = os.path.join(args.checkpoint, args.create_time + "_model_IGANet.py"))
        shutil.copyfile(src="common/utils.py", dst = os.path.join(args.checkpoint, args.create_time + "_utils.py"))
        shutil.copyfile(src="run.sh", dst = os.path.join(args.checkpoint, args.filename+"_run.sh"))

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(args.checkpoint, 'train.log'), level=logging.INFO)
        
        arguments = dict((name, getattr(args, name)) for name in dir(args)
                if not name.startswith('_'))
        file_name = os.path.join(args.checkpoint, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(arguments.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')

    root_path = 'dataset/'#args.root_path
    dataset_path = root_path + 'data_3d_' + args.dataset + '.npz'

    dataset = Human36mDataset(dataset_path, args)
    actions = define_actions(args.actions)

    test_data = Fusion(opt=args, train=False, dataset=dataset, root_path =root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                                      shuffle=True, num_workers=int(args.workers), pin_memory=True)

    model = {}
    model['IGANet'] = IGANet(args).cuda()

    # if args.reload:
    model_dict = model['IGANet'].state_dict()
        # model_path = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))[0]
    model_path = glob.glob(os.path.join('./pre_trained_model', '*.pth'))[0]
        # model_path = "./pre_trained_model/IGANet_8_4834.pth"
    print(model_path)
    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model['IGANet'].load_state_dict(model_dict)
    print("Load IGANet Successfully!")

    model_3d = model['IGANet']
        # print(model_3d)
    model_3d.eval()

    split = 'test'
    num_dataLoader = len(test_dataloader.dataset)
        
    starttime = datetime.datetime.now()
    best_epoch = 0
    
    input_stds = args.gmm_stds
    stds = torch.zeros((1, 3))
    stds[0, 0] = input_stds[0]
    stds[0, 1] = input_stds[1]
    stds[0, 2] = input_stds[2]
    num_neg = 100

    batch_size = 256
    total_epoch = 30
    ebm_lr = 0.001
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    pth_dir = 'ckp-lr{}-res'.format(ebm_lr)
    if not os.path.exists(pth_dir):
        os.makedirs(pth_dir)

    netE = EBMPrior().apply(weights_init_xavier).cuda()
    optE = torch.optim.Adam(netE.parameters(), lr=ebm_lr, betas=(0.5, 0.999))
    lr_scheduleE = torch.optim.lr_scheduler.ExponentialLR(optE, 0.99)

    netE.train()

    for epoch in range(1, total_epoch+1):
        pbar = tqdm(test_dataloader)
        for i, data in enumerate(tqdm(test_dataloader, 0)):
            batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
            
            [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])

            if split =='train':
                output_3D, _ = model_3d(input_2D) 
            else:
                input_2D, output_3D, xfeat = input_augmentation(input_2D, model_3d) # xfeat: [128, 17, 512]


            num_x = input_2D.size(0)
            # x = xfeat.clone().view(num_x, 17, 512)
            # y = out_target.view(num_x, 17, 3)
            x = xfeat.clone().view(num_x, 17*512)

            out_target = gt_3D.clone()  # gt_3D: [128, 1, 17, 3]
            # out_target[:, :, 0] = 0
            y = out_target.view(num_x, 17*3)
            
            y_pred = output_3D.clone()
            # y_pred[:, :, 0] = 0
            y_pred = y_pred.view(num_x, 17*3)

            # print(y)    
            optE.zero_grad()
            pos = -netE(x, y).squeeze()
            E_pos = netE(x, y)[0,0].item()
            E_pred = netE(x, y_pred)[0,0].item()
            # print(pos)
            x_ = x.unsqueeze(1).repeat(1, num_neg, 1) # [b,1,17*2]->[b,n,17*2]
            y_ = y.unsqueeze(1).repeat(1, num_neg, 1)

            y_samples_zero, q_y_samples, q_ys = sample_gmm_centered(stds, num_samples=num_neg)
            y_samples_zero = y_samples_zero.cuda() # (shape: (num_samples, 1))

            q_y_samples = q_y_samples.cuda() # (shape: (num_samples))
            y_samples = y_ + y_samples_zero.unsqueeze(0).repeat(num_x, 1, 17*3) # (shape: (batch_size, num_samples, joints))
                        
            q_y_samples = q_y_samples.unsqueeze(0)*torch.ones(y_samples.size(0), y_samples.size(1)).cuda() # (shape: (batch_size, num_samples))
            q_ys = q_ys[0]*torch.ones(x_.size(0)).cuda()

            x_ = x_.view(-1, 17*512)
            y_samples = y_samples.view(-1, 17*3)

            neg = -netE(x_, y_samples)
            E_neg = netE(x_, y_samples)[0,0].item()

            neg = neg.view(num_x, num_neg) # (shape: (batch_size, num_samples))
            loss_e = -torch.mean(pos-torch.log(q_ys) - torch.log(torch.exp(pos-torch.log(q_ys)) + torch.sum(torch.exp(neg-torch.log(q_y_samples)), dim=1)))  #-torch.log(pos/(pos+neg)).mean()
            loss_r = (pos**2).mean() + (neg**2).mean()
            loss = loss_e + 0.1*loss_r
            loss_e.backward()
            optE.step()
            
            pbar.set_description('Epoch: {:4d}, loss_e: {:.6f}, E_pos:{:.3f}, E_neg:{:.3f}, E_pred-E_pos:{:.3f}'.format(epoch, loss_e.item(), E_pos, E_neg, E_pred-E_pos))
            
        lr_scheduleE.step()
        torch.save(netE.state_dict(), pth_dir+'/netE_{:4d}_{:.4f}'.format(epoch, loss_e.item()))
