import sys
from pathlib import Path
#Add path to parent folder for imports
sys.path.append(str(Path.cwd().parent))

import argparse
import csv
import torch 
import numpy as np
torch.manual_seed(0)
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm 
from timebudget import timebudget
from utils.data import Ai224_RG_Dataset, RandomSampler
from utils.transforms import My_RandomFlip,My_RandomContrast,My_RandomGamma,My_Normalization
from models.unet import Ai224_RG_UNet

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size",          default=12,                                     type=int)
parser.add_argument("--n_epochs",            default=50,                                     type=int)
parser.add_argument("--n_batches_per_epoch", default=10,                                     type=int)
parser.add_argument("--val_num_samples",     default=50,                                     type=int)
parser.add_argument("--im_path",             default='../../../dat/raw/Unet_tiles_082020/',  type=str)
parser.add_argument("--lbl_path",            default='../../../dat/proc/Unet_tiles_082020/', type=str)
parser.add_argument("--result_path",         default='../../../dat/Ai224_RG_models/',        type=str)
parser.add_argument("--expt_name",           default='TEMP',                                 type=str)


def main(batch_size=12, n_epochs=50, n_batches_per_epoch=10, val_num_samples=50,
         im_path='../../../dat/raw/Unet_tiles_082020/',
         lbl_path='../../../dat/proc/Unet_tiles_082020/',
         result_path='../../../dat/Ai224_RG_models/',
         expt_name='TEMP'):

    num_samples = n_batches_per_epoch*batch_size
    patch_size = 260
    train_pad = int(patch_size/2)

    assert Path.is_dir(Path(im_path)), f'{im_path} not found'
    assert Path.is_dir(Path(lbl_path)), f'{lbl_path} not found'
    expt_path = Path(result_path + expt_name)
    expt_path.mkdir(parents=True, exist_ok=True)

    log_file = expt_path / 'log.csv'
    def ckpt_file(epoch): return str(expt_path / f'{epoch}_ckpt.pt')
    
    #Data =============================
    train_np_transform = My_RandomFlip(p=0.5)
    train_torch_transforms = Compose([My_RandomContrast(p=1.0, contrast_factor_list=[0.75,0.875,1.0,1.125,1.25]),
                                My_RandomGamma(p=1.0, gamma_list=[0.8,0.9,1.0,1.1,1.2]),
                                My_Normalization(scale=255.)])
                                
    val_torch_transforms = Compose([My_Normalization(scale=255.)])


    train_dataset = Ai224_RG_Dataset(pad=train_pad,
                                     patch_size=patch_size,
                                     subset='train',
                                     im_path=im_path,
                                     lbl_path=lbl_path,
                                     np_transform=train_np_transform,
                                     torch_transforms=train_torch_transforms)

    train_sampler = RandomSampler(n_tiles=train_dataset.n_tiles,
                                  min_x=0, min_y=0,
                                  max_x=train_dataset.tile_shape_orig[0],
                                  max_y=train_dataset.tile_shape_orig[1],
                                  num_samples=num_samples)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                  sampler=train_sampler, drop_last=True, pin_memory=True)

    #Validation data: only a single, persistent set of validation data will be used.
    val_dataset = Ai224_RG_Dataset(pad=0,
                                   patch_size=patch_size,
                                   subset='val',
                                   im_path=im_path,
                                   lbl_path=lbl_path,
                                   np_transform=None,
                                   torch_transforms=val_torch_transforms)

    val_sampler = RandomSampler(n_tiles=val_dataset.n_tiles,
                                min_x=0,
                                min_y=0,
                                max_x=val_dataset.tile_shape_orig[0] - patch_size,
                                max_y=val_dataset.tile_shape_orig[1] - patch_size,
                                num_samples=val_num_samples)

    val_dataloader = DataLoader(val_dataset, batch_size=val_num_samples, shuffle=False,
                                sampler=val_sampler, drop_last=False, pin_memory=True)

    val_datagen = iter(val_dataloader)
    val_batch = next(val_datagen)

    #Model ============================
    model = Ai224_RG_UNet()
    optimizer = torch.optim.Adam(model.parameters())
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.4,0.4,0.4]))

    #Helpers ==========================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def tensor(x): return torch.tensor(x).to(dtype=torch.float32).to(device)
    def tensor_(x): return torch.as_tensor(x).to(dtype=torch.float32).to(device)
    def tonumpy(x): return x.cpu().detach().numpy()

    model.to(device)
    ce.to(device)

    #Training loop ====================
    best_loss = np.inf
    monitor_loss = []
    for epoch in range(n_epochs):
        loss_list = []
        train_datagen = iter(train_dataloader)
        for step in range(len(train_dataloader)):
            batch = next(train_datagen)
            
            #zero + forward + backward + udpate
            optimizer.zero_grad()
            xg, xr, _, _ = model(tensor_(batch['im']))
            target = Ai224_RG_UNet.crop_tensor(tensor_(batch['lbl']), xg)
            loss = ce(xg, target[:, 0, ...].type(torch.long)) + \
                   ce(xr, target[:, 1, ...].type(torch.long))
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())
            if (step+1) % len(train_dataloader) == 0: #For last step in every epoch
                #Report average training loss
                train_loss = np.mean(loss_list)
                
                #Validation: train mode -> eval mode + no_grad + eval mode -> train mode
                model.eval()
                with torch.no_grad():
                    val_xg,val_xr,_,_ = model(tensor_(val_batch['im']))
                    val_target = Ai224_RG_UNet.crop_tensor(tensor_(val_batch['lbl']), val_xg)
                    val_loss = ce(val_xg, val_target[:,0,...].type(torch.long)) + \
                            ce(val_xr, val_target[:,1,...].type(torch.long))
                    
                val_loss = tonumpy(loss)
                print('\repoch {:04d} validation loss: {:0.6f}'.format(epoch,val_loss),end='')
                model.train()
                
                print(f'\repoch {epoch:04d} training loss: {train_loss:0.6f}, val. loss: {val_loss:0.6f}',end='')
                #Logging ==============
                with open(log_file, "a") as f:
                    writer = csv.writer(f, delimiter=',')
                    if epoch == 0:
                        writer.writerow(['epoch','train_ce','val_ce'])
                    writer.writerow([epoch+1,train_loss,val_loss])
                    
                monitor_loss.append(val_loss)
                
                #Checkpoint ===========
                if (monitor_loss[-1] < best_loss) and (epoch>500):
                    best_loss = monitor_loss[-1]
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': best_loss,}, ckpt_file(epoch))
    print('\nTraining completed.')

    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
