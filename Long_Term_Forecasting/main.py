import argparse
import time
import datetime
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import random
import numpy as np

from  einops import rearrange
import torch.distributed as dist
import torch.multiprocessing as mp


import lightning as L
from lightning.fabric.strategies import DDPStrategy
import glob
from data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from metrics import metric
from tqdm import tqdm
       
from transformers import AutoModelForCausalLM



data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}



def data_provider(data, 
                  root_path,
                  data_path,
                  batch_size = 128,
                  seq_len = 96,
                  pred_len = 96,
                  flag= 'test',
                  dataset = None,
                  num_workers = 8,
                  seasonal_patterns = None,
                  target = 'OT',
                  features = 'M',
                 ):
    

    Data = data_dict[data]
    shuffle_flag = False
    drop_last = False
    batch_size = batch_size 
    data_set = Data(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[seq_len, 0, pred_len],
        features=features,
        target=target,
        timeenc=1,
        freq='h',
        seasonal_patterns=seasonal_patterns
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory = True)
    return data_set, data_loader

datasets_configs = {
'ETTh1': {'data': 'ETTh1',
            'root_path': './Long_Term_Forecasting/dataset/ETT-small/',
            'data_path': 'ETTh1.csv',
            'batch_size':32,
            'best_other_mse':'0.390',
            'best_other_mae':'0.406',
         },
'ETTh2': {'data': 'ETTh2',
            'root_path': './Long_Term_Forecasting/dataset/ETT-small/',
            'data_path': 'ETTh2.csv',
            'batch_size':32,
            'best_other_mse':'0.330',
            'best_other_mae':'0.375',
         },
'ETTm1': {'data': 'ETTm1',
            'root_path': './Long_Term_Forecasting/dataset/ETT-small/',
            'data_path': 'ETTm1.csv',
            'batch_size':32,
            'best_other_mse':'0.351',
            'best_other_mae':'0.372',
         },
'ETTm2': {'data': 'ETTm2',
            'root_path': './Long_Term_Forecasting/dataset/ETT-small/',
            'data_path': 'ETTm2.csv',
            'batch_size':32,
            'best_other_mse':'0.255',
            'best_other_mae':'0.315',
         },
'Weather': {'data': 'custom',
            'root_path': './Long_Term_Forecasting/dataset/weather/',
            'data_path': 'weather.csv',
            'batch_size':32,
            'best_other_mse':'0.226',
            'best_other_mae':'0.261',
         }, 
'Electricity': {'data': 'custom',
            'root_path': './dataset/',
            'data_path': 'electricity/electricity.csv',
            'batch_size':1,
            'best_other_mse':'0.159',
            'best_other_mae':'0.253',
         },
'Traffic': {'data': 'custom',
            'root_path': './dataset/',
            'data_path': 'traffic.csv',
            'batch_size':32,
            'best_other_mse':'0.391',
            'best_other_mae':'0.264',
         },
    
'GlobalTemp': {'data': 'Global_Temp',
            'root_path': './dataset/',
            'data_path': 'solar_AL.csv',
            'batch_size':1,
            'best_other_mse':'0.322',
            'best_other_mae':'0.370',
         },

}


if __name__ == '__main__':
    
    


    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ETTh2', help='dataset type')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--num_gpus', type=int,default=1)
    parser.add_argument('--future_token', type=int,default=3072)
    parser.add_argument('-t', '--task_list', action='append')
    parser.add_argument('--model_name',type=str)

    
    
    args = parser.parse_args()
    

    torch.set_float32_matmul_precision("high")

    

    strategy = DDPStrategy(find_unused_parameters=True)
    
    
    fabric = L.Fabric(devices=args.num_gpus, strategy=strategy)
    
    local_rank = fabric.global_rank
    
    if local_rank == 0:
        print(args)
        

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    model = model.to(local_rank).bfloat16()
    
    model = fabric.setup(model)
    name = args.model_name
        
    if local_rank == 0:
        with open(f'results.txt', 'a') as f:
                f.write(f"---------------------------------------------------------------------------------")
    
  
    
    for task_name in args.task_list:
       
        task = datasets_configs[task_name]
        data = task['data']
        root_path = task['root_path']
        data_path = task['data_path']
        best_mse = task['best_other_mse']
        best_mae = task['best_other_mae']
        batch_size = min(args.batch_size,task['batch_size'])
        seq_len = args.seq_len
        if local_rank == 0:
            with open(f'results.txt', 'a') as f:
                seconds_since_epoch = time.time()
                human_readable_time = datetime.datetime.fromtimestamp(seconds_since_epoch).strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{human_readable_time}-------------\n")
                
                
                
                
    

        data_set, data_loader = data_provider(data,root_path,data_path,batch_size = batch_size,seq_len=seq_len)
        data_loader = fabric.setup_dataloaders(data_loader)



        model.eval()
        preds = []
        truths = []
        preds_s = [[],[],[],[]]
        truths_s = [[],[],[],[]]
        intermediates = []
        xs = []

        seperate_s = [96, 192, 336,720]
        remains = args.future_token 
        prevs = 0

        with torch.no_grad():
            for idx,(x_ori,y,y1,y2,y3,y4) in enumerate(tqdm(data_loader,disable = local_rank != 0)):

                b,c = x_ori.shape[0],x_ori.shape[2]
                x = rearrange(x_ori, 'b l c -> (b c) l').float().to(local_rank).bfloat16().contiguous()
                y = rearrange(y1, 'b l c -> (b c) l').float()

                y_s = [y1,y2,y3,y4]
                res = []
                res1 = []
                res2 = []
                res3 = []

                logits = 0
                used = 0


                for history in [512,1024,2048,4096]:
                    if history > x.shape[1]:
                        continue
                    else:
                        used += 2

                    x_mean = x[:,-history:].mean(dim = -1,keepdims = True)
                    x_std = x[:,-history:].std(dim = -1,keepdims = True)

                    x_train = torch.cat((x[:,-history:],-x[:,-history:]),dim = 0)


                    logits_all = model(idx = x_train, future_token = args.future_token)


                    logits_all = rearrange(logits_all, '(t b) l c d -> b (l c) d t', t = 2)
                    logits += logits_all[...,0]  -logits_all[...,1].flip(dims = [-1]) 

                logits = logits / used



                x = torch.cat([x,logits[:,:720,49]],dim = -1).float()

                median = logits[:,:720,49].float()
                median = median[:,:720] 

                median0 =  rearrange(median, '(b c) l -> b l c',b = b).contiguous().cpu().detach().numpy()
                y0 = rearrange(y, '(b c) l -> b l c',b = b).contiguous().cpu().detach().numpy()



                for i, seperate in enumerate(seperate_s):
                    median_s =  logits[:,:seperate,49].float()
                    median_s = rearrange(median_s, '(b c) l -> b l c',b = b).contiguous().cpu().detach().numpy()
                    preds_s[i].append(median_s)
                    truths_s[i].append(y_s[i].contiguous().cpu().detach().numpy())


                xs.append(x_ori.contiguous().cpu().detach().numpy())



        def gather_losses(loss):
            """Gather loss values from all GPUs."""
            if dist.is_initialized():
                loss_tensor = torch.tensor([loss], device=local_rank)
                gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(args.num_gpus)]
                dist.all_gather(gathered_losses, loss_tensor)
                return torch.cat(gathered_losses).mean()
            else:
                return loss







        if local_rank == 0:
            print(f'Eval on {task_name}-{seq_len}...')
        mses = []
        maes = []


        for i, seperate in enumerate(seperate_s):
            if i == 4:
                break
            truths = truths_s[i]
            preds = preds_s[i]
            truths = np.concatenate(truths,axis = 0)
            preds = np.concatenate(preds,axis = 0)

            truths = rearrange(truths,'b l c -> b c l')
            preds = rearrange(preds,'b l c -> b c l')
            mask = np.isnan(truths).any(axis=2)

            truths1 = truths[~mask]
            preds1 = preds[~mask]

            truths = rearrange(truths,'b c l-> b l c')
            preds = rearrange(preds,'b c l-> b l c')

            mae, mse, rmse, mape, mspe = metric(preds1[:,:seperate], truths1[:,:seperate])
            mae,mse = gather_losses(mae),  gather_losses(mse)


            if local_rank == 0:
                print(f'ours-{name}: mse {mse:.4f} mae {mae:.4f}')
                mses.append(mse.cpu().numpy())
                maes.append(mae.cpu().numpy())
                with open(f'results.txt', 'a') as f:

                    f.write(f"ours-{name}, {data_path.split('.')[0]}-{args.seq_len}-{seperate}-{args.future_token}, mse,  {mse:.5f}, mae, {mae:.5f}\n")
        if local_rank == 0:

            print(f'ours-{name}-avg: mse {np.mean(mses):.3f} mae {np.mean(maes):.3f}')

        
            with open(f'results.txt', 'a') as f:
                f.write(f"ours-{name}, {data_path.split('.')[0]}-avg, mse, {np.mean(mses):.5f}, mae, {np.mean(maes):.5f}\n")
            print(f'best-avg: mse {best_mse} mae {best_mae}')


       
    dist.destroy_process_group()