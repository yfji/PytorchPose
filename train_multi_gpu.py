import data_loader
import mobilenet
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse
import torchvision.transforms as transforms
import os
import numpy as np

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_ids=[0,1]

def parse():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def construct_model():
    net = mobilenet.Mobilenet()
    #net.load_weights(model_path='models/model_iter_80000.pkl')
    net.cuda(gpu_ids[0])
    net = nn.DataParallel(net, device_ids=gpu_ids)
    return net

def main():
    anno_path='/mnt/sda6/Keypoint/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json'
    img_dir='/mnt/sda6/Keypoint/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902/'
    
    batch=10
    base_lr=0.00004
    decay_ratio=0.125
    max_iters=440000
    stepvalues=[140000, 360000, 440000]
    #stepvalues=[200,400,440000]
    g_steps=stepvalues[0]
    
    display=20
    snapshot=20000
    
    cudnn.benchmark = True
    
    log_file=open('unlabeled.log','w')
    d_loader=data_loader.DataLoader(anno_path=anno_path, img_dir=img_dir, log_file=log_file, transforms=transforms.ToTensor(), batch_size=batch)

    train_loader = torch.utils.data.DataLoader(
        		d_loader,
        		batch_size=batch, shuffle=True,
        		num_workers=6, pin_memory=True)
    
    net=construct_model()
    
    criterion_L1 = nn.MSELoss().cuda()
    criterion_L2 = nn.MSELoss().cuda()
    
    params = []
    for key, value in net.named_parameters():
#        print(key,value.size())
        if value.requires_grad != False:
            print(key,value.shape)
            params.append({'params': value, 'lr': base_lr})

   
    optimizer = torch.optim.SGD(params, base_lr, momentum=0.9,
	                            weight_decay=0.0005)
    
    iters=0
    lr=base_lr
    step=0
    step_index=0
    
    heat_weight=32*32*22*0.5
    
    while iters<max_iters:
        for i, (data, label) in enumerate(train_loader):
            data_var=torch.autograd.Variable(data.cuda(async=True))
            label_var=torch.autograd.Variable(label.cuda(async=True))
            keypoints, pafs=net(data_var)
            
            loss_L1=criterion_L1(keypoints, label_var[:,:d_loader.num_parts+1])*heat_weight
            loss_L2=criterion_L2(pafs, label_var[:,d_loader.num_parts+1:])*heat_weight

            loss=loss_L1+loss_L2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
   
            rate=lr*np.power(decay_ratio, 1.0*step/g_steps)
#            for param_group in optimizer.param_groups:
#                param_group['lr']=rate

            if iters%display==0:
                print('[%d/%d] learn rate: %e\nloss: %f\nloss_L1: %f\nloss_L2: %f'%(iters, max_iters, lr, loss, loss_L1, loss_L2))
                
            if iters==stepvalues[step_index]:
                print('learning rate decay: %e'%rate)
                for param_group in optimizer.param_groups:
                    param_group['lr']=rate
                step=0
                lr=rate
                g_steps=stepvalues[step_index+1]-stepvalues[step_index]
                step_index+=1
            if iters>0 and iters%snapshot==0:
                model_name='models/model_iter_%d.pkl'%iters
                print('Snapshotting to %s'%model_name)
                torch.save(net.state_dict(),model_name)
            step+=1
            iters+=1
            if iters==max_iters:
                break
    model_name='models/model_iter_%d.pkl'%max_iters
    print('Snapshotting to %s'%model_name)
    torch.save(net.state_dict(),model_name)
    log_file.close()

if __name__=='__main__':
    main()
    