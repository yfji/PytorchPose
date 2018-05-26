import torch
import torch.utils.data as data
import json
import data_augment as da
import numpy as np
import cv2
import os.path as op
from gen_labels import putGaussianMaps, putVecMaps

class DataLoader(data.Dataset):
    def __init__(self, anno_path=None, img_dir=None, log_file=None, transforms=None, batch_size=8):
        self.anno_path=anno_path
        self.image_dir=img_dir
        self.transforms=transforms
        
        self.dataset=json.load(open(self.anno_path, 'r'))   #list
        self.num_samples=len(self.dataset)
        
        self.batch_size=batch_size
        self.center_perterb_max=20
        self.angle_max=25
        self.target_scale=0.9
        self.scale_range=[0.8,1.2]
        self.stride=8
        
        self.num_parts=14
        self.num_pafs=13
        self.label_channels=self.num_parts+2*self.num_pafs+1
        
        self.sigma=5.0
        self.vec_thres=0.8
        
        self.visualize=True
        self.savedir='./visualize'
        self.image_index=0
        self.net_input_size=256
        
        self.log_file=log_file
        self.shuffle()
        
    def __getitem__(self, index):
        image, all_keypoints, img_path=self.parse_anno(index)
        
        base_scale=1.0*self.target_scale/(1.0*image.shape[0]/self.net_input_size)
        
        label_side=int(self.net_input_size/self.stride)
        imagelabel=np.zeros((label_side,label_side,self.label_channels),dtype=np.float32)

        valid_samples=0
        
        for i in range(all_keypoints.shape[0]):
            kpt=all_keypoints[i]
            valid_indices=np.nonzero(kpt[:,2]<=2)[0]
            valid_samples+=len(valid_indices)
            
        if valid_samples>0:
            image=da.aug_scale(image, base_scale, self.scale_range, all_keypoints)
            image=da.aug_rotate(image, self.angle_max, all_keypoints)
            center=np.mean(all_keypoints.reshape(-1,3)[:,:2],axis=0) 
            image,flag=da.aug_crop(image, center, self.net_input_size, self.center_perterb_max, all_keypoints)
        
            if flag==0:
                g_map=imagelabel[:,:,-1]
                g_map=cv2.resize(g_map, (0,0), fx=self.stride,fy=self.stride,interpolation=cv2.INTER_CUBIC)
                raw_image=image.astype(np.uint8)
                vis_img=self.add_weight(raw_image,g_map)
                cv2.imwrite('0.jpg', vis_img)
                assert(0)
                
            putGaussianMaps(imagelabel[:,:,:self.num_parts+1], all_keypoints, stride=self.stride, sigma=self.sigma)
            putVecMaps(imagelabel[:,:,self.num_parts+1:], all_keypoints, image.shape[1], image.shape[0], stride=self.stride, sigma=self.sigma, thre=self.vec_thres)
               
            if self.visualize and self.image_index<50:
                image_kpt=image.astype(np.uint8)
                image_paf=image_kpt.copy()
                vis_img=self.add_weight(image_kpt, imagelabel, name='kpt')
                cv2.imwrite(op.join(self.savedir, 'keypoints','sample_%d.jpg'%self.image_index),vis_img)
                vis_img=self.add_weight(image_paf, imagelabel, name='paf')
                cv2.imwrite(op.join(self.savedir, 'pafs','sample_%d.jpg'%self.image_index),vis_img)
                self.image_index+=1
        else:
            c_image=128*np.ones((self.net_input_size,self.net_input_size,3),dtype=np.float32)
            h=min(c_image.shape[0],image.shape[0])
            w=min(c_image.shape[1],image.shape[1])
            c_image[:h,:w,:]=image[:h,:w,:]
            image=c_image
            self.log_file.write(img_path+'\n')
        image-=128.0
        image/=255.0
        return torch.from_numpy(image.transpose(2,0,1)), torch.from_numpy(imagelabel.transpose(2,0,1))

    def __len__(self):
        return self.num_samples
    
    def shuffle(self):
        self.random_order=np.random.permutation(np.arange(self.num_samples))
        self.cur_index=0
        
    def parse_anno(self, index):
        entry=self.dataset[self.random_order[index]]
        img_path=op.join(self.image_dir, entry['image_id']+'.jpg')
        assert(op.exists(img_path))
        
        image=cv2.imread(img_path)
#        hand_pos=entry['objpos']
        kpt_annot=entry['keypoint_annotations']
        num_people=len(kpt_annot.keys())
        keypoints=np.zeros((num_people, self.num_parts, 3), dtype=np.float32)#21x3
        
        cnt=0
        for i, kpt in kpt_annot.items():
            keypoints[cnt]=np.asarray(kpt).reshape(-1,3)
            cnt+=1
        #avoid kpts coordinates out of image
        return image, keypoints, img_path
    
    
        
    def add_weight(self,image, imagelabel, name='kpt'):
        g_map=imagelabel[:,:,self.num_parts]
        g_map=cv2.resize(g_map, (0,0), fx=self.stride,fy=self.stride,interpolation=cv2.INTER_CUBIC)
        paf_map=np.max(imagelabel[:,:,self.num_parts+1:], axis=2)
        paf_map=cv2.resize(paf_map, (0,0), fx=self.stride,fy=self.stride,interpolation=cv2.INTER_CUBIC)

        heatmap_bgr=np.zeros(image.shape, dtype=np.uint8)
        for i in range(heatmap_bgr.shape[0]):
            for j in range(heatmap_bgr.shape[1]):
                if name=='kpt':
                    heatmap_bgr[i,j]=self.getJetColor(g_map[i,j],0,1)
                elif name=='paf':
                    heatmap_bgr[i,j]=self.getJetColor(paf_map[i,j],0,1)
        out_image=cv2.addWeighted(image, 0.7, heatmap_bgr, 0.3, 0).astype(np.uint8)
        return out_image
    
    def getJetColor(self, v, vmin, vmax):
        c = np.zeros((3))
        if (v < vmin):
            v = vmin
        if (v > vmax):
            v = vmax
        dv = vmax - vmin
        if (v < (vmin + 0.125 * dv)): 
            c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
        elif (v < (vmin + 0.375 * dv)):
            c[0] = 255
            c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
        elif (v < (vmin + 0.625 * dv)):
            c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
            c[1] = 255
            c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
        elif (v < (vmin + 0.875 * dv)):
            c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
            c[2] = 255
        else:
            c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5                      
        return c    

