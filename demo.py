import mobilenet as model
import torch
import cv2
import argparse
import os
import os.path as op
import numpy as np
import scipy.ndimage.filters as filt
import matplotlib.cm as cm
import math
import time

limbSeq=[[14,1],[1,2],[2,3],[14,4],[4,5],[5,6],  [14,7], [7,8],  [8,9],  [14,10],[10,11],[11,12],[14,13]]
map_idx=[[0,1], [2,3],[4,5],[6,7], [8,9],[10,11],[12,13],[14,15],[16,17],[18,19],[20,21],[22,23],[24,25]]

def calc_peaks(keypoints_map, parts=14, peak_thre=0.2):
    all_peaks = []
    peak_counter = 0
    for p in range(parts):
        kpt_map_ori=keypoints_map[:,:,p]
        kpt_map=filt.gaussian_filter(kpt_map_ori, sigma=3)
        
        map_left = np.zeros(kpt_map.shape)
        map_left[1:,:] = kpt_map[:-1,:]
        map_right = np.zeros(kpt_map.shape)
        map_right[:-1,:] = kpt_map[1:,:]
        map_up = np.zeros(kpt_map.shape)
        map_up[:,1:] = kpt_map[:,:-1]
        map_down = np.zeros(kpt_map.shape)
        map_down[:,:-1] = kpt_map[:,1:]
        
        peaks_binary = np.logical_and.reduce((kpt_map>=map_left, kpt_map>=map_right, kpt_map>=map_up, kpt_map>=map_down, kpt_map > peak_thre))
        peaks=np.hstack((np.nonzero(peaks_binary)[1].reshape(-1,1), np.nonzero(peaks_binary)[0].reshape(-1,1)))
        peaks_score=[(x[0],x[1],)+(kpt_map_ori[x[1],x[0]],) for x in peaks]
        peak_ids=range(peak_counter, peak_counter+len(peaks_score))
        
        peaks_score_id=[peaks_score[i]+(peak_ids[i],) for i in range(len(peak_ids))]
        '''
        peaks_score_id
        [x,y,v,score,id]
        '''
        all_peaks.append(peaks_score_id)
        peak_counter+=len(peaks_score)
        
    return all_peaks

def calc_connections(pafs, all_peaks, parts=14, dist_thre=0.8, score_thre=0.05):
    connection_all=[]
    specials=[]
    mid_num=10
    for k in range(len(map_idx)):
        paf_xy=pafs[:,:,[x for x in map_idx[k]]]
        candA=all_peaks[limbSeq[k][0]-1]
        candB=all_peaks[limbSeq[k][1]-1]
        nA=len(candA)
        nB=len(candB)
        
        if nA!=0 and nB!=0:
            conn_candidate=[]
            for i in range(nA):
                for j in range(nB):
                    vec=np.subtract(candB[j][:2],candA[i][:2]).astype(np.float32)
                    vec_l1=np.sqrt(np.sum(vec**2))
                    vec/=vec_l1
                    
                    startend=np.hstack((np.linspace(candA[i][0], candB[j][0], num=mid_num).reshape(-1,1),np.linspace(candA[i][1], candB[j][1], num=mid_num).reshape(-1,1)))
                    vec_x = np.array([paf_xy[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in range(mid_num)])
                    vec_y = np.array([paf_xy[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in range(mid_num)])
                    
#                    dist=np.multiply(vec_x,vec[1])-np.multiply(vec_y,vec[0])    #x1y2-x2y1
                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts)
                    criterion1 = len(np.nonzero(score_midpts > score_thre)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        conn_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])
            conn_candidate = sorted(conn_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(conn_candidate)):
                i,j,s = conn_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break
    
            connection_all.append(connection)
        else:
            specials.append(k)
            connection_all.append([])
    
    subset = -1 * np.ones((0, parts+2))
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    
    print('============')
    for k in range(len(map_idx)):
        if k in specials:
            continue
        partAs = connection_all[k][:,0]
        partBs = connection_all[k][:,1]
        indexA, indexB = np.array(limbSeq[k]) - 1

        for i in range(len(connection_all[k])):
            found = 0
            subset_idx = [-1, -1]
            for j in range(len(subset)): #1:size(subset,1):
                if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                    subset_idx[found] = j
                    found += 1
            if found == 1:
                j = subset_idx[0]
                if(subset[j][indexB] != partBs[i]):
                    subset[j][indexB] = partBs[i]
                    subset[j][-1] += 1
                    subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
            elif not found:
                row = -1 * np.ones(parts+2)
                row[indexA] = partAs[i]
                row[indexB] = partBs[i]
                row[-1] = 2
                row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                subset = np.vstack((subset, row))
    
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    return subset, candidate

def draw_keypoints(image, all_peaks, subset, candidates, parts=14):
    colors = [[255, 0, 0], [255, 85, 0], 
              [255, 170, 0], [255, 255, 0], 
              [170, 255, 0], [85, 255, 0], 
              [0, 255, 0],   [0, 255, 85], 
              [0, 255, 170], [0, 255, 255], 
              [0, 170, 255], [0, 85, 255], 
              [0, 0, 255], [85, 0, 255]
              ]
    cmap = cm.get_cmap('hsv')
    canvas=image.copy()
    
    for i in range(parts):
        rgba = np.array(cmap(1 - i/18. - 1./36))
        rgba[0:3] *= 255
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
            
    for i in range(parts-1):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])-1]   #id of part
            if -1 in index:
                continue
#            cur_canvas = canvas.copy()
            Y = candidates[index.astype(int), 0]
            X = candidates[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), 4), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])
#            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas
       

    
def run_model(image_name, net, thre=0.2):
    image_ori=cv2.imread(image_name)
    input_side=256
    scale=1.0*input_side/max(image_ori.shape[0],image_ori.shape[1])
    image=cv2.resize(image_ori, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pad_image=np.zeros((input_side,input_side,3),dtype=np.float32)
    pad_image+=128
    pad_image[:image.shape[0],:image.shape[1],:]=image.astype(np.float32)
    
    data=pad_image.transpose(2,0,1)
    data=(data-128)/255.0
    data=torch.FloatTensor(data[np.newaxis,:,:,:])
    
    start=time.time()
    keypoints, pafs=net(torch.autograd.Variable(data.cuda()))
    elapsed=time.time()-start

    keypoints=keypoints.data.cpu().numpy().squeeze()
    pafs=pafs.data.cpu().numpy().squeeze()
    
    stride=8
    keypoints=cv2.resize(keypoints.transpose(1,2,0), (0,0), fx=1.0*stride/scale,fy=1.0*stride/scale, interpolation=cv2.INTER_CUBIC)
    pafs=cv2.resize(pafs.transpose(1,2,0), (0,0), fx=1.0*stride/scale,fy=1.0*stride/scale, interpolation=cv2.INTER_CUBIC)
    
    all_peaks=calc_peaks(keypoints_map=keypoints, peak_thre=thre)    
    subset, candidates=calc_connections(pafs, all_peaks)
    
    
    kpt_image=draw_keypoints(image_ori, all_peaks, subset, candidates)
    
    return kpt_image, elapsed
    
def main(parser):
    args=parser.parse_args()
    threshold=args.threshold
    filename=args.filename
    imagedir=args.imagedir
    savedir=args.savedir
    net_type=args.name
    
    net=None
    if net_type=='res18':
        net=fpn.pose_estimation(pretrain=True)
        model_path='models_fpn/model_iter_20000.pkl'
        net.load_weights(model_path=model_path)
    elif net_type=='mobilenet':
        net=model.Mobilenet(pretrain=True)
        model_path='models/model_iter_100000.pkl'
        net.load_weights(model_path=model_path)
    else:
        raise Exception('Unknown network architecture')
        
    net.cuda()
    
    if filename!='' and imagedir!='':
        raise Exception('Only one of image_dir and image can be used')
    if imagedir!='':
        if imagedir[0]!='/':
            imagedir=op.join(os.getcwd(),imagedir)
        if savedir=='':
            savedir=op.join(os.getcwd(),'preds')
    if savedir[0]!='/':
        savedir=op.join(os.getcwd(),savedir)
        imagenames=os.listdir(imagedir)
        num_samples=len(imagenames)
        cnt=0
        for imagename in imagenames:
            imagename_no_format=imagename[:imagename.rfind('.')]
            image_path=op.join(imagedir, imagename)
            
            kpt_image, elapse=run_model(image_path, net, thre=threshold)
                        
            print('[%d/%d] %s. time: %.3fms'%(cnt+1,num_samples,imagename, elapse*1000))

            cv2.imwrite(op.join(savedir,'%s_keypoints.jpg'%imagename_no_format), kpt_image)
            cnt+=1
    print('done')
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', dest='name', type=str, default='res18',help='mobilenet or resnet18')
    parser.add_argument('--thresh', dest='threshold', type=float,default=0.2, help='threshold for heatmap')
    parser.add_argument('--image', dest='filename', type=str, default='', help='image file path')
    parser.add_argument('--imagedir', dest='imagedir', type=str, default='', help='directory containing images')
    parser.add_argument('--savedir', dest='savedir', type=str, default='', help='directory to save predicted images')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main(parser)