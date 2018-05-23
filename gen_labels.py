import numpy as np

def putGaussianMaps(label, all_keypoints, stride=8, sigma=7.0):
    for i in range(all_keypoints.shape[0]):
        putGaussianMap(label, all_keypoints[i], stride=stride, sigma=sigma)
        
def putGaussianMap(label, keypoints, stride=8, sigma=7.0):
    start = stride / 2.0 - 0.5
    for i in range(label.shape[2]-1):    #[h,w,c]
        kp=keypoints[i]
        if kp[-1]<=2:
            for y in range(label.shape[0]):
                for x in range(label.shape[1]):
                    yy = start + y * stride
                    xx = start + x * stride
                    dis = ((xx - kp[0]) * (xx - kp[0]) + (yy - kp[1]) * (yy - kp[1])) / 2.0 / sigma / sigma
                    if dis > 4.6052:
                        continue
                    label[y,x,i] += np.exp(-dis)
                    label[y,x,i]=min(1,label[y,x,i])
    label[:,:,-1]=np.max(label[:,:,:-1],axis=2)

def putVecMaps(label, all_keypoints, im_w, im_h, stride=8, sigma=7.0, thre=1.0):
    paf_from = [14,1,2,14,4,5,14,7,8,14,10,11,14]
    paf_to   = [1, 2,3,4, 5,6,7, 8,9,10,11,12,13]
    
    grid_x=int(im_w/stride)
    grid_y=int(im_h/stride)
    
    for k in range(all_keypoints.shape[0]):      #every one
        keypoints=all_keypoints[k]
        count=np.zeros((grid_y,grid_x), dtype=np.int32)

        for i in range(len(paf_from)):
            from_pt=keypoints[paf_from[i]-1]
            to_pt=keypoints[paf_to[i]-1]
            if from_pt[-1]<=2 and to_pt[-1]<=2:
                putVecMap(label[:,:,2*i:2*i+2],from_pt[:2], to_pt[:2], count, grid_x, grid_y, stride=stride, sigma=sigma, thre=thre)
        
#        for i in range(numOtherPeople)

def putVecMap(label, from_pt, to_pt, count, grid_x, grid_y, stride=8, sigma=7.0, thre=1.0):
    grid_from=1.0*from_pt/stride
    grid_to=1.0*to_pt/stride
    vec=grid_to-grid_from
    vec_l1=np.sqrt(vec[0]**2+vec[1]**2)
    vec/=vec_l1
    
    min_x = max(int(round(min(grid_from[0], grid_to[0])-thre)), 0)
    max_x = min(int(round(max(grid_from[0], grid_to[0])+thre)), grid_x)

    min_y = max(int(round(min(grid_from[1], grid_to[1])-thre)), 0)
    max_y = min(int(round(max(grid_from[1], grid_to[1])+thre)), grid_y)
    
    for g_y in range(min_y, max_y):
        for g_x in range(min_x, max_x):
            ba_x = g_x - grid_from[0]
            ba_y = g_y - grid_from[1]
            dist = abs(ba_x*vec[1] -ba_y*vec[0])

            if dist <= thre:
                cnt = count[g_y, g_x]
                if cnt == 0:
                    label[g_y,g_x,0]=vec[0]
                    label[g_y,g_x,1]=vec[1]
                else:
                    label[g_y,g_x,0]=(1.0*label[g_y,g_x,0]*cnt+vec[0])/(cnt+1)
                    label[g_y,g_x,1]=(1.0*label[g_y,g_x,1]*cnt+vec[1])/(cnt+1)
                    count[g_y, g_x] = cnt + 1;
    
"""
1:right shoulder
2:right elbow
3:right wrist

4:left shoulder
5:left elbow
6:left wrist

7:right hip
8:right knee
9:right ankle

10:left hip
11:left knee
12:left ankle

13:head top
14:neck
"""

    