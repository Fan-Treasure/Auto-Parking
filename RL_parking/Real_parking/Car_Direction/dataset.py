import os
import numpy as np
import imageio

from skimage.transform import resize
from skimage import img_as_ubyte
from skimage import filters 
from skimage import color,exposure
from skimage.filters import threshold_local
from skimage import color,exposure
from skimage import transform as tf
from skimage import exposure
from Car_Direction import tools


def process_data(im_path,box,angle,aug=True):
    neww = 100

    c1,r1,width,height = box
    c2,r2 = c1+width,r1+height
    
    im = imageio.imread(im_path)
    if aug:
        im = exposure.adjust_gamma(im, np.random.rand()+0.5)
    
    (h,w,_) = im.shape
    
    im = im/255
            
    im_hsv = color.rgb2hsv(im)
    hh,ss,vv = im_hsv[:,:,0],im_hsv[:,:,1],im_hsv[:,:,2]
    hh = np.abs(hh)
    hh = np.where(hh>0.93,hh,0)
    im_red =  hh * ss * vv
        
    im_gray = color.rgb2gray(im)
    im_g = filters.scharr(im_gray)
    
    im = np.stack((im_red,im_gray,im_g),axis=2)
    
    r1,c1,r2,c2 = int(r1),int(c1),int(r2),int(c2)
    
    mv = min([5,r1,c1,h-r2,w-c2])
    
    mvs = np.array([[0,0,0,0],
                    [-mv,0,-mv,0],
                    [0,-mv,0,-mv],
                    [mv,0,mv,0],
                    [0,mv,0,mv]])
    
    if aug:
        u = np.random.permutation(5)[0]     
    else:
        u = 0   
    
    rcs = np.array([r1,c1,r2,c2])+mvs[u,:]
    imn = im[rcs[0]:rcs[2],rcs[1]:rcs[3],:]
    imn = resize(imn,[neww,neww],mode='constant',cval=0)

    imn = (imn-0.4627)/np.array([0.99601961, 1., 0.78049476])+0.5

    ratio = 1-height/width if width>height else width/height-1    
    
    return imn.astype(np.float32),ratio.astype(np.float32),angle.astype(np.float32)    
       

def load_data(data_path):
    ims = []
    boxes = []

    for i in range(1):
        seq_id = f'{i+1}'
        seq_id = '0'*(2-len(seq_id))+seq_id
        bbox_path = os.path.join(data_path, f'bbox_{seq_id}.txt') 
            
        with open(bbox_path,'r') as f:
            bboxs = f.readlines()
        
        bboxs = [b.strip().split() for b in bboxs]
        bboxs = np.array([[float(b) for b in bs] for bs in bboxs])
        
        boxes.append(bboxs)
        
        im_num = np.size(bboxs,0)
        
        seqs = []
        for j in range(im_num):
            x = 3*j
            im_id = f'{j}'
            im_id_2 = f'{x}'
            im_path = os.path.join(data_path, f'car_{im_id}_{im_id_2}.jpg')
            
            seqs.append(im_path)
        
        ims.append(seqs)

    #角度信息
    with open(os.path.join(data_path,'tripod-seq.txt'),'r') as f:
        tseq = f.readlines()
        
    angle_info = np.array([list(map(float,txt.strip().split())) for txt in tseq[-3:]],dtype=np.int)

    #时间信息
    with open(os.path.join(data_path,'times.txt'),'r') as f:
        tms = f.readlines()
        
    secs = [np.array(list(map(int,txt.strip().split()))) for txt in tms]

    #计算角度值
    angles = []
    for i in range(1):
        iterval = 360/secs[i][angle_info[0,i]-1]
        base = secs[i][angle_info[1,i]-1]
        direc = angle_info[2,i]
        
        aseqs = (secs[i]-base)*iterval*direc
        aseqs = np.mod(aseqs,360)
        
        angles.append(aseqs)

    return ims,boxes,angles

class Dataset():

    def __init__(self, data_path=None, batch_size=8, mode='train'):
        if data_path is not None:
            with open(data_path,'r') as f:
                txts = np.array([line.split() for line in f.readlines()])
                self.im_paths = txts[:,0]

                self.boxes = np.array([list(map(float,bb)) for bb in txts[:,1:5]],dtype=np.float32)
                self.angles = np.array(list(map(float,txts[:,5])),dtype=np.float32)

            assert len(self.im_paths)==len(self.boxes)
            assert len(self.im_paths)==len(self.angles)

            self.length = len(self.im_paths)
        
        self.batch_size = batch_size
        self.mode = mode

    def split_train_val(self,ratio = 0.9):
        train = Dataset()
        val = Dataset(mode='val')

        enum = int(self.length*0.1)
        nind = np.random.permutation(self.length)

        train.im_paths = self.im_paths[nind[enum:]]
        train.boxes = self.boxes[nind[enum:]]
        train.angles = self.angles[nind[enum:]]
        train.length = len(train.im_paths)

        val.im_paths = self.im_paths[nind[:enum]]
        val.boxes = self.boxes[nind[:enum]]
        val.angles = self.angles[nind[:enum]]
        val.length = len(val.im_paths)

        return train,val

    def get_label(self):
        return self.angles

    def __len__(self):
        return self.length

    def __iter__(self):
        if self.mode=='train':
            self.inds = np.random.permutation(self.length)
        else:
            self.inds = np.arange(self.length)
        
        self.cur = 0
        return self

    def __next__(self):
        if self.cur+self.batch_size <= self.length:
            ims = []
            ratios = []
            angles = []

            for i in self.inds[self.cur:self.cur+self.batch_size]:
                im,ratio,angle = process_data(self.im_paths[i],self.boxes[i],self.angles[i],self.mode=='train') 
                ims.append(im)
                ratios.append(ratio)
                angles.append(tools.angle2label(angle))

            self.cur+=self.batch_size

            ims = np.stack(ims,axis=0)
            ims = np.transpose(ims,(0,3,1,2))
            ratios = np.expand_dims(np.stack(ratios,axis=0),axis=1)
            angles = np.vstack(angles)

            return ims,ratios,angles
        else:
            raise StopIteration
