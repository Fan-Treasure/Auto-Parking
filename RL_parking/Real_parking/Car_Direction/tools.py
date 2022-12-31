import numpy as np

class_num = 360
idx = np.arange(class_num)
mn = np.mean(idx)
rg = np.max(idx)

idx = (idx-mn)/rg*2

pdf = np.exp(-np.power(idx,2)/0.005).astype(np.float32)
pdf = np.hstack((pdf[int(class_num/2):],pdf[:int(class_num/2)]))


def angle2label(ag,pdf=pdf):
    
    class_num = len(pdf)
    
    csG = 360/class_num
        
    idx = int(np.floor(ag/csG))
    
    l = len(pdf)
        
    label = np.hstack((pdf[l-idx:],pdf[:l-idx]))
    
    return label
    
        
        
def label2angle(lb):
    class_num = len(lb)
    csG = 360/class_num
    
    idm = np.argmax(lb)
    
    return idm*csG+csG/2