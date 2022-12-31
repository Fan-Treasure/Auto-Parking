import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph import Sequential,Conv2D,Pool2D,Linear,BatchNorm,Pool2D
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.base import to_variable
# Convolutional neural network (two convolutional layers)

class ConvBNLayer(fluid.dygraph.Layer):
    """
    卷积 + 批归一化，BN层之后激活函数默认用leaky_relu
    """
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act="ReLU",
                 is_test=True):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv2D(
            num_channels=ch_in,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02)),
            bias_attr=False,
            act=None)

        self.batch_norm = BatchNorm(
            num_channels=ch_out,
            is_test=is_test,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02),
                regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                regularizer=L2Decay(0.)))
        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'ReLU':
            out = fluid.layers.relu(x=out)
        return out


class ConvNet(fluid.dygraph.Layer):
    def __init__(self,name_scope,class_num=360,is_test=False):
        super(ConvNet, self).__init__(name_scope)
                
        self.convs = Sequential(          #99*99
            ConvBNLayer(
                ch_in=1,
                ch_out=48,
                filter_size=5,
                stride=1,
                padding=2,
                is_test=is_test
                ),
            Pool2D(pool_size=3, pool_stride =3, pool_type="avg"),  #33*33
            
            ConvBNLayer(
                ch_in=48,
                ch_out=48,
                filter_size=5,
                stride=2,
                padding=2,
                is_test=is_test
                ),
            Pool2D(pool_size=3, pool_stride=2, pool_type="avg"), #8*8
           
        )
        
        
        self.convs_red = Sequential(          #99*99
            Pool2D(pool_size=6, pool_stride=5, pool_padding =1, pool_type="max"),  #20*20
            
            ConvBNLayer(
                ch_in=1,
                ch_out=32,
                filter_size=6,
                stride=2,
                padding=3,
                is_test=is_test
                ),
            
            Pool2D(pool_size=3, pool_stride=2,pool_type="avg"), #5*5
        )
        
        self.convs_grad = Sequential(          #99*99
            ConvBNLayer(
                ch_in=1,
                ch_out=48,
                filter_size=5,
                stride=1,
                padding=2,
                is_test=is_test
                ),
            
            Pool2D(pool_size=3, pool_stride=3, pool_type="avg"),  #33*33
            
            ConvBNLayer(
                ch_in=48,
                ch_out=48,
                filter_size=5,
                stride=2,
                padding=2,
                is_test=is_test
                ),
            
            Pool2D(pool_size=3, pool_stride=2, pool_type="avg"), #8*8
           
        )
        
        self.fc_ratio = Linear(1, 512, act="sigmoid")
              
               
        self.fc = Linear(8*8*48+5*5*32+8*8*48+512, class_num, act="sigmoid")
        
           
    def forward(self, x0, x1, x2, r):
        y1 = self.convs(x1)
        y1 = fluid.layers.reshape(x=y1,shape=[y1.shape[0], -1])
                
        y2 = self.convs_red(x0)
        y2 = fluid.layers.reshape(x=y2,shape=[y2.shape[0], -1])
        
        y3 = self.convs_grad(x2)
        y3 = fluid.layers.reshape(x=y3,shape=[y3.shape[0], -1])
        
        y4 = self.fc_ratio(r)
        
        ya = fluid.layers.concat(input=[y1,y2,y3,y4],axis=1)
        
        out = self.fc(ya)
        #out2 = self.fc_red(out_red1.view(out_red1.size(0),-1))*0
             
        
        return out