import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
from Car_Direction import cnn
from Car_Direction import tools
import numpy as np
import os

# Train the model
def train(loader,eloader,num_epochs,state,resume=False,optim="Adam"):
    tlosses = []
    vlosses = []
    with fluid.dygraph.guard():
        
        model = cnn.ConvNet("convnet",class_num=360,is_test=False)
        if resume:
            model_state_dict, _ = fluid.load_dygraph(os.path.join(state['save_path'],'best_model'))
            model.load_dict(model_state_dict)

        criterion = fluid.layers.mse_loss
        if optim=="Adam":
            optimizer = fluid.optimizer.Adam(
                        learning_rate=state['lr'],  #提升点：可以调整学习率，或者设置学习率衰减
                        parameter_list=model.parameters())   # 提升点： 可以添加正则化项
        else:
            optimizer = fluid.optimizer.Momentum(
                        learning_rate=state['lr'],  #提升点：可以调整学习率，或者设置学习率衰减
                        momentum=0.90,
                        parameter_list=model.parameters())   # 提升点： 可以添加正则化项

        model.train()
        total_step = len(loader)
        batch_size = loader.batch_size
        fail_ctr = 0
        for epoch in range(num_epochs):
            
            tloss = []
            vloss = float('inf')
            
            for i, (images,ratios,labels) in enumerate(loader):
                x0,x1,x2 = images[:,[0],:,:],images[:,[1],:,:],images[:,[2],:,:]
                x0,x1,x2 = to_variable(x0),to_variable(x1),to_variable(x2)

                ratios = to_variable(ratios)
                labels = to_variable(labels)

                # Forward pass
                outputs = model(x0,x1,x2,ratios)
                loss = criterion(input=outputs, label=labels)
                # Backward and optimize
                optimizer.clear_gradients()
                loss.backward()
                optimizer.minimize(loss)
                
                tloss.append(loss.numpy())

                if i%20 ==0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}/-- ' 
                        .format(epoch+1, num_epochs, (i+1)*batch_size, total_step, np.mean(tloss)))
            
            tloss = np.mean(tloss)
            vloss = val(model,eloader,criterion)
            tlosses.append(tloss)
            vlosses.append(vloss)
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}/{:.4f} ' 
                .format(epoch+1, num_epochs, i+1, total_step, np.mean(tloss), vloss))
            
            
            if vloss<state["best_score"]:
                state['best_score'] = vloss
                fluid.save_dygraph(model.state_dict(), os.path.join(state['save_path'],'best_model'))
                print(f"Best model Saved, score: {vloss} \n")
                fail_ctr = 0
            else:
                fail_ctr+=1
                print(f"Fail Counter: {fail_ctr} \n")
                if fail_ctr>state['max_ctr']:
                    break

                

    return tlosses,vlosses

# Test the model
def val(model,loader,criterion):
    model.eval()
    losses = []
    for (images,ratios,labels) in loader:
        x0,x1,x2 = images[:,[0],:,:],images[:,[1],:,:],images[:,[2],:,:]
        x0,x1,x2 = to_variable(x0),to_variable(x1),to_variable(x2)
        ratios = to_variable(ratios)
        labels = to_variable(labels)
        
        outputs = model(x0,x1,x2,ratios)
        loss = criterion(input=outputs, label=labels)
        
        losses.append(loss.numpy())
    
    model.train()
    return np.mean(losses)

# Test the model
def test(model_path,loader):

    with fluid.dygraph.guard():
        model = cnn.ConvNet("convnet",class_num=360,is_test=True)
        model_state_dict, _ = fluid.load_dygraph(model_path)
        model.load_dict(model_state_dict)
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        
        correct = 0
        total = 0
        
        pr = []
        
        for (images,ratios,_) in loader:
            x0,x1,x2 = images[:,[0],:,:],images[:,[1],:,:],images[:,[2],:,:]
            x0,x1,x2 = to_variable(x0),to_variable(x1),to_variable(x2)

            ratios = to_variable(ratios)
            #labels = to_variable(labels)

            outputs = model(x0,x1,x2,ratios)

            pr.append(outputs.numpy())

        pr = np.concatenate(pr,axis=0)
                
        pr_a = np.zeros(np.size(pr,0),dtype=np.float32)

        for i in range(np.size(pr,0)):
            pr_a[i] = tools.label2angle(pr[i,:])
        
        
        return pr_a