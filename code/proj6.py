
import os, sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import skimage
import skimage.io
import tensorflow as tf

# Define CNN architecture
def apnaModel(Img, ImageSize, MiniBatchSize):

    num_classes = 62

    net = Img
    net = tf.layers.conv2d(inputs = net, name='layer_conv1', padding='same',filters = 32, kernel_size = 5, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name ='layer_bn1')
    net = tf.nn.relu(net, name = 'layer_Relu1')
    net = tf.layers.conv2d(inputs = net, name = 'layer_conv2', padding= 'same', filters = 32, kernel_size = 5, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn2')
    net = tf.nn.relu(net, name = 'layer_Relu2')
    net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)

    net = tf.layers.dropout(net,rate=0.5)

    net = tf.layers.conv2d(inputs = net, name = 'layer_conv3', padding= 'same', filters = 64, kernel_size = 5, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn3')
    net = tf.nn.relu(net, name = 'layer_Relu3')


    net = tf.layers.conv2d(inputs = net, name = 'layer_conv4', padding= 'same', filters = 32, kernel_size = 5, activation = None)
    net = tf.layers.batch_normalization(inputs = net,axis = -1, center = True, scale = True, name = 'layer_bn4')
    net = tf.nn.relu(net, name = 'layer_Relu4')

    net  = tf.layers.max_pooling2d(inputs = net, pool_size = 2, strides = 2)

    net = tf.layers.flatten(net)

    net1 = tf.layers.dense(inputs = net, name ='layer_fc1', units = 128, activation = tf.nn.relu)

    net2 = tf.layers.dense(inputs = net1, name ='layer_fc2',units=256, activation=tf.nn.relu)

    net3 = tf.layers.dense(inputs = net2, name='layer_fc_out', units = num_classes, activation = None)

    prLogits = net3
    prSoftMax = tf.nn.softmax(logits = prLogits)

    return prLogits, prSoftMax

# Helper function to run forward pass of CNN and predict class of traffic sign
def Classify(image):
    tf.reset_default_graph()
    ImgPH = tf.placeholder('float', shape=(1, 64, 64, 3))
    _, prSoftMaxS = apnaModel(ImgPH, 0, 1)
    Saver = tf.train.Saver()
    with tf.Session() as sess:
        Saver.restore(sess,'../data/Checkpoints/4model.ckpt')
        I1Batch=[]
        I1=image
        I1 = (I1-np.mean(I1))/255
        
        I1=cv2.resize(I1,(64,64))
        I1Batch.append(I1)
        FeedDict = {ImgPH: I1Batch}
        PredT = np.argmax(sess.run(prSoftMaxS, FeedDict))
        predAcc=np.max(sess.run(prSoftMaxS, FeedDict))
        return PredT,predAcc
    
imageList=glob.glob('../data/input/*')
imageList.sort()
frno = 2489
imageList=imageList[frno:]



for img in imageList:
    print('######## Frame No. : ',frno)
    firstImage=cv2.imread(img)

    # Perfrom thresholding for MSER and filter based on sign color
    
    rChannel=firstImage[:,:,2]
    gChannel=firstImage[:,:,1]
    bChannel=firstImage[:,:,0]
    rChannel=np.float32(rChannel)
    gChannel=np.float32(gChannel)
    bChannel=np.float32(bChannel)

    rChannel=(rChannel-np.min(rChannel))*255/(np.max(rChannel)-np.min(rChannel))
    gChannel=(gChannel-np.min(gChannel))*255/(np.max(gChannel)-np.min(gChannel))
    bChannel=(bChannel-np.min(bChannel))*255/(np.max(bChannel)-np.min(bChannel))


    compareArrayBlue=np.subtract(bChannel,rChannel)/(rChannel+gChannel+bChannel)
    compareArrayRed=np.minimum(np.subtract(rChannel,bChannel),np.subtract(rChannel,gChannel))/(rChannel+gChannel+bChannel)

    cDashBlue = np.maximum(0,compareArrayBlue)
    cDashRed = np.maximum(0,compareArrayRed)


    mask=np.ma.greater(cDashBlue,0.25)
    pos_NaNs = np.isnan(cDashBlue)
    cDashBlue[pos_NaNs] = 0
    cDashBlue[mask]=255
    cDashBlue[500:,:]=0    
    cDashBlue=np.uint8(cDashBlue)
    
    mask=np.ma.greater(cDashRed,0.26)
    pos_NaNs = np.isnan(cDashRed)
    cDashRed[pos_NaNs] = 0
    cDashRed[mask]=255
    cDashRed[500:,:]=0    
    cDashRed=np.uint8(cDashRed)
    
    # Use MSER for detection
    mser = cv2.MSER_create(_delta=10,_min_diversity = 0.2,_min_area=400,_max_area=3000,_max_variation=0.35)
    # bmser = cv2.MSER_create(10, 100, 1000, 0.5, 0.2, 200, 1.01, 0.003, 5)
    # rmser = cv2.MSER_create(10, 100, 1000, 0.5, 0.2, 200, 1.01, 0.003, 5)
    grayBlue =cDashBlue
    grayRed=cDashRed
    mod_img = firstImage.copy()
    regions, _ = mser.detectRegions(grayBlue)
    bbs=[]
    for i, region in enumerate(regions):
        # Apply filter on MSER blobs based on different signs
        (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))

        if(h>=0.75*w):
            if(h<2.5*w):
                buffer=5
                if(y-buffer>5 and x-buffer>5):
                    patch=mod_img[y-buffer:y+h+buffer,x-buffer:x+w+buffer,:]
                    patch=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)
                    pred,predAcc=Classify(patch)
                    # print(pred)
                    # print(predAcc)
                    acc=0.85
                    if(pred==45 and predAcc>acc):
                        sign=skimage.io.imread('../data/Training/00045/00318_00001.ppm')
                        sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                        sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                        mod_img[y-buffer:y+h+buffer,x-buffer-40:x+w+buffer-40,:]=sign
                        cv2.rectangle(mod_img,(x-buffer,y-buffer),(x+w+buffer,y+h+buffer),(0,0,255),2)
                        print('box')
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(mod_img,str(pred),(x+buffer,y+buffer), font, 2,(0,255,0),2,cv2.LINE_AA)


                    elif(pred==35 and predAcc>acc):
                        sign=skimage.io.imread('../data/Training/00035/00585_00000.ppm')
                        sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                        sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                        mod_img[y-buffer:y+h+buffer,x-buffer:x+w+buffer,:]=sign
                        cv2.rectangle(mod_img,(x-buffer,y-buffer),(x+w+buffer,y+h+buffer),(0,0,255),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(mod_img,str(pred),(x+buffer,y+buffer), font, 2,(0,255,0),2,cv2.LINE_AA)
                        print('box')

                    elif(pred==38 and predAcc>acc):
                        sign=skimage.io.imread('../data/Training/00038/00452_00002.ppm')
                        sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                        sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                        mod_img[y-buffer:y+h+buffer,x-buffer:x+w+buffer,:]=sign
                        cv2.rectangle(mod_img,(x-buffer,y-buffer),(x+w+buffer,y+h+buffer),(0,0,255),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(mod_img,str(pred),(x+buffer,y+buffer), font, 2,(0,255,0),2,cv2.LINE_AA)
                        print('box')

                        
    regions, _ = mser.detectRegions(grayRed)
    
    for i, region in enumerate(regions):
        (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
        
        if(h>=0.8*w):
            if(h<2.5*w):
                buffer=10
                if(y-buffer>10 and x-buffer>10):
                    patch=mod_img[y-buffer:y+h+buffer,x-buffer:x+w+buffer,:]
                    patch=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)
                    pred,predAcc=Classify(patch)
                    # print(pred)
                    # print(predAcc)
                    acc=0.8
                if(pred==21 and predAcc>acc):
                    sign=skimage.io.imread('../data/Training/00021/00715_00000.ppm')
                    sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                    sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                    if(x-40>10 and x+w-40>10 ):
                        mod_img[y-buffer:y+h+buffer,x-buffer:x+w+buffer,:]=sign
                        cv2.rectangle(mod_img,(x-buffer,y-buffer),(x+w+buffer,y+h+buffer),(0,0,255),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(mod_img,str(pred),(x+buffer,y+buffer), font, 2,(0,255,0),2,cv2.LINE_AA)
                        print('box')

                elif(pred==14 and predAcc>acc):
                    sign=skimage.io.imread('../data/Training/00014/00448_00001.ppm')
                    sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                    sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                    if(x-40>10 and x+w-40>10 ):
                        mod_img[y-buffer:y+h+buffer,x-buffer:x+w+buffer,:]=sign
                        cv2.rectangle(mod_img,(x-buffer,y-buffer),(x+w+buffer,y+h+buffer),(0,0,255),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(mod_img,str(pred),(x+buffer,y+buffer), font, 2,(0,255,0),2,cv2.LINE_AA)
                        print('box')

                elif(pred==1 and predAcc>acc):
                    sign=skimage.io.imread('../data/Training/00001/00025_00000.ppm')
                    sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                    sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                    if(x-40>10 and x+w-40>10 ):
                        mod_img[y-buffer:y+h+buffer,x-buffer:x+w+buffer,:]=sign
                        cv2.rectangle(mod_img,(x-buffer,y-buffer),(x+w+buffer,y+h+buffer),(0,0,255),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(mod_img,str(pred),(x+buffer,y+buffer), font, 2,(0,255,0),2,cv2.LINE_AA)
                        print('box')

                elif(pred==17 and predAcc>acc):
                    sign=skimage.io.imread('../data/Training/00017/00319_00002.ppm')
                    sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                    sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                    if(x-40>10 and x+w-40>10 ):
                        mod_img[y-buffer:y+h+buffer,x-buffer:x+w+buffer,:]=sign
                        cv2.rectangle(mod_img,(x-buffer,y-buffer),(x+w+buffer,y+h+buffer),(0,0,255),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(mod_img,str(pred),(x+buffer,y+buffer), font, 2,(0,255,0),2,cv2.LINE_AA)
                        print('box')

                elif(pred==19 and predAcc>acc):
                    sign=skimage.io.imread('../data/Training/00019/00066_00002.ppm')
                    sign = cv2.cvtColor(sign, cv2.COLOR_BGR2RGB)
                    sign=cv2.resize(sign,(patch.shape[1],patch.shape[0]))
                    if(x-40>10 and x+w-40>10 ):
                        mod_img[y-buffer:y+h+buffer,x-buffer:x+w+buffer,:]=sign
                        cv2.rectangle(mod_img,(x-buffer,y-buffer),(x+w+buffer,y+h+buffer),(0,0,255),2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(mod_img,str(pred),(x+buffer,y+buffer), font, 2,(0,255,0),2,cv2.LINE_AA)
                        print('box')

    
    mod_img = cv2.resize(mod_img,(480,320))
    cv2.imwrite('../output/frame' + str(frno) + '.png',mod_img)
    frno = frno + 1
