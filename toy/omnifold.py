from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
import sys, os
import horovod.tensorflow.keras as hvd
import horovod.tensorflow
import json, yaml
from datetime import datetime

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss
    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    return K.mean(t_loss)

def LoadJson(file_name):
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))


class Multifold():
    def __init__(self,nevts,version,config_file='config_omnifold.json',verbose=1):
        self.nevts = nevts
        self.opt = LoadJson(config_file)
        self.niter = self.opt['General']['NITER']
        self.ntrial=None
        self.version=version
        self.mc_gen = None
        self.mc_reco = None
        self.data=None

        self.weights_folder = '../weights'
        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)
            
    def Unfold(self):
        self.BATCH_SIZE=self.opt['General']['BATCH_SIZE']
        self.EPOCHS=self.opt['General']['EPOCHS']
        self.CompileModel(float(self.opt['General']['LR']))
                                        
        self.weights_pull = np.ones(self.weights_mc.shape[0])
        self.weights_push = np.ones(self.weights_mc.shape[0])
        
        for i in range(self.niter):
            print("ITERATION: {}".format(i + 1))            
            self.RunStep1(i)        
            self.RunStep2(i)            

    def RunStep1(self,i):
        '''Data versus reco MC reweighting'''
        print("RUNNING STEP 1")

        self.RunModel(
            np.concatenate((self.mc_reco, self.data)),
            np.concatenate((self.labels_mc, self.labels_data)),
            np.concatenate((self.weights_push*self.weights_mc,self.weights_data )),
            i,self.model1,stepn=1,
        )
        

        new_weights=self.reweight(self.mc_reco,self.model1)            
        new_weights[self.not_pass_reco]=1.0
        self.weights_pull = self.weights_push *new_weights
        self.weights_pull = self.weights_pull/np.average(self.weights_pull)

    def RunStep2(self,i):
        '''Gen to Gen reweighing'''        
        print("RUNNING STEP 2")
            
        self.RunModel(
            np.concatenate((self.mc_gen, self.mc_gen)),
            np.concatenate((self.labels_mc, self.labels_gen)),
            np.concatenate((self.weights_mc, self.weights_mc*self.weights_pull)),
            i,self.model2,stepn=2,
        )

        new_weights=self.reweight(self.mc_gen,self.model2)
        new_weights[self.not_pass_gen]=1.0
        self.weights_push = new_weights
        self.weights_push = self.weights_push/np.average(self.weights_push)

    def RunModel(self,sample,labels,weights,iteration,model,stepn):
        
        mask = sample[:,0]!=-10        
        data = tf.data.Dataset.from_tensor_slices((
            sample[mask],
            np.stack((labels[mask],weights[mask]),axis=1))
        ).cache().shuffle(np.sum(mask))

        #Fix same number of training events between ranks
        NTRAIN,NTEST = self.GetNtrainNtest(stepn)        
        test_data = data.take(NTEST).repeat().batch(self.BATCH_SIZE)
        train_data = data.skip(NTEST).repeat().batch(self.BATCH_SIZE)

        verbose = 1 if hvd.rank() == 0 else 0
        
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            hvd.callbacks.LearningRateWarmupCallback(
                initial_lr=self.hvd_lr, warmup_epochs=self.opt['General']['NWARMUP'],
                verbose=verbose),
            ReduceLROnPlateau(patience=8, min_lr=1e-7,verbose=verbose),
            EarlyStopping(patience=self.opt['General']['NPATIENCE'],restore_best_weights=True)
        ]
        
        base_name = "Omnifold_toy{}".format(self.ntrial)
        
        if hvd.rank() ==0:
            callbacks.append(
                ModelCheckpoint('{}/{}_{}_iter{}_step{}.h5'.format(self.weights_folder,base_name,self.version,iteration,stepn),
                                save_best_only=True,mode='auto',period=1,save_weights_only=True))
            
        _ =  model.fit(
            train_data,
            epochs=self.EPOCHS,
            steps_per_epoch=int(NTRAIN/self.BATCH_SIZE),
            validation_data=test_data,
            validation_steps=int(NTEST/self.BATCH_SIZE),
            verbose=verbose,
            callbacks=callbacks)




    def Preprocessing(self,weights_mc=None,weights_data=None):
        self.PrepareWeights(weights_mc,weights_data)
        self.PrepareInputs()
        self.PrepareModel()

    def PrepareWeights(self,weights_mc,weights_data):
        
        self.not_pass_reco = self.mc_reco[:,0]==-10
        self.not_pass_gen = self.mc_gen[:,0]==-10

        
        if weights_mc is None:
            self.weights_mc = np.ones(self.mc_reco.shape[0])
        else:
            self.weights_mc = weights_mc

        if weights_data is None:
            self.weights_data = np.ones(self.data.shape[0])
        else:
            self.weights_data =weights_data


    def CompileModel(self,lr):
        self.hvd_lr = lr*np.sqrt(hvd.size())
        opt = tensorflow.keras.optimizers.Adam(learning_rate=self.hvd_lr)
        opt = hvd.DistributedOptimizer(
            opt, average_aggregated_gradients=True)

        self.model1.compile(loss=weighted_binary_crossentropy,
                            optimizer=opt,experimental_run_tf_function=False)

        self.model2.compile(loss=weighted_binary_crossentropy,
                            optimizer=opt,experimental_run_tf_function=False)


    def PrepareInputs(self):
        self.labels_mc = np.zeros(len(self.mc_reco))
        self.labels_data = np.ones(len(self.data))
        self.labels_gen = np.ones(len(self.mc_gen))


    def PrepareModel(self):
                        
        nvars = self.mc_gen.shape[1]
        inputs1,outputs1 = MLP(nvars)
        inputs2,outputs2 = MLP(nvars)
                                   
        self.model1 = Model(inputs=inputs1, outputs=outputs1)
        self.model2 = Model(inputs=inputs2, outputs=outputs2)


    def GetNtrainNtest(self,stepn):
        NTRAIN=int(0.8*self.nevts/hvd.size())
        NTEST=int(0.2*self.nevts/hvd.size())                        
        return NTRAIN,NTEST

    def reweight(self,events,model):
        f = np.nan_to_num(model.predict(events, batch_size=10000),posinf=1,neginf=0)
        weights = f / (1. - f)
        #weights = np.clip(weights,0,10)
        weights = weights[:,0]
        return np.squeeze(np.nan_to_num(weights,posinf=1))



def MLP(nvars):
    ''' Define a simple fully conneted model to be used during unfolding'''
    inputs = Input((nvars, ))
    layer = Dense(8,activation='relu')(inputs)
    layer = Dense(16, activation='relu')(layer)
    layer = Dense(8,activation='relu')(layer)    
    outputs = Dense(1,activation='sigmoid')(layer)
    return inputs,outputs
