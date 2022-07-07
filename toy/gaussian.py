import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import utils
from omnifold import  Multifold,LoadJson
import tensorflow.keras.backend as K

utils.SetStyle()

hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


parser = argparse.ArgumentParser()

parser.add_argument('--ndim', type=int,default=2, help='Number of dimensions used in the generated data')
parser.add_argument('--config', default='config_omnifold.json', help='Basic config file containing general options')
parser.add_argument('--nevts', type=float,default=1e6, help='Dataset size to use during training')
parser.add_argument('--ntrial', type=int,default=10, help='Number of independent omnifold trials')

flags = parser.parse_args()
nevts=int(flags.nevts)




data_gen = utils.Generator(nevts,flags.ndim,mean=0.2)
utils.Plot_2D(data_gen,'data_gen')
mc_gen = utils.Generator(nevts,flags.ndim)
utils.Plot_2D(mc_gen,'mc_gen')
data_reco = utils.Detector(data_gen)
utils.Plot_2D(data_reco,'data_reco')
#data only contains events that pass reconstruction
data_reco=data_reco[data_reco[:,0]!=-10]
mc_reco = utils.Detector(mc_gen)
utils.Plot_2D(mc_reco,'mc_reco')

feed_dict={
    'data reco':data_reco[:,0],
    'data gen':data_gen[:,0],
    'mc reco':mc_reco[:,0],
    'mc gen':mc_gen[:,0],
}

fig,ax = utils.HistRoutine(feed_dict,plot_ratio=False,
                           binning=np.linspace(-2,2,50),
                           xlabel='feature 1',
                           ylabel='Normalized events',
                           reference_name='data gen')
plot_folder='../plots'
fig.savefig('{}/{}.pdf'.format(plot_folder,"Hist1D_feature0"))


mfold = Multifold(nevts=nevts,version='null_toy')
mfold.mc_gen = mc_gen
mfold.mc_reco = mc_reco
mfold.data = data_reco

hist_trials=[]


for itrial in range(flags.ntrial):
    K.clear_session()
    mfold.ntrial=itrial
    mfold.Preprocessing()
    mfold.Unfold()

    weights_step2 = mfold.reweight(mfold.mc_gen,mfold.model2)

    if itrial==0:
        weight_dict = {
            'data gen':np.ones(data_gen.shape[0]),
            'mc gen': weights_step2
        }

        feed_dict = {
            'data gen':data_gen[:,0],
            'mc gen':mc_gen[:,0],
        }

        fig,ax = utils.HistRoutine(feed_dict,plot_ratio=True,
                                   weights=weight_dict,
                                   binning=np.linspace(-2,2,50),
                                   xlabel='feature 1',
                                   ylabel='Normalized events',
                                   reference_name='data gen')
        
        plot_folder='../plots'
        fig.savefig('{}/{}.pdf'.format(plot_folder,"Hist1D_feature0_unfold"))

    hist_trials.append(np.histogram2d(mc_gen[:,0],mc_gen[:,1],
                                      bins = 50,
                                      range=[[-2,2],[-2,2]],weights=weights_step2)[0])


hist_std = np.std(hist_trials,axis=0)/np.mean(hist_trials,axis=0)
utils.Plot_2D(hist_std,'std',use_hist=False)
