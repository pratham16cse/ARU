import os
import sys
import itertools
from multiprocessing import Pool
from collections import OrderedDict

dataset = sys.argv[1] # 'weeklySales', 'walmart'
model = sys.argv[2] # 'aru', 'adapt', 'baseline', 'hybrid'
configSize = sys.argv[3] # Model configuration to select from 'small', 'medium', and 'big'
nRuns = int(sys.argv[4]) # Number of runs.
coreList = sys.argv[5].split('/') # Feed core list to run, separated by '/'. eg. 0-1/2/3,4,5/6/7
experimentDir = sys.argv[6] # Name of experiment directory where outputs of this run will be stored.
print(coreList)

def getConfig(configSize):
    config = OrderedDict()
    if dataset in ['oneC']:
        config['-encoder_length'] = 8
        config['-decoder_length'] = 6
    if dataset in ['carparts']:
        config['-encoder_length'] = 12
        config['-decoder_length'] = 12
        config['-test_length'] = 12
        config['-val_length'] = 12
    if dataset in ['weeklySales']:
        config['-encoder_length'] = 8
        config['-decoder_length'] = 8
        config['-test_length'] = 8
        config['-val_length'] = 8
    if dataset in ['walmart']:
        config['-encoder_length'] = 8
        config['-decoder_length'] = 8
        config['-test_length'] = 8
        config['-val_length'] = 8
    if dataset in ['rossman']:
        config['-encoder_length'] = 16
        config['-decoder_length'] = 16
        config['-test_length'] = 16
        config['-val_length'] = 16
    if dataset in ['electricity']:
        config['-encoder_length'] = 168
        config['-decoder_length'] = 24
        config['-test_length'] = 24
        config['-val_length'] = 168
    if dataset in ['trafficPEMS']:
        config['-encoder_length'] = 336
        config['-decoder_length'] = 168
        config['-test_length'] = 168
        config['-val_length'] = 168
    if dataset in ['tourism']:
        config['-encoder_length'] = 48
        config['-decoder_length'] = 24
        config['-test_length'] = 24
        config['-val_length'] = 24
    if dataset in ['synthetic']:
        config['-encoder_length'] = 24
        config['-decoder_length'] = 24
        config['-test_length'] = 24
        config['-val_length'] = 24
        config['-synth_seq_len'] = int(sys.argv[7]) #Temporary argument for synthetic experiments
        config['-synth_num_ts'] = int(sys.argv[8]) #Temporary argument for synthetic experiments

    if model in ['snail']:
        if dataset in ['walmart']:
            config['-encoder_length'] = 50
        elif dataset in ['electricity']:
            config['-encoder_length'] = 672
        elif dataset in ['trafficPEMS']:
            config['-encoder_length'] = 672
        elif dataset in ['rossman']:
            config['-encoder_length'] = 365

    if configSize == 'small':
        config['-num_layers'] = 1
        config['-num_nodes'] = 8
        config['-lstm_dense_layer_size'] = 8
        config['-hidden1'] = 6
        config['-hidden2'] = 6
        if model in ['maruY','mEncAruY']:
            config['-numY'] = 4
    if configSize == 'medium':
        config['-num_layers'] = 1
        config['-num_nodes'] = 16
        config['-lstm_dense_layer_size'] = 16
        config['-hidden1'] = 15
        config['-hidden2'] = 10
        if model in ['maruY','mEncAruY']:
            config['-numY'] = 8
    if configSize == 'big':
        config['-num_layers'] = 1
        config['-num_nodes'] = 50
        config['-lstm_dense_layer_size'] = 32
        config['-hidden1'] = 20
        config['-hidden2'] = 15
        if model in ['maruY','mEncAruY']:
            config['-numY'] = 16


    config['-rnnType'] = 'cudnn_lstm'
    config['--is_probabilistic_pred'] = ''
    # config['--isTsId'] = ''
    config['--deep_ar_normalize_seq'] = ''

    # Set normalization flags and 
    # other dataset specific parameters.
    if dataset in ['carparts']:
        #config['-rnnType'] = 'cudnn_lstm'
        config['--isTsId'] = ''
        config['-num_iterations'] = 100
        config['-batch_size'] = 64
        if model in ['dec_aru', 'maruX', 'adapt']:
            config['--isAdapt'] = ''
            config['--fullLinear'] = ''
            if model in ['maruX']:
                config['-aruRegularizer'] = 0.05
                config['-numProjs'] = 0
            if model in ['dec_aru']:
                config['-aruRegularizer'] = 0.05
                config['-numProjs'] = 0
            if model in ['adapt']:
                config['-aruRegularizer'] = 0.01
                config['-numProjs'] = 0
            config['-aru_alpha'] = "[1.0]"
            #config['--aru_deepar_normalize'] = ''
        config['-valFrom'] = 'train'

    if dataset in ['rossman']:
        config['-lstm_dense_layer_size'] = 50
        #config['--isLogNormalised'] = ''
        #config['--isNormalised'] = ''
        #config['--deep_ar_normalize'] = ''
        config['-stepLen'] = 2
        config['-num_iterations'] = 50
        config['-batch_size'] = 128
        if model in ['dec_aru', 'maruX', 'localDecAru', 'adapt']:
            config['--isAdapt'] = ''
            config['--fullLinear'] = ''
            config['-numProjs'] = 0
            if model in ['dec_aru']:
                config['-aruRegularizer'] = 1.0
            elif model in ['maruX']:
                config['-aruRegularizer'] = 0.08
            if model in ['adapt']:
                config['-aruRegularizer'] = 0.01
                config['-numProjs'] = 0
            config['-aru_alpha'] = "[1.0]"
        config['-valFrom'] = 'train'

    if dataset in ['walmart', 'weeklySales', 'oneC']:
        if dataset in ['walmart']:
            config['-lstm_dense_layer_size'] = 50
        #config['--isNormalised'] = ''
        #config['--deep_ar_normalize'] = ''
        #config['--deep_ar_normalize_seq'] = ''
        config['-num_iterations'] = 50
        config['-batch_size'] = 64
        if model in ['dec_aru', 'maruX', 'adapt']:
            config['--isAdapt'] = ''
            config['--fullLinear'] = ''
            if model in ['maruX']:
                config['-aruRegularizer'] = 0.05
                config['-numProjs'] = 0
            if model in ['dec_aru']:
                config['-aruRegularizer'] = 0.05
                config['-numProjs'] = 0
            if model in ['adapt']:
                config['-aruRegularizer'] = 0.01
                config['-numProjs'] = 0
            config['-aru_alpha'] = "[1.0]"
            #config['--aru_deepar_normalize'] = ''
        config['-valFrom'] = 'train'

    if dataset in ['electricity']:
        # config['--isPrependZeros'] = ''
        config['--isLagFeat'] = ''
        config['-stepLen'] = 24
        #config['-learning_rate'] = 0.00005
        #config['--isNormalised'] = ''
        #config['--deep_ar_normalize'] = ''
        config['-num_iterations'] = 40
        config['-batch_size'] = 64
        if model in ['dec_aru', 'maruX', 'localDecAru', 'adapt']:
            config['--isAdapt'] = ''
            config['--fullLinear'] = ''
            config['-aruRegularizer'] = 2.0
            if model in ['maruX']:
                config['-aruRegularizer'] = 3.0
            if model in ['adapt']:
                config['-aruRegularizer'] = 0.01
                config['-numProjs'] = 0
            config['-numProjs'] = 0
            config['-aru_alpha'] = "[1.0]"
            #config['--aru_deepar_normalize'] = ''
        config['-valFrom'] = 'train'

    if dataset in ['trafficPEMS']:
        config['--isLagFeat'] = ''
        config['--isTsId'] = ''
        #config['--isNormalised'] = ''
        #config['--deep_ar_normalize'] = ''
        config['-stepLen'] = 24
        config['-num_iterations'] = 50
        config['-batch_size'] = 64
        if model in ['dec_aru', 'maruX', 'adapt']:
            config['--isAdapt'] = ''
            config['--fullLinear'] = ''
            config['-aruRegularizer'] = 2.0
            if model in ['maruX']:
                config['-aruRegularizer'] = 1.5
            if model in ['adapt']:
                config['-aruRegularizer'] = 0.01
                config['-numProjs'] = 0
            config['-numProjs'] = 0
            config['-aru_alpha'] = "[1.0,0.99]"
            #config['--aru_deepar_normalize'] = ''
        config['-valFrom'] = 'train'

    if dataset in ['tourism']:
        config['--isLagFeat'] = ''
        #config['--isNormalised'] = ''
        #config['--deep_ar_normalize'] = ''
        #config['-stepLen'] = 24
        config['-num_iterations'] = 70
        config['-batch_size'] = 64
        if model in ['dec_aru', 'maruX', 'adapt']:
            config['--isAdapt'] = ''
            config['--fullLinear'] = ''
            config['-aruRegularizer'] = 0.01
            if model in ['maruX']:
                config['-aruRegularizer'] = 0.01
            if model in ['adapt']:
                config['-aruRegularizer'] = 0.01
                config['-numProjs'] = 0
            config['-numProjs'] = 0
            config['-aru_alpha'] = "[1.0]"
            #config['--aru_deepar_normalize'] = ''
        config['-valFrom'] = 'train'


    if dataset in ['synthetic']:
        config.pop('--is_probabilistic_pred', None)
        config.pop('--deep_ar_normalize_seq', None)
        config['-rnnType'] = 'lstm'
        #config['--isLagFeat'] = ''
        #config['--isNormalised'] = ''
        #config['--deep_ar_normalize'] = ''
        #config['-stepLen'] = 24
        #config['--isTsId'] = ''
        config['-num_iterations'] = 80
        config['-batch_size'] = 64
        if model in ['dec_aru', 'maruX', 'neuralARU', 'enc_aru']:
            config['--isAdapt'] = ''
            config['--fullLinear'] = ''
            config['-aruRegularizer'] = 0.01
            if model in ['maruX']:
                config['-aruRegularizer'] = 0.01
            config['-numProjs'] = 0
            config['-aru_alpha'] = "[1.0]"
            #config['--aru_deepar_normalize'] = ''
        config['-valFrom'] = 'train'


    if model in ['localDecAru']:
        config['-num_iterations'] = 1

    config['--verbose'] = ''

    return config
modelConfig = getConfig(configSize)


seedList = [1,12,123,1234,12345]
#seedList = [100,200,300,400,500]
runIDList = range(nRuns)

def runCommand(command):
    os.system(command)

outputDirName = os.path.join('Outputs', experimentDir, dataset+'_'+model+'_'+configSize)
allDirs = [os.path.join('Outputs', experimentDir, i)
           for i in os.listdir(os.path.join('Outputs', experimentDir))
           if os.path.isdir(os.path.join('Outputs', experimentDir, i))]
if allDirs:
    matches = [i for i in allDirs if i.startswith(outputDirName)]
    if matches:
        ids = [int(i.split('_')[-1]) for i in matches]
        newId = max(ids) + 1
    else:
        newId = 1
else:
    newId = 1
outputDirName = outputDirName + '_' + str(newId)
if not os.path.exists(outputDirName):
    os.mkdir(outputDirName)
commandLineInput = '-dataset '+dataset+' -modelToRun '+model+' '
for param, val in modelConfig.items():
    commandLineInput += ' '+param+' '+str(val)
pool = Pool(nRuns)
commandList = list()
command = 'python run.py '+commandLineInput
for r in range(1, nRuns+1):
    seed = str(seedList[r-1])
    outputFileName = os.path.join(outputDirName, 'seed_'+seed)
    gtVsPredOutputFile = os.path.join(outputDirName, 'seed_'+seed+'gtVsPred')
    commandList.append('taskset -c '+str(coreList[r-1])+' '+command+' -seed '+seed+' -gtVsPredOutputFile '+gtVsPredOutputFile+' >>'+outputFileName)
    with open(outputFileName,'a') as outFile:
        outFile.write(commandLineInput+'\n')
#print(commandList)
pool.map(runCommand, commandList)
