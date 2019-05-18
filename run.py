#from __future__ import print_function

import sys
import json
import argparse
import pickle
import time

from carpartsUtil import *
from model_utils import *
import reader


def main(params=None):
    params.sequence_length = params.encoder_length + params.decoder_length

    XXTrain, lagXXTrain, YYTrain, \
    XXValidation, lagXXValidation, YYValidation, \
    XXTest, lagXXTest, YYTest, \
    maxYY, avgYY, val_indices, feature_dict, emb_dict \
    = reader.readFromPickle(params)

    if params.modelToRun in ['aru', 'adapt', 'maruY', 'maruX', 'enc_aru', 'dec_aru', 'maruX_sruAugment', 'maruX_rnnAugment',
                             'mEncAruY', 'mEncAruX_rnnAugment', 'mEncAruX_sruAugment', 'mEncAruX', 'localEncAru', 'localDecAru']:
        params.isAdapt = True
    elif params.modelToRun in ['baseline', 'hybrid', 'snail', 'attention', 'neuralARU']:
        params.isAdapt = False

    if params.modelToRun in ['aru','adapt','baseline','hybrid','maruY','maruX','snail','attention',
                             'enc_aru','dec_aru','maruX_sruAugment','maruX_rnnAugment',
                             'mEncAruY','mEncAruX_rnnAugment','mEncAruX_sruAugment','mEncAruX',
                             'localEncAru', 'localDecAru', 'neuralARU']:
        YYTestGt, YYTestPred, SSTestPred = runModel(params,
                                                    XXTrain, lagXXTrain, YYTrain,
                                                    XXValidation, lagXXValidation, YYValidation,
                                                    XXTest, lagXXTest, YYTest,
                                                    maxYY, avgYY, val_indices, feature_dict, emb_dict)
        with open(params.gtVsPredOutputFile, 'wb') as f:
            pickle.dump(YYTestGt, f)
            pickle.dump(YYTestPred, f)
            pickle.dump(SSTestPred, f)
    elif params.modelToRun == "hierarchical":
    	runHierarchicalModel(XXTrain,YYTrain,PrevYYTrain,XXTest,YYTest,PrevYYTest,
    		count,params.encoder_length,params.decoder_length,sequence_length,params.testFraction,isEmbedding,isNormalised,maxYY,params.rnnType,params.isSRUFeat,isWalmartData,numFW,deep_ar_normalize,isLogNormalised)
#    elif params.modelToRun == "attention":
#    	runAttentionModel(XXTrain,YYTrain,PrevYYTrain,XXTest,YYTest,PrevYYTest,
#    		count,params.encoder_length,params.decoder_length,sequence_length,params.testFraction,isEmbedding,isNormalised,maxYY,params.rnnType,params.isSRUFeat,isWalmartData,numFW,deep_ar_normalize,isLogNormalised)

def setup_parser(arguments, title):

    parser = argparse.ArgumentParser(description=title,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    str2Types = {'int':int, 'float':float, 'bool':bool, 'str':str}
    for key, val in arguments.items():
        if val["type"] == 'bool':
            parser.add_argument('--%s' % key,
                    help=val["help"],
                    action='store_true',
                    default=str2Types[val["type"]](val["default"]))
        else:
            parser.add_argument('-%s' % key,
                    #type=eval(val["type"]),
                    #type=type(val["default"]),
                    type=str2Types[val["type"]],
                    help=val["help"],
                    default=str2Types[val["type"]](val["default"]))
    return parser

def read_params(parser):

    parameters = parser.parse_args()
    parameters.aru_alpha =  map(float, parameters.aru_alpha.strip('[]').split(',')) # convert aru_alpha from string to list
    #print(parameters.aru_alpha, type(parameters.aru_alpha))
    return parameters

def get_parameters(title=None):

    with open("params.json") as data_file:
        data = json.load(data_file)
    parser = setup_parser(data, title)
    parameters = read_params(parser)

    return parameters

if __name__ == "__main__":
    params = get_parameters()
    #print(params)
    main(params)
