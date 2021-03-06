from collections import OrderedDict
import numpy as np
import random

class BatchGenerator():
    def __init__(self, iterable, batchSeqLen=16, numStates=0, stateSizes=None, shuffle_batches=False, seed=None):
        self.iterable = iterable
        #if self.iterable!=None:
        self.batchSeqLen = batchSeqLen
        # self.stepLen = stepLen
        keys = range(len(self.iterable))
        self.seed = seed
        self.shuffle_batches = shuffle_batches
        if self.shuffle_batches:
            random.Random(self.seed).shuffle(keys)
            self.seed = self.seed + 1
        self.seqLengths = OrderedDict()
        self.cursorDict = OrderedDict()
        for i in keys:
            self.seqLengths[i] = len(self.iterable[i])
            self.cursorDict[i] = 0
        self.iterFinished = 0

        self.numStates = numStates
        self.stateSizes = stateSizes
        if self.numStates:
            assert self.stateSizes != None
            self.states = list()
            for i in range(self.numStates):
                self.states.append(np.zeros(([len(self.iterable)]+self.stateSizes[i])))

    def reset(self):
        keys = range(len(self.iterable))
        if self.shuffle_batches:
            random.Random(self.seed).shuffle(keys)
            self.seed = self.seed + 1
        self.seqLengths = OrderedDict()
        self.cursorDict = OrderedDict()
        for i in keys:
            self.seqLengths[i] = len(self.iterable[i])
            self.cursorDict[i] = 0

        #self.iterFinished += 1
        if self.numStates:
            for i in range(self.numStates):
                self.states[i] = np.zeros(([len(self.iterable)]+self.stateSizes[i]))

    def nextBatch(self, batchSize=1, stepLen=1):

        tsIndices = self.cursorDict.keys()[:batchSize]
        startIndices = [self.cursorDict[i] for i in tsIndices]
        seqLens = [self.seqLengths[i] for i in tsIndices]
        batch = list()
        startingTs = list()
        finishedSequences = list()
        for tsInd, startInd, seqLen in zip(tsIndices, startIndices, seqLens):

            batch.append(self.iterable[tsInd][startInd:startInd+self.batchSeqLen])

            if startInd == 0:
                startingTs.append(tsInd)
    
            if startInd+stepLen+self.batchSeqLen > seqLen:
                finishedSequences.append(tsInd)
            else:
                self.cursorDict[tsInd] += stepLen

        endingTs = finishedSequences

        for index in finishedSequences:
            del self.cursorDict[index]
            del self.seqLengths[index]

        batch = np.array(batch)
        mask = np.ones((batchSize, 1))
        if batch.shape[0] < batchSize:
            mask[batch.shape[0]:] = 0
            tsIndices = tsIndices + [0] * (batchSize-batch.shape[0])
            startingTs = startingTs + [0] * (batchSize-batch.shape[0])
            endingTs = endingTs + [0] * (batchSize-batch.shape[0])
            batch = np.concatenate([batch, np.zeros(([batchSize-batch.shape[0]]+list(batch.shape[1:])))], axis=0)

        states = list()
        for i in range(self.numStates):
            state = list()
            for tsInd in tsIndices:
                state.append(self.states[i][tsInd])
            states.append(state)

        if not self.cursorDict:
            self.iterFinished += 1
            #self.reset()

        return batch, tsIndices, states, startingTs, endingTs, mask

    def updateStates(newStates, tsIndices):
        for i in self.numStates:
            self.states[i][tsIndices,:,:] = newStates[i]


if __name__ == "__main__":
    iterable = [[[11],[12],[13],[14],[15],[16]],
                [[21],[22],[23],[24]],
                [[31],[32],[33]],
                [[41],[42],[43],[44],[45],[46],[47],[48],[49],[50]]]

    seed = 6
    itr = BatchGenerator(iterable, batchSeqLen=3, numStates=2, stateSizes=[[1,2],[2,3]], shuffle_batches=True, seed=seed)
    print(itr.cursorDict.items())
    print(itr.seqLengths.items())
    print('----------')
    print(itr.iterable)
    print('----------')
    print(itr.states)
    while itr.iterFinished < 5:
        currItr = itr.iterFinished
        while currItr == itr.iterFinished:
            batch, tsIndices, states, startingTs, endingTs = itr.nextBatch(batchSize=2, stepLen=1)
            print('Batch:', batch)
            print('tsIndices:', tsIndices)
            #for i, state in enumerate(states):
            #    print('State '+str(i)+':')
            #    print(state)
            print('StartingTs:', startingTs)
            print('EndingTs:', endingTs)
            print(currItr,itr.iterFinished)
            print('-------')
        itr.reset()
        print('----Batch {} finished'.format(currItr))
