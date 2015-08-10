
from chainer import cuda
if cuda.available:
    cuda.init()
    
from chainerLayers.cukernels import activations
reLU = activations.reLU
dreLU = activations.dreLU
leakyReLU = activations.leakyReLU
dleakyReLU = activations.dleakyReLU
sigmoid = activations.sigmoid
dsigmoid = activations.dsigmoid
tanh = activations.tanh
dtanh = activations.dtanh

from chainerLayers.cukernels import additions
addVec2Mat = additions.addVec2Mat
matAdd = additions.matAdd
vecAdd = additions.vecAdd

from chainerLayers.cukernels import dots
dot = dots.dot
dotAdd = dots.dotAdd

from chainerLayers.cukernels import hotFunctions
hotdot = hotFunctions.hotdot
dothot = hotFunctions.dothot

from chainerLayers.cukernels import misc
hadamard = misc.hadamard
getByIndex_LogAndClip = misc.getByIndex_LogAndClip
dSoftmaxCrossEntropy = misc.dSoftmaxCrossEntropy
dropout = misc.dropout

from chainerLayers.cukernels import lstm
lstm_forward = lstm.lstm_forward
lstm_backward = lstm.lstm_backward

from chainerLayers.cukernels import gru
gru_forward = gru.gru_forward
gru_backward = gru.gru_backward