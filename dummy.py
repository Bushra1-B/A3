import random
import numpy as np
import torch as tr

seed = 4313149                      # <<<<<<<<<<<<<<<< Your UPM ID Goes Here
random.seed(seed)
np.random.seed(seed)
tr.manual_seed(seed)

class SimpleFeedForwardNet(tr.nn.Module):

  def __init__(self):
    super().__init__() 
    self.linear1 = tr.nn.Linear(784, 1024, bias=True)       # 784 -> 16 - Layer 1 -- Affine Transformation (Linear with Bias)
    self.linear2 = tr.nn.Linear(1024, 512, bias=True)       # 16 -> 16  - Layer 2 -- Linear Transformation (no bias)
    self.linear3 = tr.nn.Linear(512, 256, bias=True)        # 16 -> 10  - Layer 3 -- Affine Transformation (Linear with Bias)
    self.linear4 = tr.nn.Linear(256, 128, bias=True)    #changes || add layers 5 of them and number of width DONE !
    self.linear5 = tr.nn.Linear(128, 10, bias=True)
    self.init_weights()

  def init_weights(self):
    tr.nn.init.zeros_(self.linear1.bias) # all of them is bias give me an 56% - 70% DONE ! task-4
    tr.nn.init.zeros_(self.linear2.bias)  
    tr.nn.init.zeros_(self.linear3.bias) 
    tr.nn.init.ones_(self.linear4.bias)
    tr.nn.init.ones_(self.linear5.bias) 
  
  def forward(self, x):
    x = self.linear1(x)
    x = self.linear2(x)
    x = self.linear3(x)
    x = self.linear4(x)
    x = self.linear5(x)
    return x

model = SimpleFeedForwardNet()                                          # Architecture
optimizer = tr.optim.SGD(model.parameters(), lr=0.001,maximize=True)   # Optimizer 


loss_fn = tr.nn.CrossEntropyLoss()                                      # Objective Function [DO NOT CHANGE !]
