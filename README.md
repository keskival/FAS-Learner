# FAS-Learner
An LSTM network which learns the patterns in a simulated FAS system from log messages.

## Output for th learner.lua --hiddenSize "{40}"
	{
	   batchSize : 1479
	   cutOffNorm : -1
	   decayFactor : 0.001
	   hiddenSize : "{40}"
	   learningRate : 0.1
	   lrDecay : "linear"
	   maxOutNorm : 2
	   maxWait : 4
	   minLR : 1e-05
	   momentum : 0
	   saturateEpoch : 300
	   schedule : "{}"
	   seed : 1
	   trainFile : "train.json"
	   uniform : 0.8
	   useDevice : 1
	   validationFile : "validation.json"
	}
	Input read. Generating the neural network initial state. InputSize:   35
	nn.Sequential {
	  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
	  (1): nn.SplitTable
	  (2): nn.Sequencer @ nn.FastLSTM
	  (3): nn.Sequencer @ nn.Sequential {
	    [input -> (1) -> (2) -> output]
	    (1): nn.Linear(40 -> 35)
	    (2): nn.LogSoftMax
	  }
	  (4): nn.JoinTable
	  (5): nn.Reshape(1479x35)
	}
	Converting to CUDA...
	Starting experiment...
	==> epoch # 1 for optimizer :
	==> example speed = 75.040366272555 examples/s
	Bern:1444319648:1:optimizer:loss avgErr 0.0027926617816463
	Bern:1444319648:1:optimizer:confusion accuracy = 0.018931710615281
	Bern:1444319648:1:validator:confusion accuracy = 0.025016903313049
	==> epoch # 2 for optimizer :
	learningRate  0.0996667
	==> example speed = 1164.4917443041 examples/s
	Bern:1444319648:1:optimizer:loss avgErr 0.0025978151247903
	Bern:1444319648:1:optimizer:confusion accuracy = 0.021636240703178
	Bern:1444319648:1:validator:confusion accuracy = 0.027721433400947
	...
	==> epoch # 43 for optimizer :
	learningRate  0.0860014
	==> example speed = 1229.158644199 examples/s
	Bern:1444319648:1:optimizer:loss avgErr 0.0024604487854375
	Bern:1444319648:1:optimizer:confusion accuracy = 0.035158891142664
	Bern:1444319648:1:validator:confusion accuracy = 0.028397565922921
	found maxima : 0.037863421230561 at epoch 12
	Done.

## Output for th learner.lua --hiddenSize "{40,40}"
	...
	Input read. Generating the neural network initial state. InputSize:   35
	nn.Sequential {
	  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
	  (1): nn.SplitTable
	  (2): nn.Sequencer @ nn.FastLSTM
	  (3): nn.Sequencer @ nn.FastLSTM
	  (4): nn.Sequencer @ nn.Sequential {
	    [input -> (1) -> (2) -> output]
	    (1): nn.Linear(40 -> 35)
	    (2): nn.LogSoftMax
	  }
	  (5): nn.JoinTable
	  (6): nn.Reshape(1479x35)
	}
	Converting to CUDA...
	Starting experiment...
	==> epoch # 1 for optimizer :
	==> example speed = 39.910730551057 examples/s
	Bern:1444320073:1:optimizer:loss avgErr 0.0028547237981728
	Bern:1444320073:1:optimizer:confusion accuracy = 0.035158891142664
	Bern:1444320073:1:validator:confusion accuracy = 0.021636240703178
	==> epoch # 2 for optimizer :
	learningRate  0.0996667
	==> example speed = 658.28359534151 examples/s
	Bern:1444320073:1:optimizer:loss avgErr 0.0025920977537659
	Bern:1444320073:1:optimizer:confusion accuracy = 0.027721433400947
	Bern:1444320073:1:validator:confusion accuracy = 0.020283975659229
	...
	==> epoch # 156 for optimizer :
	learningRate  0.048338500000001
	==> example speed = 644.68124100535 examples/s
	Bern:1444320073:1:optimizer:loss avgErr 0.002387963250491
	Bern:1444320073:1:optimizer:confusion accuracy = 0.066260987153482
	Bern:1444320073:1:validator:confusion accuracy = 0.057471264367816
	found maxima : 0.059499661933739 at epoch 125
	Done.

## Output for th learner.lua --hiddenSize "{40,40,40}"
	Input read. Generating the neural network initial state. InputSize:   35
	nn.Sequential {
	  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> output]
	  (1): nn.SplitTable
	  (2): nn.Sequencer @ nn.FastLSTM
	  (3): nn.Sequencer @ nn.FastLSTM
	  (4): nn.Sequencer @ nn.FastLSTM
	  (5): nn.Sequencer @ nn.Sequential {
	    [input -> (1) -> (2) -> output]
	    (1): nn.Linear(40 -> 35)
	    (2): nn.LogSoftMax
	  }
	  (6): nn.JoinTable
	  (7): nn.Reshape(1479x35)
	}
	Converting to CUDA...
	Starting experiment...
	==> epoch # 1 for optimizer :
	==> example speed = 26.203059182225 examples/s
	Bern:1444322943:1:optimizer:loss avgErr 0.0030859736996458
	Bern:1444322943:1:optimizer:confusion accuracy = 0.031102096010818
	Bern:1444322943:1:validator:confusion accuracy = 0.020960108181204
	==> epoch # 2 for optimizer :
	learningRate  0.0996667
	==> example speed = 425.09564011086 examples/s
	Bern:1444322943:1:optimizer:loss avgErr 0.0026145095838092
	Bern:1444322943:1:optimizer:confusion accuracy = 0.016903313049358
	Bern:1444322943:1:validator:confusion accuracy = 0.025693035835024
	...
	==> epoch # 91 for optimizer :
	learningRate  0.070003
	==> example speed = 437.68571172151 examples/s
	Bern:1444322943:1:optimizer:loss avgErr 0.0023859672404528
	Bern:1444322943:1:optimizer:confusion accuracy = 0.05341446923597
	Bern:1444322943:1:validator:confusion accuracy = 0.059499661933739
	found maxima : 0.060851926977688 at epoch 60
	Done.
	
