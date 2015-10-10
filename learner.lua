require 'cutorch'
require 'rnn'
require 'dp'
require 'cunnx'
require 'nn'
require 'nngraph'

nngraph.setDebug(true)

-- First we initialize the neural network parameters --

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a system to model a Flexible Assembly System')
cmd:option('--useDevice', 1, '')
cmd:option('--learningRate', 0.1, '')
cmd:option('--uniform', 0.8, 'The initial values are taken from a uniform distribution between negative and positive given value.')
cmd:option('--lrDecay', 'linear', '')
cmd:option('--minLR', 0.00001, '')
cmd:option('--saturateEpoch', 300, '')
-- Note that if you change batchSize, you should also change the Reshape layer
cmd:option('--batchSize', 1479, '')
cmd:option('--schedule', '{}', '')
cmd:option('--maxWait', 4, '')
cmd:option('--decayFactor', 0.001, '')
cmd:option('--momentum', 0, '')
cmd:option('--maxOutNorm', 2, '')
cmd:option('--cutOffNorm', -1, '')
cmd:option('--trainFile', 'train.json', '')
cmd:option('--validationFile', 'validation.json', '')
cmd:option('--hiddenSize', '{40}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, LSTMs are stacked')
cmd:option('--seed', 1, '')
cmd:text()

opt = cmd:parse(arg or {})
table.print(opt)

loadstring("opt.hiddenSize = "..opt.hiddenSize)()

cutorch.setDevice(opt.useDevice)
cutorch.manualSeed(opt.seed)

-- Loading the input file --
local json = require "json"
local inputfile = assert(io.open(opt.trainFile, "r"))
local training = inputfile:read "*a"
inputfile:close()

local validationfile = assert(io.open(opt.validationFile, "r"))
local validation = validationfile:read "*a"
validationfile:close()

local decodedTraining = json.decode(training)
local decodedValidation = json.decode(validation)

-- A map from event type string to event id number --
local eventIds = {}
-- A map from event id number to event type string --
local eventTypes = {}

local nTraining = #decodedTraining
local nValidation = #decodedValidation

-- Generating the maps to encode the event types for the neural network --
local nextEventId = 0
local classes = {}

for i, event in ipairs(decodedTraining) do
  if (eventIds[event.type]) then
    event.eventId = eventIds[event.type]
  else
    event.eventId = nextEventId
    eventIds[event.type] = nextEventId
    eventTypes[nextEventId] = event.type
    nextEventId = nextEventId + 1
    classes[nextEventId] = nextEventId
  end
end
for i, event in ipairs(decodedValidation) do
  if (eventIds[event.type]) then
    event.eventId = eventIds[event.type]
  else
    event.eventId = nextEventId
    eventIds[event.type] = nextEventId
    eventTypes[nextEventId] = event.type
    nextEventId = nextEventId + 1
  end
end

function oneHot(index, inputSize)
  local oneHotTensor = torch.CudaTensor(inputSize):zero()
  -- Indexed from 1 to inputSize --
  oneHotTensor[index + 1] = 1.0
  return oneHotTensor
end

-- Inputs are indexed with one-hot method from 0 to nextEventId - 1 --
local inputSize = nextEventId

local trainingInputTensor = torch.CudaTensor(nTraining - 1, inputSize)
local validationInputTensor = torch.CudaTensor(nValidation - 1, inputSize)
-- We will do a delayed self-association here --
local trainingOutputTensor = torch.LongTensor(nTraining - 1)
local validationOutputTensor = torch.LongTensor(nValidation - 1)

for i, event in ipairs(decodedTraining) do
  -- Invariant here for clarity: inputTensor[x] = outputTensor[x-1], that is, expected output comes from the next input. --
  if (i < nTraining) then
    trainingInputTensor[i] = oneHot(eventIds[event.type], inputSize)
  end
  if (i > 1) then
    trainingOutputTensor[i - 1] = eventIds[event.type] + 1
  end
end

for i, event in ipairs(decodedValidation) do
  -- Invariant here for clarity: inputTensor[x] = outputTensor[x-1], that is, expected output comes from the next input. --
  if (i < nValidation) then
    validationInputTensor[i] = oneHot(eventIds[event.type], inputSize)
  end
  if (i > 1) then
    validationOutputTensor[i - 1] = eventIds[event.type] + 1
  end
end

print("Input read. Generating the neural network initial state. InputSize: ", inputSize)

-- Generating the neural network topology --
-- The input is one-hot coded value in an inputSize size table --

local prevOutputSize = inputSize
local model = nn.Sequential()
model:add(nn.SplitTable(1, 2))
for i, hiddenSize in ipairs(opt.hiddenSize) do
  local rnn = nn.Sequencer(nn.FastLSTM(prevOutputSize, hiddenSize))
  model:add(rnn)
  prevOutputSize = hiddenSize
end
local outputLayers = nn.Sequential()

-- output layer --
outputLayers:add(nn.Linear(prevOutputSize, inputSize))
outputLayers:add(nn.LogSoftMax())
model:add(nn.Sequencer(outputLayers))
model:add(nn.JoinTable(1, 1))
model:add(nn.Reshape(nTraining - 1, inputSize))

-- Initializing the network parameters from uniform distribution --
for k,param in ipairs(model:parameters()) do
  param:uniform(-opt.uniform, opt.uniform)
end

-- Evaluates a single sequence (no minibatch) --
model:remember('both')

print(model)

opt.decayFactor = (opt.minLR - opt.learningRate) / opt.saturateEpoch

-- Initializing the data and targets to the dp package. --
local trainingInputView = dp.DataView('bf', validationInputTensor)
local trainingOutputView = dp.ClassView('b', trainingOutputTensor)
trainingOutputView:setClasses(classes)
local validationInputView = dp.DataView('bf', validationInputTensor)
local validationOutputView = dp.ClassView('b', validationOutputTensor)
validationOutputView:setClasses(classes)

local trainingDataset = dp.DataSet{
  inputs = trainingInputView,
  targets = trainingOutputView,
  which_set = 'training'
}
local validationDataset = dp.DataSet{
  inputs = validationInputView,
  targets = validationOutputView,
  which_set = 'validation'
}
local ds = dp.DataSource{
    train_set = trainingDataset,
    valid_set = validationDataset
}

local trainingOptimizer = dp.Optimizer{
    loss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()),
    epoch_callback = function(model, report) -- called every epoch
      if report.epoch > 0 then
         opt.learningRate = opt.learningRate + opt.decayFactor
         opt.learningRate = math.max(opt.minLR, opt.learningRate)
         if not opt.silent then
            print("learningRate", opt.learningRate)
            if opt.meanNorm then
               print("mean gradParam norm", opt.meanNorm)
            end
         end
      end
    end,
    callback = function(model, report) -- called every batch
      if opt.cutOffNorm > 0 then
         local norm = model:gradParamClip(opt.cutOffNorm) -- affects gradParams
         opt.meanNorm = opt.meanNorm and (opt.meanNorm * 0.9 + norm * 0.1) or norm
      end
      model:updateGradParameters(opt.momentum) -- affects gradParams
      model:updateParameters(opt.learningRate) -- affects params
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
    end,
    feedback = dp.Confusion(),
    sampler = dp.Sampler{batch_size = opt.batchSize}, 
    acc_update = opt.accUpdate,
    progress = opt.progress
}
local validationEvaluator = dp.Evaluator{
    feedback = dp.Confusion(),
    sampler = dp.Sampler{batch_size = opt.batchSize},
    progress = opt.progress
}
local evaluator = dp.Evaluator{
    feedback = dp.Confusion(),
    sampler = dp.Sampler{batch_size = opt.batchSize}
}

xp = dp.Experiment{
    model = model,
    optimizer = trainingOptimizer,
    validator = validationEvaluator,
    tester = evaluator,
    observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
        error_report = {'validator','feedback','confusion','accuracy'},
        maximize = true,
        max_epochs = opt.maxTries
      }
    },
    random_seed = os.time(),
    max_epoch = opt.maxEpoch
}
print("Converting to CUDA...")
xp:cuda()
print("Starting experiment...")
xp:run(ds)
print("Done.")
