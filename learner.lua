require 'cutorch'
require 'rnn'
require 'dp'
require 'cunnx'
require 'nn'
require 'nngraph'

nngraph.setDebug(true)

-- Note: For the dot/svg graphs, run with: qlua learner.lua --nngraph 1
--       (nngraph uses QT).
--       Otherwise: th learner.lua

-- First we initialize the neural network parameters --

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a system to model a Flexible Assembly System')
cmd:option('--name', '1', 'The name of the run for the report')
cmd:option('--useDevice', 1, '')
cmd:option('--learningRate', 0.8, '')
cmd:option('--uniform', 0.9, 'The initial values are taken from a uniform distribution between negative and positive given value.')
cmd:option('--lrDecay', 'linear', '')
cmd:option('--minLR', 0.00001, '')
cmd:option('--saturateEpoch', 500, '')
cmd:option('--maxEpoch', 500, '')
cmd:option('--maxTries', 500, '')
cmd:option('--decayFactor', 0.001, '')
cmd:option('--momentum', 0, '')
cmd:option('--maxOutNorm', 2, '')
cmd:option('--cutOffNorm', -1, '')
cmd:option('--trainFile', 'train.json', 'The input file used for training')
cmd:option('--validationFile', 'validation.json', 'The input file used for validation')
cmd:option('--reportFile', 'report.json', 'The file for appending final results to')
cmd:option('--hiddenSize', '{20,20}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, LSTMs are stacked')
cmd:option('--seed', 1, '')
cmd:option('--nngraph', 0, 'Set this to one to print ngraph output.')
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

reportfile = assert(io.open(opt.reportFile, "a"))

local decodedTraining = json.decode(training)
local decodedValidation = json.decode(validation)

-- A map from event type string to event id number --
local eventIds = {}
-- A map from event id number to event type string --
local eventTypes = {}
-- For assigning weights to the ClassNLLCriterion to prevent it from obsessing over two overrepresented classes.
local countEventsPerId = {}

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
  if (countEventsPerId[eventIds[event.type]]) then
    countEventsPerId[eventIds[event.type]] = countEventsPerId[eventIds[event.type]] + 1
  else
    countEventsPerId[eventIds[event.type]] = 1
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

local classWeights = {}
for i, count in ipairs(countEventsPerId) do
  classWeights[i] = count / nTraining
end

function oneHot(index, inputSize)
  local oneHotTensor
  if (opt.nngraph == 1) then
    -- nngraph does not work with CUDA tensors
    oneHotTensor = torch.DoubleTensor(inputSize):zero()
  else
    oneHotTensor = torch.CudaTensor(inputSize):zero()
  end
  -- Indexed from 1 to inputSize --
  oneHotTensor[index + 1] = 1.0
  return oneHotTensor
end

-- Inputs are indexed with one-hot method from 0 to nextEventId - 1 --
local inputSize = nextEventId

local trainingInputTensor
local validationInputTensor
if (opt.nngraph == 1) then
  -- nngraph does not work with CUDA tensors
  trainingInputTensor = torch.DoubleTensor(nTraining - 1, inputSize)
  validationInputTensor = torch.DoubleTensor(nValidation - 1, inputSize)
else
  trainingInputTensor = torch.CudaTensor(nTraining - 1, inputSize)
  validationInputTensor = torch.CudaTensor(nValidation - 1, inputSize)
end

-- We don't use minibatch here, but simply learn a single continuous sequence.
local batchSize = nTraining - 1

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

local splitLayer = nn.SplitTable(1, 2);
model:add(splitLayer)
local rnnLayers = {}
for i, hiddenSize in ipairs(opt.hiddenSize) do
  local rnnLayer = nn.Sequencer(nn.FastLSTM(prevOutputSize, hiddenSize))
  model:add(rnnLayer)
  rnnLayers[i] = rnnLayer
  prevOutputSize = hiddenSize
end
local outputLayers = nn.Sequential()

-- output layer --
outputLayers:add(nn.Linear(prevOutputSize, inputSize))
outputLayers:add(nn.LogSoftMax())
local outputSequencer = nn.Sequencer(outputLayers)
model:add(outputSequencer)

-- We want an inverse operation for nn.SplitTable(1, 2)
local joinLayer = nn.JoinTable(1, 1)
model:add(joinLayer)
local reshapeLayer = nn.Reshape(batchSize, inputSize)
model:add(reshapeLayer)

-- The model is as follows:
-- -> torch.CudaTensor(nTraining - 1, inputSize)
-- Minibatching:
-- -> torch.CudaTensor(batchSize, inputSize)
-- nn.Sequential {
--   [input -> (1) -> (2) -> (3) -> (4) -> (5) -> output]
--   -> torch.CudaTensor(batchSize, inputSize)
--   (1): nn.SplitTable(1,2)
--   -> Table(torch.CudaTensor(inputSize), batchSize)
--     Note: The step 2 can be repeated depending on command line arguments for multilayer LSTMs.
--   (2): nn.Sequencer @ nn.FastLSTM
--   -> torch.CudaTensor(inputSize)
--   -> torch.CudaTensor(hiddenLayerSize)
--   -> Table(torch.CudaTensor(batchSize), hiddenLayerSize)
--   (3): nn.Sequencer @ nn.Sequential {
--     [input -> (1) -> (2) -> output]
--     -> torch.CudaTensor(hiddenLayerSize)
--     (1): nn.Linear(hiddenLayerSize -> inputSize)
--     -> torch.CudaTensor(inputSize)
--     (2): nn.LogSoftMax
--     -> torch.CudaTensor(inputSize)
--   }
--   -> Table(torch.CudaTensor(inputSize), batchSize)
--   (4): nn.JoinTable(1,1)
--   -> Table(torch.CudaTensor(inputSize), batchSize)
--   (5): nn.Reshape(batchSize x inputSize)
--   -> Table(torch.CudaTensor(batchSize, inputSize)
-- }



-- Initializing the network parameters from uniform distribution --
for k,param in ipairs(model:parameters()) do
  param:uniform(-opt.uniform, opt.uniform)
end

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

local classWeightsTensor = torch.CudaTensor(classWeights)

if (opt.nngraph == 1) then
  -- We can only drive the nngraph with non-CUDA tensors for some reason.
  -- With CudaTensors it will fail like this:
  -- Linear.lua:38: invalid arguments: DoubleTensor number DoubleTensor CudaTensor 
  --   expected arguments: *DoubleTensor~1D* [DoubleTensor~1D] [double] DoubleTensor~2D DoubleTensor~1D |   
  --   *DoubleTensor~1D* double [DoubleTensor~1D] double DoubleTensor~2D DoubleTensor~1D

  local graphBatch = trainingDataset:batch(batchSize)
  local graphInput = graphBatch:inputs():input()
  local graphNode = nn.Identity()()
  local graphInputLayers = splitLayer(
            {graphNode}
          )
  -- Adding rnnLayrs for graphing.
  local graphRnnLayers
  for k, graphRnnLayer in ipairs(rnnLayers) do 
    if (graphRnnLayers) then
      graphRnnLayers = graphRnnLayer({graphRnnLayers})
    else
      graphRnnLayers = graphRnnLayer({graphInputLayers})
    end
  end

  local graphModel = reshapeLayer({
    joinLayer({
      outputSequencer({
        graphRnnLayers
      })
    })
  })
  local graphModule = nn.gModule({graphNode}, {graphModel})

  -- graphModule:updateOutput(graphInput)
  local graphPrediction = graphModule:forward(graphInput)
  local graphOutput = graphBatch:targets():input()
  local graphCriterion = nn.ModuleCriterion(nn.ClassNLLCriterion(classWeightsTensor), nil, nn.Convert())
  local graphError = graphCriterion:forward(graphPrediction, graphOutput)
  local gradCriterion = graphCriterion:backward(graphPrediction, graphOutput)
  graphModule:zeroGradParameters()
  graphModule:backward(graphInput, gradCriterion)

  -- For helping the nngraph to get the real inputs and gradients:
  -- graphModule:updateOutput(graphInput)
  -- graphModule:updateGradInput(graphInput, gradCriterion)
  -- graphModule:accGradParameters(graphInput, gradCriterion)

  print("Outputting graph...")
  graph.dot(graphModule.fg, 'LSTM_fg', 'LSTM_fg')
  graph.dot(graphModule.bg, 'LSTM_bg', 'LSTM_bg')
  print("Outputted graph.")
end
local trainingConfusion = dp.Confusion()
local validationConfusion = dp.Confusion()
local evaluationConfusion = dp.Confusion()

local trainingOptimizer = dp.Optimizer{
    loss = nn.ModuleCriterion(nn.ClassNLLCriterion(classWeightsTensor), nil, nn.Convert()),
    epoch_callback = function(model, report) -- called every epoch
      local validationReport = validationConfusion:report() 
      if (validationReport.confusion and validationReport.confusion.accuracy and report.epoch and opt.name) then
         reportfile:write("{\"run\": ", opt.name)
         reportfile:write(", \"epoch\": ", report.epoch)
         reportfile:write(", \"accuracy\": ",
           validationReport.confusion.accuracy, "},\n")
      end

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
      local report = validationConfusion:report() 
      if (report.confusion and report.confusion.matrix) then
        print("Validation confusion matrix: ", report.confusion.matrix)
        print("Validation confusion per_class accuracy: ", report.confusion.per_class.accuracy)
      end
    end,
    feedback = trainingConfusion,
    sampler = dp.Sampler{batch_size = batchSize}, 
    acc_update = opt.accUpdate,
    progress = opt.progress
}
local validationEvaluator = dp.Evaluator{
    feedback = validationConfusion,
    sampler = dp.Sampler{batch_size = batchSize},
    progress = opt.progress
}
local evaluator = dp.Evaluator{
    feedback = evaluationConfusion,
    sampler = dp.Sampler{batch_size = batchSize}
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
if (opt.nngraph == 0) then
  -- We will only run the whole training with CUDA.
  print("Converting to CUDA...")
  xp:cuda()
  print("Starting experiment...")
  xp:run(ds)
end
reportfile:close()
print("Done.")

