#!/usr/local/bin/lua qlua

require 'torch'
require 'cutorch'
require 'cunn'
require 'image'

require 'libadcensus'
require 'libcv'

require("net/model/AccurateModel")
require("expert/TrainingExpert")

local opts_parser = require 'opts_parser'
local opt = opts_parser:parse(args)

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
cutorch.setDevice(tonumber(opt.gpu))

local dataset = require("data/" .. opt.ds .. 'Processor')(opt)
dataset:load(opt)

local model = AccurateModel(opt)

local function main()

   print('working...')
   
   -- Load last checkpoint if exists
   print('===> Loading matching cost network...')
   local checkpoint, optim_state = model:load(opt)
   print('===> Loaded! Network: ' .. model.name)
   
   local start_epoch = checkpoint and checkpoint.epoch +1 or opt.start_epoch
   
   --model:buildNet()
   
   --model:buildTestNet(model.net)
   
   -- Initialize new trainer for the MCN
   local trainingExpert = TrainingExpert(model, optim_state, opt)
   
   trainingExpert:train(dataset, start_epoch)
   
   --model:testclone()
   
   --local trainingExp = TrainingExpert:new{model = model, dataset = dataset}
   
   --trainingExp:train()
   
end

main()
