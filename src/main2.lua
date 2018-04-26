#!/usr/local/bin/lua qlua

require 'torch'
require 'cutorch'
require 'cunn'
require 'image'

require 'libadcensus'
require 'libcv'

require("net/model/AccurateModel")
require("net/model/SqueezeModel")
require("net/model/SqueezeFastModel")
require("net/model/FastModel")
require("expert/TrainingExpert")

local opts_parser = require 'opts_parser'
local opt = opts_parser.parse(arg)

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
cutorch.setDevice(tonumber(opt.gpu))

local dataset = require("data/" .. opt.ds .. 'Processor')(opt)
dataset:load(opt)

local model = AccurateModel(opt)
local model2 = SqueezeModel(opt)
local model3 = SqueezeFastModel(opt)
local model4 = FastModel(opt)

local function main()

   print('working...')
   
   -- Load last checkpoint if exists
   print('===> Loading matching cost network...')
   local checkpoint, optim_state = model4:load(opt)
   print('===> Loaded! Network: ' .. model4.name)
   
   if opt.a == 'train' then
   
      local start_epoch = checkpoint and checkpoint.epoch +1 or opt.start_epoch
      
      -- Initialize new trainer for the MCN
      local trainingExpert = TrainingExpert(model4, optim_state, opt)
      
      trainingExpert:train(dataset, start_epoch)
      
   end
   
   if opt.a == 'test' then
   
      local testingExpert = TestingExpert(dataset, model4, nil, opt)
      
      testingExpert:test(dataset:getTestRange(), true, false)
      
   end
   
end

main()
