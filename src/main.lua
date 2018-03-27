#!/usr/local/bin/lua luajit

require 'torch'
require 'cutorch'

local opts_parser = require 'opts_parser'
local opt = opts_parser:parse(args)

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
cutorch.setDevice(tonumber(opt.gpu))

local dataset = require("data/" .. opt.ds .. 'Processor')(opt)
-- dataset:load(opt)

local Network = require("net/Network")
local NetworkLayerType = require("net/NetworkLayerType")

--local model = require("net/models/AccurateModel"):new(opt)
require("net/models/AccurateModel")
require("expert/TrainingExpert")

local model = AccurateModel(opt)


local function main()

   print('working...')
   
   model:buildModel()
   
   model:buildNet()
   
   --model:buildTestNet()
   
   --model:testclone()
   
   local trainingExp = TrainingExpert:new{model = model, dataset = dataset}
   
   trainingExp:train()
   
end

main()
