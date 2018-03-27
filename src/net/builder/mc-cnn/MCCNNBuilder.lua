require 'torch'
require 'nn'
require 'cudnn'

require 'net/builder/NetworkBuilder'

local MCCNNBuilder, parent = torch.class('MCCNNBuilder', 'NetworkBuilder')

function MCCNNBuilder:__init(args)
   parent.__init(self)
end

function MCCNNBuilder:buildNet()
   local accurateNet = nn.Sequential()
   
   local descriptionNet = self:buildDescriptionNet()
   local decisionNet = self:buildDecisionNet()
   
   --self.description = descriptionNet
   --self.decision = decisionNet
   
   accurateNet:add(descriptionNet)
   accurateNet:add(decisionNet)
   
   accurateNet:cuda()
   
   print(accurateNet)
   
   self.net = accurateNet
   
   self.params.ws = self.getWindowSize(self.net)
end

function MCCNNBuilder:buildTestNet(netModel)

   local testnet = netModel:clone('weight', 'bias')
   
   -- replace linear with 1X1 conv
   local nodes, containers = testnet:findModules('nn.Linear')
   for i = 1, #nodes do
      for j = 1, #(containers[i].modules) do
         if containers[i].modules[j] == nodes[i] then
            -- Replace with a new instance
            containers[i].modules[j] = self:getConvolutional1FromLinear(nodes[i])
         end
      end
   end

   -- replace reshape with concatenation
   nodes, containers = testnet:findModules('nn.Reshape')
   for i = 1, #nodes do
      for j = 1, #(containers[i].modules) do
         if containers[i].modules[j] == nodes[i] then
            -- Replace with a new instance
            containers[i].modules[j] = self:getConcatenation()
         end
      end
   end

   -- pad convolutions
   self:padConvs(testnet)
   
   -- switch to evalutation mode
   testnet:evaluate()
   
   --print(testnet)
   
   --self.testNet = testnet
   return testnet
   
end