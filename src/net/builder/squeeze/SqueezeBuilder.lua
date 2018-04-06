require 'torch'
require 'nn'
require 'cudnn'

require 'net/matcher/AccurateMatcher'
require 'net/builder/MCCNNBuilder'
require 'net/component/FireModule'

local SqueezeBuilder, parent = torch.class('SqueezeBuilder', 'MCCNNBuilder')

function SqueezeBuilder:__init()
   parent.__init(self)
   self.matcher = AccurateMatcher(self)
end

function SqueezeBuilder:buildDescriptionNet()
   local descriptionNet = nn.Sequential()
   
   descriptionNet:add(cudnn.SpatialConvolution(self.params.n_input_planes, 96, 7, 7, 2, 2))
   descriptionNet:add(nn.ReLU(true))
   descriptionNet:add(cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil())
   
   descriptionNet:add(nn.Fire(96, 16, 64, 64))
   descriptionNet:add(nn.Fire(128, 16, 64, 64))
   descriptionNet:add(nn.Fire(128, 32, 128, 128))
   
   descriptionNet:add(cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil())
   
   descriptionNet:add(nn.Fire(256, 32, 128, 128))
   descriptionNet:add(nn.Fire(256, 48, 192, 192))
   descriptionNet:add(nn.Fire(384, 48, 192, 192))
   descriptionNet:add(nn.Fire(384, 64, 256, 256))
   
   descriptionNet:add(cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil())
   
   descriptionNet:add(nn.Fire(512, 64, 256, 256))
   
   descriptionNet:add(cudnn.SpatialConvolution(512, 64, 1, 1))
   descriptionNet:add(nn.ReLU(true))
   
   descriptionNet:cuda()
   
   return descriptionNet
end

function SqueezeBuilder:buildDecisionNet()
   local decisionNet = nn.Sequential()
   
   decisionNet:add(nn.Reshape(self.params.batch_size, 64*2))
   
   for i=1, self.params.n_fully_connected_layers do
      decisionNet:add(nn.Linear(i == 1 and 64*2 or self.params.n_fully_connected_units, i == self.params.n_fully_connected_layers and 1 or self.params.n_fully_connected_units))
      if i ~= self.params.n_fully_connected_layers then
         decisionNet:add(nn.ReLU(true))
      end
   end
   
   decisionNet:add(nn.Sigmoid(false))
   
   decisionNet:cuda()
   
   return decisionNet
end

