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

local function buildFire(input, squeeze_features, expand1_features, expand3_features)
   local squeeze = cudnn.SpatialConvolution(input, squeeze_features, 1, 1)
   local squeeze_activation = nn.ReLU(true)
   local expand1x1 = cudnn.SpatialConvolution(squeeze_features, expand1_features, 1, 1)
   local expand1x1_activation = nn.ReLU(true)
   local expand3x3 = cudnn.SpatialConvolution(squeeze_features, expand3_features, 3, 3, 1, 1, 1, 1)
   local expand3x3_activation = nn.ReLU(true)
   
   -- creating expand sequences
   local expand1x1_net = nn.Sequential()
   expand1x1_net:add(expand1x1)
   expand1x1_net:add(expand1x1_activation)
   local expand3x3_net = nn.Sequential()
   expand3x3_net:add(expand3x3)
   expand3x3_net:add(expand3x3_activation)
   
   -- creating expand parallel components
   local expand_net = nn.Concat(2)
   expand_net:add(expand1x1_net)
   expand_net:add(expand3x3_net)
   
   -- creating final net
   local module_net = nn.Sequential()
   module_net:add(squeeze)
   module_net:add(squeeze_activation)
   module_net:add(expand_net)
   
   return module_net
end

function SqueezeBuilder:buildDescriptionNet()
   local descriptionNet = nn.Sequential()
   
   descriptionNet:add(cudnn.SpatialConvolution(self.params.n_input_planes, 96, 7, 7, 2, 2))
   descriptionNet:add(nn.ReLU(true))
   descriptionNet:add(cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil())
   
   --descriptionNet:add(nn.Fire(96, 16, 64, 64))
   descriptionNet:add(buildFire(96, 16, 64, 64))
   --descriptionNet:add(nn.Fire(128, 16, 64, 64))
   descriptionNet:add(buildFire(128, 16, 64, 64))
   --descriptionNet:add(nn.Fire(128, 32, 128, 128))
   descriptionNet:add(buildFire(128, 32, 128, 128))
   
   descriptionNet:add(cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil())
   
   --descriptionNet:add(nn.Fire(256, 32, 128, 128))
   descriptionNet:add(buildFire(256, 32, 128, 128))
   --descriptionNet:add(nn.Fire(256, 48, 192, 192))
   descriptionNet:add(buildFire(256, 48, 192, 192))
   --descriptionNet:add(nn.Fire(384, 48, 192, 192))
   descriptionNet:add(buildFire(384, 48, 192, 192))
   --descriptionNet:add(nn.Fire(384, 64, 256, 256))
   descriptionNet:add(buildFire(384, 64, 256, 256))
   
   descriptionNet:add(cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil())
   
   --descriptionNet:add(nn.Fire(512, 64, 256, 256))
   descriptionNet:add(buildFire(512, 64, 256, 256))
   
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

