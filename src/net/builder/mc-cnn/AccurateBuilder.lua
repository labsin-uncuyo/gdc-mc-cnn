require 'torch'
require 'nn'
require 'cudnn'

require 'net/matcher/AccurateMatcher'
require 'net/builder/MCCNNBuilder'

local AccurateBuilder, parent = torch.class('AccurateBuilder', 'MCCNNBuilder')

function AccurateBuilder:__init()
   parent.__init(self)
   self.matcher = AccurateMatcher(self)
end

function AccurateBuilder:buildDescriptionNet()
   local descriptionNet = nn.Sequential()
   
   for i = 1, self.params.n_conv_layers do
      descriptionNet:add(cudnn.SpatialConvolution(i == 1 and self.params.n_input_planes or self.params.n_feature_maps, self.params.n_feature_maps, self.params.kernel_sizes[i], self.params.kernel_sizes[i]))
      descriptionNet:add(nn.ReLU(true))
   end
   descriptionNet:cuda()
   
   return descriptionNet
end

function AccurateBuilder:buildDecisionNet()
   local decisionNet = nn.Sequential()
   
   decisionNet:add(nn.Reshape(self.params.batch_size, self.params.n_feature_maps*2))
   
   for i=1, self.params.n_fully_connected_layers do
      decisionNet:add(nn.Linear(i == 1 and self.params.n_feature_maps*2 or self.params.n_fully_connected_units, i == self.params.n_fully_connected_layers and 1 or self.params.n_fully_connected_units))
      if i ~= self.params.n_fully_connected_layers then
         decisionNet:add(nn.ReLU(true))
      end
   end
   
   decisionNet:add(nn.Sigmoid(false))
   
   decisionNet:cuda()
   
   return decisionNet
end





