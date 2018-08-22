require 'torch'
require 'nn'
require 'cudnn'

require 'net/matcher/FastMatcher'
require 'net/builder/MCCNNBuilder'
require 'net/component/Normalize2'
require 'net/score/DotProduct2'

local FastBuilder, parent = torch.class('FastBuilder', 'MCCNNBuilder')

function FastBuilder:__init()
   parent.__init(self)
   self.matcher = FastMatcher(self)
end

function FastBuilder:buildDescriptionNet()
   local descriptionNet = nn.Sequential()
   
   for i = 1,self.params.n_conv_layers-1 do
      descriptionNet:add(cudnn.SpatialConvolution(i == 1 and self.params.n_input_planes or self.params.n_feature_maps, self.params.n_feature_maps, self.params.kernel_sizes[i], self.params.kernel_sizes[i]))
      descriptionNet:add(nn.ReLU(true))
   end
   descriptionNet:add(cudnn.SpatialConvolution(self.params.n_conv_layers == 1 and self.params.n_input_planes or self.params.n_feature_maps, self.params.n_feature_maps, self.params.kernel_sizes[self.params.n_conv_layers], self.params.kernel_sizes[self.params.n_conv_layers]))
   descriptionNet:add(nn.Normalize2())
     
   descriptionNet:cuda()
   
   return descriptionNet
end

function FastBuilder:buildDecisionNet()
   local decisionNet = nn.Sequential()
   
   decisionNet:add(nn.DotProduct2())
   
   decisionNet:cuda()
   
   return decisionNet
end

