require 'torch'
require 'nn'
require 'cudnn'

require 'net/matcher/FastMatcher'
require 'net/builder/MCCNNBuilder'
require 'net/component/FireModule'
require 'net/component/Normalize2'
require 'net/score/DotProduct2'

local MinFastBuilder, parent = torch.class('MinFastBuilder', 'MCCNNBuilder')

function MinFastBuilder:__init()
   parent.__init(self)
   self.matcher = FastMatcher(self)
end

function MinFastBuilder:buildDescriptionNet()
   local descriptionNet = nn.Sequential()
   
   descriptionNet:add(cudnn.SpatialConvolution(self.params.n_input_planes, 64, 3, 3))
   descriptionNet:add(nn.ReLU(true))
   descriptionNet:add(cudnn.SpatialConvolution(64, 64, 1, 1))
   descriptionNet:add(nn.ReLU(true))
   descriptionNet:add(cudnn.SpatialConvolution(64, 64, 1, 1))
   descriptionNet:add(nn.ReLU(true))
   descriptionNet:add(cudnn.SpatialConvolution(64, 64, 1, 1))
   
   descriptionNet:add(nn.Normalize2())
   
   descriptionNet:cuda()
   
   return descriptionNet
end

function SqueezeFastBuilder:buildDecisionNet()
   local decisionNet = nn.Sequential()
   
   decisionNet:add(nn.DotProduct2())
   
   decisionNet:cuda()
   
   return decisionNet
end

