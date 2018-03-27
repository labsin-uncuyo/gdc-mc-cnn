require 'torch'
require 'nn'
require 'cudnn'

local Fire, parent = torch.class('nn.Fire', 'nn.Module')

function Fire:__init(args)
   parent.__init(self)
   self.n_input = args.n_planes
   self.n_squeeze1x1 = args.n_squeeze1x1
   self.n_expand1x1 = args.n_expand1x1
   self.n_expand3x3 = args.n_expand3x3
   self:cuda()
end

function Fire:buildModule()
   -- creating components of the fire module
   self.squeeze = cudnn.SpatialConvolution(self.n_input, self.n_squeeze1x1, 1, 1)
   self.squeeze_activation = nn.ReLU(true)
   self.expand1x1 = cudnn.SpatialConvolution(self.n_squeeze1x1, self.n_expand1x1, 1, 1)
   self.expand1x1_activation = nn.ReLU(true)
   self.expand3x3 = cudnn.SpatialConvolution(self.n_squeeze1x1, self.n_expand3x3, 3, 3, 1, 1, 1, 1)
   self.expand3x3_activation = nn.ReLU(true)
   
   -- creating expand sequences
   local expand1x1_net = nn.Sequential()
   expand1x1_net:add(self.expand1x1)
   expand1x1_net:add(self.expand1x1_activation)
   local expand3x3_net = nn.Sequential()
   expand3x3_net:add(self.expand3x3)
   expand3x3_net:add(self.expand3x3_activation)
   
   -- creating expand parallel components
   local expand_net = nn.Parallel()
   expand_net:add(expand1x1_net)
   expand_net:add(expand3x3_net)
   
   -- creating final net
   local module_net = nn.Sequential()
   module_net:add(self.squeeze)
   module_net:add(self.squeeze_activation)
   module_net:add(expand_net)
   
   self.fire_net = module_net
end