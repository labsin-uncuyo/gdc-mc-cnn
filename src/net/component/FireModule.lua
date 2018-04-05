require 'torch'
require 'nn'
require 'cudnn'

local Fire, parent = torch.class('nn.Fire', 'nn.Module')

function Fire:__init(n_planes, n_squeeze1x1, n_expand1x1, n_expand3x3)
   parent.__init(self)
   self.n_input = n_planes
   self.n_squeeze1x1 = n_squeeze1x1
   self.n_expand1x1 = n_expand1x1
   self.n_expand3x3 = n_expand3x3
   self:cuda()
   
   self:buildModule()
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
   local expand_net = nn.Concat(2)
   expand_net:add(expand1x1_net)
   expand_net:add(expand3x3_net)
   
   -- creating final net
   local module_net = nn.Sequential()
   module_net:add(self.squeeze)
   module_net:add(self.squeeze_activation)
   module_net:add(expand_net)
   
   self.fire_net = module_net
end

function Fire:updateOutput(input)
   --local squeeze_output = self.squeeze_activation:updateOutput(self.squeeze:updateOutput(input))
   --self.output = torch.cat(self.expand1x1_activation:updateOutput(self.expand1x1:updateOutput(squeeze_output)), self.expand3x3_activation:updateOutput(self.expand3x3:updateOutput(squeeze_output)), 2)
   
   --print("Squeeze output size: ", squeeze_output:size())
   self.output = self.fire_net:updateOutput(input)
   return self.output
end

function Fire:updateGradInput(input, gradOutput)
   self.gradInput = self.fire_net:updateGradInput(input, gradOutput)
   return self.gradInput
end

function Fire:accGradParameters(input, gradOutput, scale)
   self.gradOutput = self.fire_net:accGradParameters(input, gradOutput, scale)
   return self.gradOutput
end

function Fire:accUpdateGradParameters(input, gradOutput, lr)
   self.fire_net:accUpdateGradParameters(input, gradOutput, lr)
end

function clearModules(container)
   for i=1, #container.modules do
      local m  = container.modules[i]
      if m.modules then
         clearModules(m)
      else
         m:apply(
            function(mod)
               mod:clearState()
            end
            )
      end
   end
end

function Fire:clearState()
   self.output = nil
   clearModules(self.fire_net)
end

function Fire:__tostring__()
   return tostring(self.fire_net)
end