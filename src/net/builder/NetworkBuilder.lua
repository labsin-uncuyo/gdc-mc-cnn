require 'torch'
require 'nn'
require 'paths'

require('net/Network')
require('net/component/Concatenation')
require('net/component/SpatialConvolution1_fw')

local NetworkBuilder, parent = torch.class('NetworkBuilder', 'Network')

function NetworkBuilder:__init(args)
   parent.__init(self)
end

function NetworkBuilder:getConvolutional1FromLinear(linearLayer)
   local w = linearLayer.weight
   local b = linearLayer.bias
   local conv = nn.SpatialConvolution1_fw(w:size(2), w:size(1)):cuda()
   conv.weight:copy(w)
   conv.bias:copy(b)
   return conv
end

function NetworkBuilder:getConcatenation()
   return nn.Concatenation():cuda()
end

function NetworkBuilder:padConvs(module)
   -- Pads the convolutional layers to maintain the image resolution
   for i = 1,#module.modules do
      local m = module:get(i)
      if torch.typename(m) == 'cudnn.SpatialConvolution' or torch.typename(m) == 'cudnn.SpatialMaxPooling' then
         m.dW = 1
         m.dH = 1
         if m.kW > 1 then
            m.padW = (m.kW - 1) / 2
         end
         if m.kH > 1 then
            m.padH = (m.kH - 1) / 2
         end
      --elseif torch.typename(m) == 'cudnn.Fire' then
      --   self:padConvs(m.fire_net)
      elseif m.modules then
         self:padConvs(m)
      end
   end
end

function NetworkBuilder.getWindowSize(net, ws)
   ws = ws or 1

   --for i = 1,#net.modules do
   for i = #net.modules,1,-1 do
      local module = net:get(i)
      if torch.typename(module) == 'cudnn.SpatialConvolution' or torch.typename(module) == 'cudnn.SpatialMaxPooling' then
         --ws = ws + module.kW - 1 - module.padW - module.padH
         ws = ((ws - 1) * module.dW) - (2 * module.padW) + module.kW
      end
      if module.modules then
         ws = NetworkBuilder.getWindowSize(module, ws)
      end
   end
   return ws
end

function NetworkBuilder:save(epoch, optimState)
   local fname = ''
   if epoch == 0 then
      fname = (self.name)
   else
      fname = ('debug/%s_%d'):format(self.name, epoch)
   end

   local modelPath = paths.concat(self.path, fname .. '_net.t7')
   local optimPath = paths.concat(self.path, fname .. '_optim.t7')
   local latestPath = paths.concat(self.path, fname .. '.t7')
   local modelFile = {
      description = self.clean(self.net.modules[1]),
      decision = self.clean(self.net.modules[2]),
      params = self.params,
      name = self.name}

   torch.save(modelPath, modelFile)
   torch.save(optimPath, optimState)
   torch.save(latestPath, {
      epoch = epoch,
      modelPath = modelPath,
      optimPath = optimPath,
   })

   return latestPath
end

function NetworkBuilder:load(opt)
   if opt.mcnet == '' then
      self:buildNet()
      return nil
   else
      local checkpoint = torch.load(opt.mcnet)
      local model, optimState
      if checkpoint.modelPath and paths.filep(checkpoint.modelPath) then
         model = torch.load(checkpoint.modelPath)
         optimState = torch.load(checkpoint.optimPath)
      else
         model = checkpoint
         checkpoint = nil
      end

      --self.description = model.description
      --self.decision = model.decision
      self.net = nn.Sequential()
         :add(model.description)
         :add(model.decision)
         :cuda()

      self.params = model.params
      self.name = model.name

      return checkpoint, optimState
   end
end