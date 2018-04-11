require 'torch'

local Network = torch.class('Network')

function Network:__init()
end

function Network:getModelParameters()
   return self.net:getParameters()
end

function Network:evaluation(x, dl_dx, inputs, targets)
   dl_dx:zero()

   local prediction = self.net:forward(inputs)

   local loss_x = self.criterion:forward(prediction, targets)

   self.net:backward(inputs,
      self.criterion:backward(prediction, targets))

   return loss_x, dl_dx

end

function Network:fixBorder(vol, direction, ws)
   local n = (ws - 1) / 2
   for i=1,n do
      vol[{{},{},{},direction * i}]:copy(vol[{{},{},{},direction * (n + 1)}])
   end
end

function Network:forwardFree(net, input)
   -- Forwards the network w.r.t input module by module
   -- while cleaning previous modules state
   local currentOutput = input
   for i=1, #net.modules do
      local m  = net.modules[i]
      local nextOutput
      if torch.typename(m) == 'nn.Sequential' then
         nextOutput = self.forwardFree(m, currentOutput)
         currentOutput = nextOutput:clone()
         
         --model b
         --if currentOutput:storage() ~= nextOutput:storage() then
         --   currentOutput:storage():resize(1)
         --   currentOutput:resize(0)
         --end
         --currentOutput = nextOutput
      elseif torch.typename(m) == 'nn.ConcatTable' or torch.typename(m) == 'nn.ParallelTable' or 'nn.Concat' then
         nextOutput = m:forward(currentOutput)
         currentOutput = {}
         currentOutput[1] = nextOutput[1]:clone()
         currentOutput[2] = nextOutput[2]:clone()
         
         --model b
         --currentOutput[1] = nextOutput[1]
         --currentOutput[2] = nextOutput[2]
      else
         nextOutput = m:updateOutput(currentOutput)
         currentOutput = nextOutput:clone()
         
         --model b
         --if currentOutput:storage() ~= nextOutput:storage() then
         --   currentOutput:storage():resize(1)
         --   currentOutput:resize(0)
         --end
         --currentOutput = nextOutput
         
         -- model a
         --currentOutput = nextOutput:clone()
         --m:apply(
         --   function(mod)
         --      mod:clearState()
         --   end
         --   )
      end
      m:apply(
      function(mod)
         mod:clearState()
      end
      )

      collectgarbage()
   end

   --local lastOutput = currentOutput:clone()

   --for i=1, #net.modules do
   --   local m  = net.modules[i]
   --   m:apply(
   --      function(mod)
   --       mod:clearState()
   --      end
   --      )
   --end

   return currentOutput
end

function Network:getName(opt)
   local name = opt.ds .. '_' .. opt.mc .. '_' .. opt.arch
   
   if opt.mc == 'mc-cnn' then
      name = name .. '_cl_' .. opt.cl .. '_fm_' .. opt.fm .. '_fcl_' .. opt.fcl .. '_nu_' .. opt.nu
      if opt.cks ~= '' then
         name = name .. '_cks_' .. opt.cks
      else
         name = name .. '_ks_' .. opt.ks
      end
   else
      name = name .. '_squeeze_'
   end
   
   if opt.color == 'rgb' then
      name = name .. '_rgb'
   end
   if opt.name ~= '' then
      name = name .. '_' .. opt.name
   end
   if opt.subset < 1 then 
      name = name .. '_sub_' .. opt.subset 
   elseif opt.all then
      name = name .. '_all'
   end

   return name
end

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function Network.clean(model)
   return deepCopy(model):float():clearState()
end