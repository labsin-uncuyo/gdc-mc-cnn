require 'data/KittiProcessor.lua'

local Kitti2015Processor, parent = torch.class('Kitti2015Processor','KittiProcessor')

local function createProcessor(opt)
   return Kitti2015Processor:new(opt)
end

function Kitti2015Processor:__init(self, opt)
   parent.__init(parent, self, opt)
   self.name = 'kitti2015'
   self.dir = parent.kitti2015Dir

   --better parameters for the network
   self.n_te =  200
   self.n_tr =  200
end

return createProcessor