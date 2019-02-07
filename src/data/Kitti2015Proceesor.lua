require 'data/KittiProcessor.lua'

local Kitti2015Processor, parent = torch.class('Kitti2015Processor','KittiProcessor')

local function createProcessor(opt)
   return Kitti2015Processor:new(opt)
end

function Kitti2015Processor:__init(self, opt)
   self.name = 'kitti2015'
   local color = opt.color == 'rgb' and '.rgb' or ''   
   parent.kitti2015Dir = opt.dataset .. '/kitti2015' .. color
   self.dir = parent.kitti2015Dir

   --better parameters for the network
   self.n_te =  200
   self.n_tr =  200
   
   parent.__init(parent, self, opt)
end

return createProcessor
