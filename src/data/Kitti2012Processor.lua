require 'data/KittiProcessor'

local Kitti2012Processor, parent = torch.class('Kitti2012Processor','KittiProcessor')

local function createProcessor(opt)
   return Kitti2012Processor:new(opt)
end

function Kitti2012Processor:__init(self, opt)
   self.name = 'kitti2012'
   local color = opt.color == 'rgb' and '.rgb' or '' 
   parent.kitti2012Dir = opt.dataset .. '/kitti' .. color
   parent.dir = parent.kitti2012Dir

   --better parameters for the network
   self.n_te = 195
   self.n_tr = 194
   
   parent.__init(parent, self, opt)
end

return createProcessor