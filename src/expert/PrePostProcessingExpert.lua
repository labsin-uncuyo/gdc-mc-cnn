require 'torch'
require 'libgdcutils'

local PrePostProcessingExpert = torch.class('PrePostProcessingExpert')

function PrePostProcessingExpert:__init(opt)
   self.depth_threshold = opt.depth_threshold
end

function PrePostProcessingExpert:depth_filter(map)
   local out = gdcutils.depth_filter(map[1], self.depth_threshold)
   return out
end

function PrePostProcessingExpert:erode(map)
   local kernel = torch.CharTensor({{0,1,0},{1,1,1},{0,1,0}}):cuda()
   local out = gdcutils.erode(map, kernel)
end

function PrePostProcessingExpert:npp_info()
   print("entering npp info test")
   --gdcutils.print_npp_info()
   print("npp info test successful")
end