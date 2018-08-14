require 'torch'
require 'libgdcutils'
local cv = require 'cv'
require 'cv.highgui'
require 'cv.imgproc'
require 'cv.cudafilters'
require 'image'

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
   print("Showing element 0 of the map: ", map[1][1][1][1])
   local floatTensor = nn.utils.recursiveType(map[1]:permute(2,3,1), 'torch.FloatTensor')
   local out = gdcutils.erode(floatTensor, kernel)
   map[1] = nn.utils.recursiveType(out:permute(3,1,2), "torch.CudaTensor")
   return out
end

-- This method was used to test cuda erode and dilate, but has issues converting ByteTensors to Cuda Tensors
function PrePostProcessingExpert:erode_dilate_disp(disp)
   print("Processing first map")
   local floatTensor = nn.utils.recursiveType(disp[1][1]:permute(2,3,1):clone(), 'torch.FloatTensor')
   local byteTensor = floatTensor:byte()
   local out_cross = torch.FloatTensor():resizeAs(floatTensor):zero()
   local out_cross_b = torch.ByteTensor():resizeAs(byteTensor):zero()
   local out_square = torch.FloatTensor():resizeAs(floatTensor):zero()
   local kernel_cross_3x3 = torch.ByteTensor({{0,1,0},{1,1,1},{0,1,0}})
   local kernel_square_3x3 = torch.ByteTensor({{1,1,1},{1,1,1},{1,1,1}})
   
   local test_b_in = torch.ByteTensor({{1,1,1},{1,1,1},{1,1,1}})
   local test_b_out = torch.ByteTensor():resizeAs(test_b_in):zero()
   
   local kernel_cross_3x3_c = torch.ByteTensor({{0,1,0},{1,1,1},{0,1,0}}):cuda()
   local kernel_cross_3x3_w = cv.wrap_tensor(torch.ByteTensor({{0,1,0},{1,1,1},{0,1,0}}))
   
   local kernel_cross_5x5 = torch.ByteTensor({{0,0,1,0,0},{0,1,1,1,0},{1,1,1,1,1},{0,1,1,1,0},{0,0,1,0,0}})
   local kernel_square_5x5 = torch.ByteTensor({{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}})
   
   local kernel = torch.ByteTensor({{0,1,0},{1,1,1},{0,1,0}})
   
   local filter = cv.cuda.createMorphologyFilter{op=cv.MORPH_ERODE, srcType=cv.CV_8UC1, kernel=kernel_cross_3x3_w}
   filter:apply{src=byteTensor, dst=out_cross_b}
   
   cv.erode{floatTensor, out_cross, kernel_cross_3x3}
   cv.erode{floatTensor, out_square, kernel_square_3x3}
   
   cv.dilate{out_cross, out_cross, kernel_cross_3x3}
   cv.dilate{out_square, out_square, kernel_square_3x3}
   
   cv.erode{floatTensor, out_cross, kernel_cross_5x5}
   cv.erode{floatTensor, out_square, kernel_square_5x5}
   
   cv.dilate{out_cross, out_cross, kernel_cross_5x5}
   cv.dilate{out_square, out_square, kernel_square_5x5}
   
   disp[1][1] = nn.utils.recursiveType(out_cross:permute(3,1,2), 'torch.CudaTensor')
   
   local floatTensor = nn.utils.recursiveType(disp[2][1]:permute(2,3,1), 'torch.FloatTensor')
   
   cv.erode{floatTensor, out_cross, kernel_cross_3x3}
   cv.erode{floatTensor, out_square, kernel_square_3x3}
   
   cv.dilate{out_cross, out_cross, kernel_cross_3x3}
   cv.dilate{out_square, out_square, kernel_square_3x3}
   
   cv.erode{floatTensor, out_cross, kernel_cross_5x5}
   cv.erode{floatTensor, out_square, kernel_square_5x5}
   
   cv.dilate{out_cross, out_cross, kernel_cross_5x5}
   cv.dilate{out_square, out_square, kernel_square_5x5}
   
   disp[2][1] = nn.utils.recursiveType(out_cross:permute(3,1,2), 'torch.CudaTensor')
   
   cv.imshow{'mapita', floatTensor}
   cv.imshow{'dilate cross', out_cross}
   cv.imshow{'dilate square', out_square}
   cv.waitKey{0}
   
   return disp
end

function PrePostProcessingExpert:cv_erode(map)
   --imgT = image.lena()
   --cv.imshow{"Lena", imgT:permute(2,3,1)}
   --cv:waitKey{0}
   
   local floatTensor = nn.utils.recursiveType(map[1][1]:permute(2,3,1), 'torch.FloatTensor')
   local out_cross = torch.FloatTensor():resizeAs(floatTensor):zero()
   local kernel_cross_3x3 = torch.ByteTensor({{0,1,0},{1,1,1},{0,1,0}})
   
   local kernel_cross_5x5 = torch.ByteTensor({{0,0,1,0,0},{0,1,1,1,0},{1,1,1,1,1},{0,1,1,1,0},{0,0,1,0,0}})
   
   
   cv.erode{floatTensor, out_cross, kernel_cross_3x3}
   
   --cv.imshow{'mapita', floatTensor}
   --cv.imshow{'erode cross', out_cross}
   --cv.waitKey{0}
   
   cv.dilate{out_cross, out_cross, kernel_cross_3x3}
   
   --cv.imshow{'mapita', floatTensor}
   --cv.imshow{'dilate cross', out_cross}
   --cv.waitKey{0}
   
   cv.erode{out_cross, out_cross, kernel_cross_5x5}
   
   --cv.imshow{'mapita', floatTensor}
   --cv.imshow{'erode cross 2', out_cross}
   --cv.waitKey{0}
   
   cv.dilate{out_cross, out_cross, kernel_cross_5x5}
   
   --cv.imshow{'mapita', floatTensor}
   --cv.imshow{'dilate cross 2', out_cross}
   --cv.waitKey{0}
   
   map[1][1] = nn.utils.recursiveType(out_cross:permute(3,1,2), 'torch.CudaTensor')
   
   local floatTensor = nn.utils.recursiveType(map[2][1]:permute(2,3,1), 'torch.FloatTensor')
   
   cv.erode{floatTensor, out_cross, kernel_cross_3x3}
   
   --cv.imshow{'mapita', floatTensor}
   --cv.imshow{'erode cross', out_cross}
   --cv.waitKey{0}
   
   cv.dilate{out_cross, out_cross, kernel_cross_3x3}
   
   --cv.imshow{'mapita', floatTensor}
   --cv.imshow{'dilate cross', out_cross}
   --cv.waitKey{0}
   
   cv.erode{out_cross, out_cross, kernel_cross_5x5}
   
   --cv.imshow{'mapita', floatTensor}
   --cv.imshow{'erode cross 2', out_cross}
  -- cv.waitKey{0}
   
   cv.dilate{out_cross, out_cross, kernel_cross_5x5}
   
   --cv.imshow{'mapita', floatTensor}
   --cv.imshow{'dilate cross 2', out_cross}
   --cv.waitKey{0}
   
   map[2][1] = nn.utils.recursiveType(out_cross:permute(3,1,2), 'torch.CudaTensor')
   --prueba 1
   --cv.imshow{'mapita', map[1][1]:permute(2,3,1)}
   return map
end

function PrePostProcessingExpert:npp_info()
   print("entering npp info test")
   --gdcutils.print_npp_info()
   print("npp info test successful")
end