require 'torch'
require 'image'

local StoringExpert = torch.class('StoringExpert')

function StoringExpert:__init(path)
   self.path = path
end

function StoringExpert:save_png(dataset, img, disp_max, pred, pred_bad, pred_good, mask, network_name)

   local img_pred = torch.Tensor(1, 3, pred:size(3), pred:size(4))
   adcensus.grey2jet(pred:double():add(1)[{1,1}]:div(disp_max):double(), img_pred)
   local x0 = img.x_batch[1]
   if x0:size(1) == 1 then
      x0 = torch.repeatTensor(x0:cuda(), 3, 1, 1)
   end
   img_err = x0:mul(50):add(150):div(255)

   local real = torch.CudaTensor():resizeAs(img_err):copy(img_err)
   img_err[{1}]:add( 0.7, pred_bad)
   img_err[{2}]:add(-0.7, pred_bad)
   img_err[{3}]:add(-0.7, pred_bad)
   img_err[{1}]:add(-0.7, pred_good)
   img_err[{2}]:add( 0.7, pred_good)
   img_err[{3}]:add(-0.7, pred_good)

   local gt
   if dataset.name == 'kitti2012' or dataset.name == 'kitti2015' then
      gt = img.dispnoc
   elseif dataset.name == 'mb' then
      gt = img.dispnoc:resize(1, 1, pred:size(3), pred:size(4))
   end
   local img_gt = torch.Tensor(1, 3, pred:size(3), pred:size(4)):zero()
   adcensus.grey2jet(gt:double():add(1)[{1}]:div(disp_max):double(), img_gt)
   img_gt[{1,3}]:cmul(mask:double())
   
   image.save((self.path .. '/%s_%s_gt.png'):format(dataset.name, img.id), img_gt[1])
   image.save((self.path .. '/%s_%s_real.png'):format(dataset.name, img.id), real[1])
   image.save((self.path .. '/%s_%s_%s_pred.png'):format(dataset.name, network_name, img.id), img_pred[1])
   image.save((self.path .. '/%s_%s_%s_err.png'):format(dataset.name, network_name, img.id), img_err[1])
end

function StoringExpert:save_png_noerror(dataset, img, disp_filtered, disp_max, sufix)
   local norm_disp = disp_filtered:double():add(1)[{1}]:div(disp_max):double()
   image.save((self.path .. '/%s_%s_noerr_%s.png'):format(dataset.name, img.id, sufix), norm_disp[1])
end