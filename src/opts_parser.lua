local opt_module = {}

function opt_module.parse(arg)

   math.randomseed(os.time())

   local cmd = torch.CmdLine()
   cmd:option('-nc', false, 'Determine if the application is launched from the IDE or console')
   
   cmd:option('-ds', 'Kitti2012', 'Dataset to be used')
   
   cmd:option('-mc', 'mc-cnn', 'Matching cost network architecture')
   
--   cmd:option('-dataset', 'dataset', 'Path to the dataset folder')
   cmd:option('-dataset', 'dataset', 'Path to the dataset folder')
   cmd:option('-storage', 'storage', 'Path to the trained nets and cache')
   cmd:option('-arch', 'acrt', 'Architecture type: acrt, fast, squeeze')
   cmd:option('-mix', false, 'Train on both kitti 2012 and kitti 2015')
   cmd:option('-color', 'rgb', 'Defines if the dataset images will be parsed as rgb or gray')
   cmd:option('-seed', math.random(1,100), 'Random seed')
   cmd:option('-gpu', 1, 'gpu id')
   cmd:option('-name', '', 'Add string to the network name')
   
   cmd:option('-mcnet', '', 'Path to MC trained network')
   
   cmd:option('-all', false, 'Train on both train and validation sets')
   cmd:option('-subset', 0.01, 'Percentage of the data set used for training')
   cmd:option('-epochs', 14, 'Number of epochs of training')
   cmd:option('-debug', false)
   cmd:option('-times', 10, 'Test the pipeline every X epochs')
   cmd:option('-after', 10, 'Test every epoch after this one')
   cmd:option('-make_cache', false)
   cmd:option('-use_cache', false)
   cmd:option('-save_img', true)
   cmd:option('-wait', false, 'Wait some time between training batchs to cooldown the cpu')
   cmd:option('-wait_time', 0.2, 'Time to wait between training batchs')
   cmd:option('-wait_batchs', 10, 'Number of batchs to process before cooldown waiting')

   -- Parameters of the matching cost network
   cmd:option('-cl', 4, 'Convolutional Layers number')
   cmd:option('-fm', 112, 'Number of feature maps for convolutional layers')
   cmd:option('-ks', 3, 'Fixed kernel size')
   cmd:option('-cks', '', 'Custom kernel sizes')
   cmd:option('-fcl', 4, 'Fully Connected Layers number')
   cmd:option('-nu', 384, 'Number of units for fully connected layers')
   cmd:option('-bs', 128, 'Batch size')
   cmd:option('-margin', 0.2, '')
   cmd:option('-lambda', 0.8)
   cmd:option('-ws', 3, 'Convolution windows size')
   
   local opt = cmd:parse(arg)
   
   if not opt.nc then
      opt.dataset = '../' .. opt.dataset
      opt.storage = '../' .. opt.storage
      if opt.mcnet ~= '' then
         opt.mcnet = '../' .. opt.mcnet
      end
   end
   
   opt.kss = {}
   if opt.cks ~= '' then
      for i = 1, #opt.cks do
         local currentChar = opt.cks:sub(i, i) 
         if tonumber(currentChar) then
            opt.kss[#opt.kss+1] = currentChar
         end
         
         if #opt.kss == opt.cl then
            break
         end
      end
      if #opt.kss < opt.cl then
         error("cks has less kernel sizes than the amount of layers: got " .. #opt.kss .. ", expected " .. opt.cl)
      end
   else
      for i = 1, opt.cl do
         opt.kss[i] = opt.ks
      end   
   end
   
   return opt
end

return opt_module
