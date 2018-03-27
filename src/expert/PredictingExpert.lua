require 'torch'

require 'expert/MatchingExpert'
require 'expert/PostProcessingExpert'
require 'expert/DisparityExpert'
require 'expert/RefinementExpert'

local PredictingExpert = torch.class('PredictingExpert')

function PredictingExpert:__init(network, dataset, opt, path)
   self.network = network
   self.dataset = dataset
   self.opt = opt
   self.path = path
   self.matching = MatchingExpert()
   self.post_processing = PostProcessingExpert()
   self.disparity = DisparityExpert()
   self.refinement = RefinementExpert()
end

function PredictingExpert:predict(img, disp_max, directions, make_cache)

   local vox
   -- compute matching cost
   if self.opt.use_cache then
      vox = torch.load(('%s_%s.t7'):format(self.path, img.id)):cuda()
   else
      vox = self.matching:match(self.network, img.x_batch,
         disp_max, directions):cuda()
      if make_cache then
         torch.save(('%s_%s.t7'):format(self.path, img.id), vox)
      end
   end
   collectgarbage()

   -- post_process
   vox = self.post_processing:process(vox, img.x_batch, disp_max, self.network.params, self.dataset, self.opt.sm_terminate, self.opt.sm_skip, directions)
   collectgarbage()

   -- pred after post process
   local vox_simple = vox:clone()

   -- disparity image
   local disp, vox, conf, t = self.disparity:disparityImage(vox, self.gdn)

   -- refinement
   disp = self.refinement:refine(disp, vox_simple, self.network.params, self.dataset, self.opt.sm_skip ,self.opt.sm_terminate, disp_max, conf, t.t1, t.t2)

   return disp[2]

end