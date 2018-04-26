Processor = require('data/Processor')

local KittiProcessor, parent = torch.class('KittiProcessor', 'Processor')

function KittiProcessor:__init(self, opt)
   local color = opt.color == 'rgb' and '.rgb' or '' 
   self.kitti2012Dir = opt.dataset .. '/kitti' .. color
   self.kitti2015Dir = opt.dataset .. '/kitti2015' .. color
   parent.__init(parent, self, opt)
end

local function createProcessor(opt)
   return KittiProcessor:new(opt)
end

--- Sets the parameters used for the patch generation and image transformation. It also 
-- holds the information of the image sizes which are later used in the network creation.
--
function KittiProcessor:setParams()
   -- parameters for training
   self.true1 = 1
   self.false1 = 4
   self.false2 = 10
   -- parameters for image transformations
   self.hflip = 0
   self.vflip = 0
   self.rotate = 7
   self.hscale = 0.9
   self.scale = 1
   self.trans = 0
   self.hshear = 0.1
   self.brightness = 0.7
   self.contrast = 1.3
   self.d_vtrans = 0
   self.d_rotate = 0
   self.d_hscale = 1
   self.d_hshear = 0
   self.d_brightness = 0.3
   self.d_contrast = 1

   --parameters for the network
   self.height = 350
   self.width = 1242
   self.disp_max = 228
   self.err_at = 3
end

local function get_sample(xs, p)
   local perm = torch.randperm(xs:nElement()):long()
   return xs:index(1, perm[{{1, xs:size(1) * p}}])
end

local function get_subset(ds, tr, subset)
   local tr_subset = get_sample(tr, subset)
   local nnz_tr_output = torch.FloatTensor(ds:size()):zero()
   local t = adcensus.subset_dataset(tr_subset, ds, nnz_tr_output);

   return nnz_tr_output[{{1,t}}]
end

local function prepareTrainingData(self, subset, all)
   self.nnz_tr = torch.cat(self.nnz_tr, self.nnz_disp, 1)
   self.tr = torch.cat(self.tr, self.tr_disp, 1)
   self.nnz_disp = self.nnz_tr
   self.tr_disp = self.tr
   -- subset training dataset
   if subset < 1 then
      self.nnz = get_subset(self.nnz_tr, self.tr, subset)
      self.disp = get_subset(self.nnz_disp, self.tr_disp, subset)
   elseif all then
      self.nnz = torch.cat(self.nnz_tr, self.nnz_te, 1)   
      self.disp = torch.cat(self.nnz_disp, self.nnz_te, 1)
   else
      self.nnz = self.nnz_tr
      self.disp = self.nnz_disp
   end

   collectgarbage()
end

--- Loads de dataset preprocessed files. It makes use of the options parsed in the application to determine whether both datasets are loaded or only one of them.
--
-- @param opt Options passed to the application.
function KittiProcessor:load(opt)
   if not opt.mix then
      self:load_individual_dataset()
   else
      function load_x(fname)
         local X_12 = torch.load(self.kitti2012Dir .. '/' .. fname)
         local X_15 = torch.load(self.kitti2015Dir .. '/' .. fname)
         local X = torch.cat(X_12[{{1,194}}], X_15[{{1,200}}], 1)
         X = torch.cat(X, dataset == 'kitti2012' and X_12[{{195,389}}] or X_15[{{200,400}}], 1)
         return X
      end

      function load_nnz(fname)
         local X_12 = torch.load(self.kitti2012Dir .. '/' .. fname)
         local X_15 = torch.load(self.kitti2015Dir .. '/' .. fname)
         X_15[{{},1}]:add(194)
         return torch.cat(X_12, X_15, 1)
      end

      print('Loading X0...')
      self.X0 = load_x('x0.t7')
      print('Loading X1...')
      self.X1 = load_x('x1.t7')
      print('Loading metadata...')
      self.metadata = load('metadata.t7')

      print('Loading disparity non occluded...')
      self.dispnoc = torch.cat(torch.load(self.kitti2012Dir .. '/dispnoc.t7'), torch.load(self.kitti2015Dir .. '/dispnoc.t7'), 1)
      print('Loading training subset indexes...')
      self.tr_disp = torch.cat(torch.load(self.kitti2012Dir .. '/tr_disp.t7'), torch.load(self.kitti2015Dir .. '/tr_disp.t7'):add(194))
      print('Loading remaining training subset indexes...')
      self.tr = torch.cat(torch.load(self.kitti2012Dir .. '/tr.t7'), torch.load(self.kitti2015Dir .. '/tr.t7'):add(194))
      print('Loading testing set indexes...')
      self.te = self.name == 'kitti2012' and torch.load(self.kitti2012Dir .. '/te.t7') or torch.load(self.kitti2015Dir .. '/te.t7'):add(194)

      print('Loading training subset disparity locations...')
      self.nnz_disp = load_nnz('nnz_disp.t7')
      print('Loading remaining training subset disparity locations...')
      self.nnz_tr = load_nnz('nnz_tr.t7')
      print('Loading testing set disparity locations...')
      self.nnz_te = load_nnz('nnz_te.t7')
   end
   
   prepareTrainingData(self, opt.subset, opt.all)
end

--- Loads an individual dataset. It makes use of the self.dir attribute to determine the dataset to load.
--
function KittiProcessor:load_individual_dataset()
   print('Loading X0...')
   self.X0 = torch.load(('%s/x0.t7'):format(self.dir))
   print('Loading X1...')
   self.X1 = torch.load(('%s/x1.t7'):format(self.dir))
   print('Loading disparity non occluded...')
   self.dispnoc = torch.load(('%s/dispnoc.t7'):format(self.dir))
   print('Loading metadata...')
   self.metadata = torch.load(('%s/metadata.t7'):format(self.dir))
   print('Loading training subset indexes...')
   self.tr_disp = torch.load(('%s/tr_disp.t7'):format(self.dir))
   print('Loading remaining training subset indexes...')
   self.tr = torch.load(('%s/tr.t7'):format(self.dir))
   print('Loading testing set indexes...')
   self.te = torch.load(('%s/te.t7'):format(self.dir))
   print('Loading training subset disparity locations...')
   self.nnz_disp = torch.load(('%s/nnz_disp.t7'):format(self.dir))
   print('Loading remaining training subset disparity locations...')
   self.nnz_tr = torch.load(('%s/nnz_tr.t7'):format(self.dir))
   print('Loading testing set disparity locations...')
   self.nnz_te = torch.load(('%s/nnz_te.t7'):format(self.dir))
end

function KittiProcessor:shuffle()
   self.perm = torch.randperm(self.nnz:size(1))
   self.perm_disp = torch.randperm(self.disp:size(1))
end

function KittiProcessor:trainingSamples(start, size, ws)
   local x = torch.FloatTensor(size * 4, self.n_channels, ws, ws)
   local y = torch.FloatTensor(size * 2)

   for i=start, start+size-1 do
      local idx = self.perm[i]
      local img = self.nnz[{idx, 1}]
      local dim3 = self.nnz[{idx, 2}]
      local dim4 = self.nnz[{idx, 3}]
      local d = self.nnz[{idx, 4}]

      local d_pos = torch.uniform(-self.true1, self.true1)
      local d_neg = torch.uniform(self.false1, self.false2)
      if torch.uniform() < 0.5 then
         d_neg = -d_neg
      end
      local x0, x1 = self:getLR(img)

      idx = i-start+1
      local params = self:obfuscationParams()
      self:makePatch(x0, x[idx * 4 - 3], dim3, dim4, ws, params.x0)
      self:makePatch(x1, x[idx * 4 - 2], dim3, dim4 - d + d_pos, ws, params.x1)
      self:makePatch(x0, x[idx * 4 - 1], dim3, dim4, ws, params.x0)
      self:makePatch(x1, x[idx * 4 - 0], dim3, dim4 - (d-d_neg), ws, params.x1)

      y[idx * 2 - 1] = 0
      y[idx * 2] = 1
   end

   return x:cuda(), y:cuda()
end

function KittiProcessor:getLR(img)
   local x0 = self.X0[img]
   local x1 = self.X1[img]
   return x0, x1
end

function KittiProcessor:getTestSample(i, submit)
   local img = {}

   img.height = self.metadata[{i,1}]
   img.width = self.metadata[{i,2}]

   img.id = self.metadata[{i,3}]
   if not submit then
      img.dispnoc = self.dispnoc[{i,{},{},{1,img.width}}]:cuda()
   end
   local x0 = self.X0[{{i},{},{},{1,img.width}}]
   local x1 = self.X1[{{i},{},{},{1,img.width}}]

   img.x_batch = torch.CudaTensor(2, self.n_channels, self.height, self.width)
   img.x_batch:resize(2, self.n_channels, x0:size(3), x0:size(4))
   img.x_batch[1]:copy(x0)
   img.x_batch[2]:copy(x1)

   return img
end

function KittiProcessor:getTestRange()
   return torch.totable(self.te)
end

function KittiProcessor:obfuscationParams()
   assert(self.hscale <= 1 and self.scale <= 1)

   local params = {}
   params.x0 = {}
   params.x1 = {}
   local s = torch.uniform(self.scale, 1)
   params.x0.scale = {s * torch.uniform(self.hscale, 1), s}
   if self.hflip == 1 and torch.uniform() < 0.5 then
      params.x0.scale[1] = -params.x0.scale[1]
   end
   if self.vflip == 1 and torch.uniform() < 0.5 then
      params.x0.scale[2] = -params.x0.scale[2]
   end
   params.x0.hshear = torch.uniform(-self.hshear, self.hshear)
   params.x0.trans = {torch.uniform(-self.trans, self.trans), torch.uniform(-self.trans, self.trans)}
   params.x0.phi = torch.uniform(-self.rotate * math.pi / 180, self.rotate * math.pi / 180)
   params.x0.brightness = torch.uniform(-self.brightness, self.brightness)

   assert(self.contrast >= 1 and self.d_contrast >= 1)
   params.x0.contrast = torch.uniform(1 / self.contrast, self.contrast)

   params.x1.scale = {params.x0.scale[1] * torch.uniform(self.d_hscale, 1), params.x0.scale[2]}
   params.x1.hshear = params.x0.hshear + torch.uniform(-self.d_hshear, self.d_hshear)
   params.x1.trans = {params.x0.trans[1], params.x0.trans[2] + torch.uniform(-self.d_vtrans, self.d_vtrans)}
   params.x1.phi = params.x0.phi + torch.uniform(-self.d_rotate * math.pi / 180, self.d_rotate * math.pi / 180)
   params.x1.brightness = params.x0.brightness + torch.uniform(-self.d_brightness, self.d_brightness)
   params.x1.contrast = params.x0.contrast * torch.uniform(1 / self.d_contrast, self.d_contrast)

   return params
end

return createProcessor
