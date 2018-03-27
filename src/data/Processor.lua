local module = {}

local Processor = torch.class('Processor', module)

function Processor:__init(self, opt)
   torch.manualSeed(opt.seed)
   cutorch.manualSeed(opt.seed)
   self.__index = self
   self:setParams()
   self.n_channels = opt.color == 'rgb' and 3 or 1
   --if opt.a == 'train_mcn' or opt.a == 'train_gdn' then
   --   self:prepareTrainingData(opt.subset, opt.all)
   --end
   
end

-- function copied from https://github.com/jzbontar/mc-cnn/blob/master/main.lua
local function mul32(a,b)
   return {a[1]*b[1]+a[2]*b[4], a[1]*b[2]+a[2]*b[5], a[1]*b[3]+a[2]*b[6]+a[3], a[4]*b[1]+a[5]*b[4], a[4]*b[2]+a[5]*b[5], a[4]*b[3]+a[5]*b[6]+a[6]}
end

-- function copied from https://github.com/jzbontar/mc-cnn/blob/master/main.lua
function Processor:makePatch(src, dst, dim3, dim4, ws, params)
   local m = {1, 0, -dim4, 0, 1, -dim3}

   if params then
      m = mul32({1, 0, params.trans[1], 0, 1, params.trans[2]}, m) -- translate
      m = mul32({params.scale[1], 0, 0, 0, params.scale[2], 0}, m) -- scale
      local c = math.cos(params.phi)
      local s = math.sin(params.phi)
      m = mul32({c, s, 0, -s, c, 0}, m) -- rotate
      m = mul32({1, params.hshear, 0, 0, 1, 0}, m) -- shear
   end

   m = mul32({1, 0, (ws - 1) / 2, 0, 1, (ws - 1) / 2}, m)
   m = torch.FloatTensor(m)
   cv.warp_affine(src, dst, m)
   if params then
      dst:mul(params.contrast):add(params.brightness)
   end
end

return module.Processor