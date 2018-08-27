require 'torch'
require 'nn'
require 'cudnn'

require 'net/criterion/Margin2'
require 'net/builder/squeeze/MinFastBuilder'

local MinFastModel, parent = torch.class('MinFastModel', 'MinFastBuilder')

local function load_params(self, opt)
   self.params = {}
   local p = self.params
   p.n_fully_connected_layers = opt.fcl               -- fully connected layers
   p.n_fully_connected_units = opt.nu                 -- number of units in fully connected layers
   p.n_input_planes = opt.color == 'rgb' and 3 or 1
   
   p.batch_size = opt.bs
   p.epochs = opt.epochs
   
   p.learning_rate = 0.05
   p.learning_rate_decay = 0.0
   p.momentum = 0.9
   p.nesterov = true
   p.dampening = 0.0
   p.weight_decay = 1e-4
   
   if opt.ds == 'Kitti2012' then
      p.L1=0
      p.cbca_i1=0
      p.cbca_i2=0
      p.tau1=0
      p.pi1=4        --sgm_P1
      p.pi2=55.72    --sgm_P2?
      p.sgm_i=1
      p.sgm_q1=3
      p.sgm_q2=2.5
      p.alpha1=1.5   --sgm_V
      p.tau_so=0.02  --sgm_D
      p.blur_sigma=7.74
      p.blur_t=5
   elseif opt.ds == 'Kitti2015' then
      p.L1=5
      p.cbca_i1=2
      p.cbca_i2=4
      p.tau1=0.03
      p.pi1=2.3
      p.pi2=24.25
      p.sgm_i=1
      p.sgm_q1=3
      p.sgm_q2=2
      p.alpha1=1.75
      p.tau_so=0.08
      p.blur_sigma=5.99
      p.blur_t=5
   end
   
   for k,v in pairs(self.params) do print(k, v) end
end

function MinFastModel:__init(opt)
   parent.__init(self)
   load_params(self, opt)
   self.name = self:getName(opt)
   self.path = opt.storage .. '/net/mc/'
   self.criterion = nn.Margin2():cuda()
end