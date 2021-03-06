require 'torch'
require 'optim'

require('expert/TestingExpert')

local TrainingExpert = torch.class('TrainingExpert')

function TrainingExpert:__init(network, optim_state, opt)
   self.params = network.params
   self.network = network
   self.optim = optim.sgd
   
   self.optim_state = optim_state or {
      learning_rate = self.params.learning_rate,
      learning_rate_decay = self.params.learning_rate_decay,
      momentum = self.params.momentum,
      nesterov = self.params.nesterov,
      dampening = self.params.dampening,
      weight_decay = self.params.weight_decay
   }
   
   self.opt = opt
end

local function trainingBatch(dataset, start, size, ws)
   return dataset:trainingSamples(start,size,ws)
end

local function extractNetworkParams(module)
   for i=1,#module.modules do
      local m = module.modules[i]
      if m.modules then
         extractNetworkParams(m)
      elseif torch.typename(m) == 'nn.Fire' then
         extractNetworkParams(m.fire_net)
      else
         if m.weight then
            m.weight_v = torch.CudaTensor(m.weight:size()):zero()
            table.insert(params, m.weight)
            table.insert(grads, m.gradWeight)
            table.insert(momentums, m.weight_v)
         end
         if m.bias then
            m.bias_v = torch.CudaTensor(m.bias:size()):zero()
            table.insert(params, m.bias)
            table.insert(grads, m.gradBias)
            table.insert(momentums, m.bias_v)
         end
      end
   end
end

function TrainingExpert:train(dataset, start_epoch)
   self.dataset_size = dataset.nnz:size(1)
   self.train_batch_size = self.params.batch_size/2
   
   local testingExpert = TestingExpert(dataset, self.network, nil, self.opt)
   
   params = {}
   grads = {}
   momentums = {}
   extractNetworkParams(self.network.net)

   for epoch = start_epoch and start_epoch or 1, self.params.epochs do
      dataset:shuffle() -- to get random order of samples
      
      --local err_te = testingExpert:test(dataset:getTestRange(), false, false)

      -- Train one epoch of all the samples
      local err_tr = self:batchEpochTrain(epoch, dataset)

      -- Output results
      local msg = ('train epoch %g\t err %g\tlr %g\n')
         :format(epoch, err_tr, self.optim_state.learning_rate)
      print(msg)

      -- Save the current checkpoint
      self.network:save(epoch, self.optim_state)

      -- Run validation if wanted
      local validate = ((self.opt.debug and epoch % self.opt.times == 0)
         or (epoch >= self.opt.after)) and epoch < self.opt.epochs
      if validate then
         print('===> testing...')
         local err_te = testingExpert:test(dataset:getTestRange(), false, false)

         -- Output validation results
         print(('test epoch: %g\terror: %g\n'):format(epoch, err_te))
         --log:write(('test epoch: %g\terror: %g\n'):format(epoch, err_te))
      end
   end
   
   -- After training is completed test and save the final model
   self.network:save(0, self.optim_state)
   
   local err_te = testingExpert:test(dataset:getTestRange(), true, self.opt.make_cache)
   
end

function TrainingExpert:batchEpochTrain(epoch, dataset)
   
   self.optim_state.learning_rate = epoch >= 12 and self.params.learning_rate / 10 or self.params.learning_rate

   x, dl_dx = self.network:getModelParameters()

   local function feval()
      return self.network:evaluation(x, dl_dx, self.inputs, self.targets)
   end

   local err_tr = 0
   local err_tr_cnt = 0
   local t = 1

   local indexes = torch.range(1, self.dataset_size/self.train_batch_size):totable()
   local s = self.dataset_size - self.train_batch_size
   xlua.progress(1,#indexes)
   for i, idx in ipairs(indexes) do
      if i % 100 == 0 then
         xlua.progress(i,#indexes)
      end
      
      if self.opt.wait then
         if i % self.opt.wait_batchs == 0 then
            os.execute("sleep " .. self.opt.wait_time)
         end
      end

      t = (idx-1) * self.train_batch_size + 1
      self.inputs, self.targets = trainingBatch(dataset, t, self.train_batch_size, self.network.params.ws)

      --_, fs = self.optim(feval, x, self.optim_state)
      for i = 1,#params do
         grads[i]:zero()
      end
      
      self.network.net:forward(self.inputs)
      local err = self.network.criterion:forward(self.network.net.output, self.targets)
      if err >= 0 and err < 100 then
         err_tr = err_tr + err
         err_tr_cnt = err_tr_cnt + 1
      else
         print(('WARNING! err=%f'):format(err))
      end

      self.network.criterion:backward(self.network.net.output, self.targets)
      self.network.net:backward(self.inputs, self.network.criterion.gradInput)

      for i = 1,#params do
         momentums[i]:mul(self.optim_state.momentum):add(-self.optim_state.learning_rate, grads[i])
         params[i]:add(momentums[i])
      end
      
      --local err = fs[1]
      
      --if err >= 0 and err < 100 then
      --   err_tr = err_tr + err
      --   err_tr_cnt = err_tr_cnt + 1
      --else
      --   print(('WARNING! err=%f'):format(err))
      --   if err ~= err then
      --      os.exit()
      --   end
      --end
   end
   xlua.progress(#indexes, #indexes)
   return err_tr / err_tr_cnt

end