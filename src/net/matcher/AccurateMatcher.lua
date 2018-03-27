require 'torch'

local AccurateMatcher = torch.class('AccurateMatcher')

function AccurateMatcher:__init(network)
   self.network = network
end

function AccurateMatcher:computeMatchingCost(x_batch, disp_max, directions)
   local desc_l, desc_r = self:getDescriptors(x_batch)

   -- Replace with fully convolutional network with the same weights
   local testDecision = self.network:buildTestNet(self.network.net.modules[2])

   -- Initialize the output with the largest matching cost
   -- at each possible disparity ('1')
   local output = torch.CudaTensor(#directions, disp_max, desc_l:size(3), desc_l:size(4)):fill(1)

   local x2= torch.CudaTensor()
   collectgarbage()
   for _, direction in ipairs(directions) do
      local index = direction == -1 and 1 or 2
      for d = 1,disp_max do
         collectgarbage()
         -- Get the left and right images for this disparity
         local l = desc_l[{{1},{},{},{d,-1}}]
         local r = desc_r[{{1},{},{},{1,-d}}]
         x2:resize(2, r:size(2), r:size(3), r:size(4))
         x2[{{1}}]:copy(l)
         x2[{{2}}]:copy(r)

         -- Compute the matching score
         local score = testDecision:forward(x2)

         -- Copy to the right place in the output tensor
         output[{index,d,{},direction == -1 and {d,-1} or {1,-d}}]:copy(score[{1,1}])
      end
      -- Fix the borders of the obtained map
      self.network:fixBorder(output[{{index}}], direction, self.network.params.ws)
   end
   collectgarbage()
   return output
end

function AccurateMatcher:getDescriptors(x_batch)

   -- Replace with fully convolutional network
   local testDesc = self.network:buildTestNet(self.network.net.modules[1])
   testDesc:clearState()
   -- compute the two image decriptors
   -- we compute them separatly in order to reduce the memory usage
   -- to reduce more memory use forward_and_free
   local output_l = self.network:forwardFree(testDesc, x_batch[{{1}}]:clone()):clone()
   testDesc:clearState()
   local output_r = self.network:forwardFree(testDesc, x_batch[{{2}}]:clone()):clone()
   testDesc:clearState()

   return output_l, output_r

end