require 'torch'

local FastMatcher = torch.class('FastMatcher')

function FastMatcher:__init(network)
   self.network = network
end

function FastMatcher:computeMatchingCost(x_batch, disp_max, directions)
   local desc_l, desc_r = self:getDescriptors(x_batch)

   -- Initialize the output with the largest matching cost
   -- at each possible disparity ('1')
   local output = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4)):fill(1)

   -- Compute the matching cost at each possible disparity
   for _, m in pairs(self.network.net.modules[2].modules) do
      if torch.typename(m) == "nn.DotProduct2" then
         m:computeMatchingCost(desc_l, desc_r, output[{{1}}], output[{{2}}])
      end
   end

      -- Fix the borders of the obtained map
   self.network.fixBorder(output[{{1}}], -1, self.network.params.ws)
   self.network.fixBorder(output[{{2}}], 1, self.network.params.ws)

   return output
end

function FastMatcher:getDescriptors(x_batch)

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