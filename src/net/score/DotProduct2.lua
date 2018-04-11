local DotProduct2, parent = torch.class('nn.DotProduct2', 'nn.Module')

function DotProduct2:__init()
   parent.__init(self)
   self.gradInput = torch.CudaTensor()
   self.tmp = torch.CudaTensor()
   self.output = torch.CudaTensor()
end

local function sliceInput(input)
   local sizes = torch.LongStorage{input:size(1) / 2, input:size(2), input:size(3), input:size(4)}
   local strides = torch.LongStorage{input:stride(1) * 2, input:stride(2), input:stride(3), input:stride(4)}

   local input_L = torch.CudaTensor(input:storage(), 1, sizes, strides)
   local input_R = torch.CudaTensor(input:storage(), input:stride(1) + 1, sizes, strides)

   return input_L, input_R
end

function DotProduct2:updateOutput(input)
   local input_L, input_R = sliceInput(input)
   self.tmp:resizeAs(input_L)
   self.tmp:cmul(input_L, input_R)
   self.output:sum(self.tmp, 2)
   return self.output
end

function DotProduct2:updateGradInput(input, gradOutput)
   gradOutput:cuda()
   input:cuda()
   self.gradInput:resizeAs(input)
   local input_L, input_R = sliceInput(input)
   local gradInput_L, gradInput_R = sliceInput(self.gradInput)
   gradInput_L:cmul(input_R, gradOutput:expandAs(input_R):cuda())
   gradInput_R:cmul(input_L, gradOutput:expandAs(input_L):cuda())
   return self.gradInput
end

function DotProduct2:computeMatchingCost(input_L, input_R, output_L, output_R)
   adcensus.StereoJoin(input_L, input_R, output_L, output_R)
end

