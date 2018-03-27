require 'torch'

local MatchingExpert = torch.class('MatchingExpert')

function MatchingExpert:__init()
   
end

function MatchingExpert:match(network, x_batch, disp_max, directions)
   return network.matcher:computeMatchingCost(x_batch, disp_max, directions)
end
