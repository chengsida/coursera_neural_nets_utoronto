function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
   visible_data = sample_bernoulli(visible_data);
   origin_hidden_prob = visible_state_to_hidden_probabilities(rbm_w,visible_data);
   origin_hidden=sample_bernoulli(origin_hidden_prob);
   origin_gradient = configuration_goodness_gradient(visible_data,origin_hidden);
   update_visible_prob = hidden_state_to_visible_probabilities(rbm_w,origin_hidden);
   update_visible = sample_bernoulli(update_visible_prob);
   update_hidden_prob = visible_state_to_hidden_probabilities(rbm_w,update_visible);
   update_hidden = sample_bernoulli(update_hidden_prob);
   update_gradient = configuration_goodness_gradient(update_visible,update_hidden_prob);
   ret = origin_gradient - update_gradient;
   
   
end
