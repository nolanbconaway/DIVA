function old_vals = humbleTeach(old_vals,humble_range)

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% this takes in any matrix and clips values at the values specified in
% value-range

% 
% old_vals = values to be clipped, if out of range
% labels = category labels to provide the min and max values
% 
% labels=[min max]; % uncomment and fill in for arbitrary clipping

% (.1^10) prevents NaN from showing up
lower_lim = min(humble_range) + (.1^10);
upper_lim = max(humble_range) - (.1^10);

old_vals(old_vals<lower_lim)=lower_lim;
old_vals(old_vals>upper_lim)=upper_lim;

            
            