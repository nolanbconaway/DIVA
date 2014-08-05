function values = clipvalues(values,value_range)

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% this takes in any matrix and clips values at the values specified in
% value-range

% 
% old_vals = values to be clipped, if out of range
% labels = category labels to provide the min and max values
% 
% labels=[min max]; % uncomment and fill in for arbitrary clipping

values(values<value_range(1))=value_range(1);
values(values>value_range(2))=value_range(2);

            
            
