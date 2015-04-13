function subset = greedySkeletonSubset(skeleton_set)
% Greedy Skeleton Subset Method
% Author Luke Fraser

rows = size(skeleton_set, 1);
idx_1 = 0;
idx_2 = 0;
for i = 1:size(skeleton_set,1)
    if size(skeleton_set, 1) < 1
        break
    end
    idx_1 = idx_1 + 1;
    A = skeleton_set(1, :);
    skeleton_set(1, :) = [];
    result(idx_1,:) = A;
    idx_2 = 0;
    for skeleton_2 = skeleton_set.'
        idx_2 = idx_2 + 1;
        B = skeleton_2';
        Difference = bsxfun(@minus, A, B);
        Difference = reshape(Difference, 3, []).^2;
        Difference = Difference.';
        dist = Difference * ones(3,1);
        dist = dist.^(.5);
        sum = ones(1,size(dist,1)) * dist;
        if sum < 1.8
            skeleton_set(idx_2, :) = [];
            idx_2 = idx_2 - 1;
        end
    end
end
subset = result;
