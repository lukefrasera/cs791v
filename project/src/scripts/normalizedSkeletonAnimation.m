function skeletonAnim = normalizedSkeletonAnimation(asf_file, amc_file)
% Skeleton Normalize and Build Set
% Takes an ASF file and computes a matrix of all Animation Clips Normalized
% Author Luke Fraser

skel = acclaimReadSkel(asf_file);
[channels, skel] = acclaimLoadChannels(amc_file, skel);

% Normalize skeleton
skel.tree(1).rotInd = [0 0 0];
skel.tree(1).posInd = [0 0 0];

% For each frame get 3D positions
rows = size(channels, 1);
result = zeros(rows, 31*3);

for i = 1:rows
    result(i,:) = reshape(skel2xyz(skel,channels(i,:)).', [1 31*3]);
end
skeletonAnim = result;