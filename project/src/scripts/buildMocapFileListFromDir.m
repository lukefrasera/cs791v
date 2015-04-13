function file_list_mat = buildMocapFileListFromDir(dirname)
% Mocap File Matrix Builder
% Author Luke Fraser
olddir = cd(dirname);
asf_files = dir('*.asf')';
amc_files = dir('*.amc')';
cd(olddir);
idx = 1;
result = cell(size(amc_files,1),2);
for asf_file = asf_files
    % Find the asf File that matches
    [path, asf_file_str, ext] = fileparts(asf_file.name);
    amc_idx = 1;
    for amc_file = amc_files
        amc_file_split = strsplit(amc_file.name, '_');
        amc_file_str = amc_file_split(1);
        if strcmp(asf_file_str, amc_file_str)
            result{idx,1} = asf_file.name;
            result{idx,2} = amc_file.name;
            amc_files(:,amc_idx) = [];
            amc_idx = amc_idx - 1;
            idx = idx + 1;
        end
        amc_idx = amc_idx + 1;
    end
end

file_list_mat = result;