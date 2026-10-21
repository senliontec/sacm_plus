function inspect_cremi(hdfFile, startSlice, previewPath)
%INSPECT_CREMI Browse CREMI HDF5 slices with the boundary mask overlaid.
%
% Usage:
%   inspect_cremi
%   inspect_cremi('/path/to/sample_A_20160501.hdf')
%   inspect_cremi('/path/to/sample_A_20160501.hdf', 42)
%   inspect_cremi('/path/to/sample_A_20160501.hdf', 1, 'last_slice.png')
%
% startSlice is 1-based; startSlice=1 corresponds to HDF5 z=0. If omitted,
% browsing starts at slice 1.
%
% After each image is shown:
%   Enter : next slice (wraps to slice 1 after the last slice)
%   b     : previous slice
%   q     : quit
%
% The displayed image is the grayscale raw slice with the derived
% neuron-boundary mask drawn in red. If the HDF has no neuron_ids
% (CREMI test volumes A+/B+/C+), only the raw slice is shown.
%
% If previewPath is supplied, the currently displayed slice is saved when
% you quit. No source HDF is modified.
%
% h5info/h5read expose HDF5 [Z,H,W] as MATLAB [W,H,Z]; this is handled
% below, and label slices are read as hyperslabs to avoid loading the
% full uint64 volume.

if nargin < 1 || isempty(hdfFile)
    [name, folder] = uigetfile( ...
        {'*.hdf;*.h5', 'HDF5 files'; '*.*', 'All files'}, ...
        'Select a CREMI HDF5 file');
    if isequal(name, 0)
        return;
    end
    hdfFile = fullfile(folder, name);
end
hdfFile = char(hdfFile);

if ~isfile(hdfFile)
    error('inspect_cremi:fileNotFound', 'File not found: %s', hdfFile);
end

fileInfo = dir(hdfFile);
fprintf('CREMI HDF5: %s (%.1f MB)\n', hdfFile, fileInfo.bytes / 1e6);
info = h5info(hdfFile);
fprintf('\nDatasets in %s:\n', hdfFile);
print_group(info);

rawPath = '/volumes/raw';
nidPath = '/volumes/labels/neuron_ids';
if ~dataset_exists(hdfFile, rawPath)
    error('inspect_cremi:noRaw', '%s has no %s dataset', hdfFile, rawPath);
end
hasLabels = dataset_exists(hdfFile, nidPath);

rawInfo = h5info(hdfFile, rawPath);
dims = double(rawInfo.Dataspace.Size);
if numel(dims) ~= 3
    error('inspect_cremi:badShape', ...
        '%s: expected [Z,H,W], got [%s]', rawPath, num2str(dims));
end

% h5info/h5read expose MATLAB order: CREMI's HDF5 [Z,H,W] becomes
% MATLAB [W,H,Z].
W = dims(1);
H = dims(2);
Z = dims(3);
fprintf('\nraw dimensions in MATLAB order: [W,H,Z] = [%d,%d,%d]\n', W, H, Z);
fprintf('equivalent HDF5 order: [Z,H,W] = [%d,%d,%d]\n', Z, H, W);

% Load raw once; it is ~195 MB for a 1250x1250x125 volume.
raw = squeeze(h5read(hdfFile, rawPath));
if ndims(raw) ~= 3
    error('inspect_cremi:rawRead', ...
        'Unexpected raw array after squeeze: [%s]', num2str(size(raw)));
end
rawSize = size(raw);
if isequal(rawSize, [H, W, Z])
    % already [H,W,Z]
elseif isequal(rawSize, [W, H, Z])
    raw = permute(raw, [2, 1, 3]);
elseif isequal(rawSize, [Z, H, W])
    raw = permute(raw, [2, 3, 1]);
elseif isequal(rawSize, [Z, W, H])
    raw = permute(raw, [3, 2, 1]);
else
    error('inspect_cremi:rawOrder', ...
        'Cannot map raw size [%s] to [H,W,Z]=[%d,%d,%d]', ...
        num2str(rawSize), H, W, Z);
end

if nargin < 2 || isempty(startSlice)
    startSlice = 1;
end
startSlice = double(startSlice);
if ~isscalar(startSlice) || startSlice < 1 || startSlice > Z || ...
        startSlice ~= round(startSlice)
    error('inspect_cremi:badSlice', ...
        'startSlice must be an integer in [1, %d]', Z);
end

if hasLabels
    nidInfo = h5info(hdfFile, nidPath);
    nidDims = double(nidInfo.Dataspace.Size);
    if numel(nidDims) == 3 && ~isequal(nidDims, [W, H, Z])
        warning('inspect_cremi:labelShape', ...
            '%s shape [%s] differs from raw [W,H,Z]=[%d,%d,%d]', ...
            nidPath, num2str(nidDims), W, H, Z);
    end
else
    warning('inspect_cremi:noLabels', ...
        '%s has no %s; showing raw slices without a mask (test volume).', ...
        hdfFile, nidPath);
end

fig = figure('Name', sprintf('CREMI: %s', fileInfo.name), 'Color', 'w');
ax = axes(fig);
z = startSlice;

while true
    rawSlice = raw(:, :, z);

    if hasLabels
        labels = read_label_slice(hdfFile, nidPath, z, H, W);
        boundary = false(H, W);
        boundary(:, 1:end-1) = labels(:, 1:end-1) ~= labels(:, 2:end);
        boundary(1:end-1, :) = boundary(1:end-1, :) | ...
            (labels(1:end-1, :) ~= labels(2:end, :));
        overlay = make_overlay(rawSlice, boundary);
        titleText = sprintf('%s | slice %d/%d (HDF5 z=%d) | boundary %.2f%%', ...
            fileInfo.name, z, Z, z - 1, 100 * mean(boundary(:)));
        fprintf('slice %d/%d (HDF5 z=%d): boundary fg %.3f%%\n', ...
            z, Z, z - 1, 100 * mean(boundary(:)));
    else
        overlay = make_overlay(rawSlice, false(H, W));
        titleText = sprintf('%s | slice %d/%d (HDF5 z=%d) | raw only', ...
            fileInfo.name, z, Z, z - 1);
    end

    if all(rawSlice(:) == 0)
        warning('inspect_cremi:blankSlice', ...
            'raw slice %d is all zero in the source HDF.', z);
    end

    image(ax, overlay);
    axis(ax, 'image');
    axis(ax, 'off');
    title(ax, titleText, 'Interpreter', 'none');
    drawnow;

    answer = input(sprintf( ...
        'slice %d/%d  [Enter]=next  [b]=previous  [q]=quit: ', z, Z), 's');
    answer = lower(strtrim(answer));
    if strcmp(answer, 'q')
        break;
    elseif strcmp(answer, 'b')
        z = max(1, z - 1);
    else
        if z >= Z
            z = 1;
            fprintf('回到第 1 个 slice。\n');
        else
            z = z + 1;
        end
    end
end

if nargin >= 3 && ~isempty(previewPath)
    previewPath = char(previewPath);
    exportgraphics(fig, previewPath, 'Resolution', 150);
    fprintf('Saved preview: %s\n', previewPath);
end
end


function tf = dataset_exists(hdfFile, datasetPath)
tf = true;
try
    h5info(hdfFile, datasetPath);
catch
    tf = false;
end
end


function print_group(group, indent)
if nargin < 2
    indent = '';
end
for k = 1:numel(group.Datasets)
    d = group.Datasets(k);
    fprintf('%s%-42s %-12s [%s]\n', indent, d.Name, ...
        d.Datatype.Class, num2str(d.Dataspace.Size));
end
for k = 1:numel(group.Groups)
    fprintf('%s%s/\n', indent, group.Groups(k).Name);
    print_group(group.Groups(k), [indent, '  ']);
end
end


function labels = read_label_slice(hdfFile, datasetPath, z, H, W)
% Read one z-slice without loading the full uint64 label volume.
% h5read starts/counts are 1-based and follow MATLAB order [W,H,Z].
% The second candidate is the fallback for toolboxes that expose the
% HDF5 axis order instead.
candidates = {
    {[1, 1, z], [W, H, 1]}, ...
    {[z, 1, 1], [1, H, W]}
};
lastError = [];
for k = 1:numel(candidates)
    try
        s = h5read(hdfFile, datasetPath, ...
            candidates{k}{1}, candidates{k}{2});
        if numel(s) ~= H * W
            continue;
        end
        s = squeeze(s);
        if isvector(s) && numel(s) == H * W
            s = reshape(s, H, W);
        end
        if size(s, 1) == W && size(s, 2) == H && H ~= W
            s = s.';
        end
        if isequal(size(s), [H, W])
            labels = s;
            return;
        end
    catch ME
        lastError = ME;
    end
end
if isempty(lastError)
    error('inspect_cremi:labelRead', ...
        'Could not read a [%d,%d] slice from %s', H, W, datasetPath);
else
    error('inspect_cremi:labelRead', ...
        'Could not read %s: %s', datasetPath, lastError.message);
end
end


function overlay = make_overlay(rawSlice, boundary)
% Grayscale background with red boundary pixels.
lo = double(min(rawSlice(:)));
hi = double(max(rawSlice(:)));
if hi > lo
    raw01 = (double(rawSlice) - lo) / (hi - lo);
else
    raw01 = zeros(size(rawSlice));
end
overlay = repmat(raw01, [1, 1, 3]);
overlay(:, :, 1) = overlay(:, :, 1) .* ~boundary + double(boundary);
overlay(:, :, 2) = overlay(:, :, 2) .* ~boundary;
overlay(:, :, 3) = overlay(:, :, 3) .* ~boundary;
end
