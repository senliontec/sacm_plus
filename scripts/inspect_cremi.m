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
% After each image is shown, click the figure window and use:
%   Enter : next slice (wraps to slice 1 after the last slice)
%   b     : previous slice
%   1     : overlay derived neuron-boundary mask (red)
%   2     : overlay official synaptic-cleft mask (green)
%   3     : overlay both masks
%   q     : quit
%
% The displayed image is the grayscale raw slice with the selected mask(s)
% overlaid. If the HDF has no neuron_ids or no clefts (for example CREMI
% test volumes A+/B+/C+), the unavailable masks are omitted with a warning.
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
cleftPath = '/volumes/labels/clefts';
if ~dataset_exists(hdfFile, rawPath)
    error('inspect_cremi:noRaw', '%s has no %s dataset', hdfFile, rawPath);
end
hasLabels = dataset_exists(hdfFile, nidPath);
hasClefts = dataset_exists(hdfFile, cleftPath);

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
end
if hasClefts
    cleftInfo = h5info(hdfFile, cleftPath);
    cleftDims = double(cleftInfo.Dataspace.Size);
    if numel(cleftDims) == 3 && ~isequal(cleftDims, [W, H, Z])
        warning('inspect_cremi:cleftShape', ...
            '%s shape [%s] differs from raw [W,H,Z]=[%d,%d,%d]', ...
            cleftPath, num2str(cleftDims), W, H, Z);
    end
end
if ~hasLabels && ~hasClefts
    warning('inspect_cremi:noMasks', ...
        '%s has neither %s nor %s; showing raw slices only.', ...
        hdfFile, nidPath, cleftPath);
end

fig = figure( ...
    'Name', sprintf('CREMI: %s', fileInfo.name), ...
    'Color', 'w', ...
    'ToolBar', 'figure', ...
    'MenuBar', 'figure');
ax = axes(fig);
z = startSlice;
interactive = usejava('desktop');

if hasLabels
    maskMode = 'boundary';
elseif hasClefts
    maskMode = 'clefts';
else
    maskMode = 'raw';
end

if interactive
    set(fig, 'KeyPressFcn', @on_key);
    fprintf(['Keys (click the figure first): Enter=next, b=back, ' ...
        '1=boundary, 2=clefts, 3=both, q=quit\n']);
    fprintf('Use the figure toolbar magnifier to zoom.\n');
    figure(fig);
    drawnow;
else
    fprintf('Headless mode: rendering one slice (no keyboard input).\n');
end

while true
    rawSlice = raw(:, :, z);
    boundary = [];
    clefts = [];

    boundaryView = [];
    if hasLabels
        labels = read_label_slice(hdfFile, nidPath, z, H, W);
        boundary = false(H, W);
        boundary(:, 1:end-1) = labels(:, 1:end-1) ~= labels(:, 2:end);
        boundary(1:end-1, :) = boundary(1:end-1, :) | ...
            (labels(1:end-1, :) ~= labels(2:end, :));
        % Display-only thickening; exported masks remain one pixel wide.
        boundaryView = dilate_mask(boundary);
    end
    if hasClefts
        cleftLabels = read_label_slice(hdfFile, cleftPath, z, H, W);
        clefts = (cleftLabels ~= 0) & (cleftLabels ~= intmax('uint64'));
    end

    overlay = make_overlay(rawSlice, boundaryView, clefts, maskMode);
    titleText = sprintf('slice %d/%d (z=%d) | %s', ...
        z, Z, z - 1, maskMode);
    statsText = sprintf('slice %d/%d (HDF5 z=%d), mode=%s', ...
        z, Z, z - 1, maskMode);
    if hasLabels
        titleText = sprintf('%s | B %.2f%%', ...
            titleText, 100 * mean(boundary(:)));
        statsText = sprintf('%s, boundary %.3f%%', ...
            statsText, 100 * mean(boundary(:)));
    end
    if hasClefts
        titleText = sprintf('%s | C %.2f%%', ...
            titleText, 100 * mean(clefts(:)));
        statsText = sprintf('%s, clefts %.3f%%', ...
            statsText, 100 * mean(clefts(:)));
    end
    fprintf('%s\n', statsText);

    if all(rawSlice(:) == 0)
        warning('inspect_cremi:blankSlice', ...
            'raw slice %d is all zero in the source HDF.', z);
    end

    image(ax, overlay);
    axis(ax, 'image');
    axis(ax, 'off');
    title(ax, titleText, 'Interpreter', 'none');
    drawnow;

    if ~interactive
        break;
    end

    key = wait_for_key(fig);
    if ~isvalid(fig)
        break;
    end
    key = lower(strtrim(key));

    switch key
        case {'return', 'enter'}
            if z >= Z
                z = 1;
                fprintf('回到第 1 个 slice。\n');
            else
                z = z + 1;
            end
        case 'q'
            break;
        case 'b'
            z = max(1, z - 1);
        case '1'
            if hasLabels
                maskMode = 'boundary';
            else
                warning('inspect_cremi:noBoundary', ...
                    'This HDF has no neuron_ids; boundary unavailable.');
            end
        case '2'
            if hasClefts
                maskMode = 'clefts';
            else
                warning('inspect_cremi:noClefts', ...
                    'This HDF has no clefts; cleft mask unavailable.');
            end
        case '3'
            if hasLabels && hasClefts
                maskMode = 'both';
            else
                warning('inspect_cremi:maskMode', ...
                    'Both masks are required for mode 3.');
            end
        otherwise
            % Ignore unknown keys; keep the current slice/mode.
    end
end

if nargin >= 3 && ~isempty(previewPath)
    previewPath = char(previewPath);
    exportgraphics(fig, previewPath, 'Resolution', 150);
    fprintf('Saved preview: %s\n', previewPath);
end
end


function key = wait_for_key(fig)
% Block until the figure receives a key press (or is closed).
setappdata(fig, 'lastKey', '');
try
    uiwait(fig);
catch
    key = 'q';
    return;
end
if isvalid(fig)
    key = getappdata(fig, 'lastKey');
else
    key = 'q';
end
end


function on_key(src, evt)
% Store the key and release uiwait.
key = evt.Key;
if isempty(key)
    key = evt.Character;
end
key = lower(key);
% Numeric keypad may report numpad1/numpad2/numpad3.
key = regexprep(key, '^numpad([123])$', '$1');
setappdata(src, 'lastKey', key);
uiresume(src);
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


function mask = dilate_mask(mask)
% One-pixel four-neighbour dilation for display only.
mask(2:end, :) = mask(2:end, :) | mask(1:end-1, :);
mask(1:end-1, :) = mask(1:end-1, :) | mask(2:end, :);
mask(:, 2:end) = mask(:, 2:end) | mask(:, 1:end-1);
mask(:, 1:end-1) = mask(:, 1:end-1) | mask(:, 2:end);
end


function overlay = make_overlay(rawSlice, boundary, clefts, mode)
% Grayscale background with red boundary and green cleft overlays.
lo = double(min(rawSlice(:)));
hi = double(max(rawSlice(:)));
if hi > lo
    raw01 = (double(rawSlice) - lo) / (hi - lo);
else
    raw01 = zeros(size(rawSlice));
end
overlay = repmat(raw01, [1, 1, 3]);

showBoundary = ~isempty(boundary) && ...
    (strcmp(mode, 'boundary') || strcmp(mode, 'both'));
showClefts = ~isempty(clefts) && ...
    (strcmp(mode, 'clefts') || strcmp(mode, 'both'));

if showBoundary
    overlay(:, :, 1) = overlay(:, :, 1) .* ~boundary + double(boundary);
    overlay(:, :, 2) = overlay(:, :, 2) .* ~boundary + 0.15 * double(boundary);
    overlay(:, :, 3) = overlay(:, :, 3) .* ~boundary + 0.15 * double(boundary);
end
if showClefts
    overlay(:, :, 1) = overlay(:, :, 1) .* ~clefts + 0.15 * double(clefts);
    overlay(:, :, 2) = overlay(:, :, 2) .* ~clefts + double(clefts);
    overlay(:, :, 3) = overlay(:, :, 3) .* ~clefts + 0.15 * double(clefts);
end
end
