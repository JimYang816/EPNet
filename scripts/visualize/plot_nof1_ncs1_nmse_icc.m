function [fig, ax] = plot_nof1_ncs1_nmse_icc(runDir, figureWidth)
%PLOT_NOF1_NCS1_NMSE_ICC 单栏、单坐标系中的双信道初始/修正 CSI NMSE。
%   addpath('scripts/visualize');
%   [fig, ax] = plot_nof1_ncs1_nmse_icc;
%   默认读取 configs/evaluation/ber_comparison.json 的 output 目录。
%   同一坐标系绘制 NOF1、NCS1 各自的初始 CSI 和 CRU-Net 修正 CSI。
%   颜色区分信道，虚线/实线区分初始/修正 CSI，四种标记便于灰度阅读。
%   直接使用报告中在线性域求均值后转为 dB 的 NMSE，不再求均值或取对数。
%   默认宽度 3.5 in，不保存；可用第二个参数等比调整宽度。

projectDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if nargin < 1 || isempty(runDir)
    config = jsondecode(fileread(fullfile(projectDir, ...
        'configs', 'evaluation', 'ber_comparison.json')));
    runDir = fullfile(projectDir, config.output);
end
if nargin < 2 || isempty(figureWidth)
    figureWidth = 3.5;
end
validateattributes(figureWidth, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});

channels = {'NOF1', 'NCS1'};
nmseFields = {'mean_nmse_db', 'corrected_mean_nmse_db'};
states = {'Initial CSI', 'Refined CSI'};
colors = [36, 119, 179; 180, 42, 85] / 255;
markers = {'o', 's'; '^', 'd'};
lineStyles = {'--', '-'};
snr = cell(1, numel(channels));
nmse = cell(1, numel(channels));
for c = 1:numel(channels)
    metricsFile = fullfile(runDir, lower(channels{c}), 'metrics.json');
    assert(isfile(metricsFile), '找不到指标文件：%s', metricsFile);
    report = jsondecode(fileread(metricsFile));
    assert(isfield(report, 'complete') && report.complete && ...
        isfield(report, 'input_view') && strcmp(report.input_view, 'compensated') && ...
        isfield(report, 'channel') && strcmp(report.channel, channels{c}), ...
        '需要 %s 已完成的 compensated CSI 指标报告：%s', channels{c}, metricsFile);
    assert(isfield(report, 'nmse') && all(isfield(report.nmse, nmseFields)), ...
        '报告缺少初始或修正后的平均 NMSE：%s', metricsFile);
    snr{c} = report.snr_db(:);
    assert(numel(snr{c}) >= 2 && all(isfinite(snr{c})) && ...
        all(diff(snr{c}) > 0), 'SNR 必须有限且严格递增：%s', metricsFile);
    nmse{c} = zeros(numel(snr{c}), numel(nmseFields));
    for k = 1:numel(nmseFields)
        values = report.nmse.(nmseFields{k});
        values = values(:);
        assert(numel(values) == numel(snr{c}) && isreal(values) && all(isfinite(values)), ...
            '平均 NMSE 长度或数值无效：%s / %s', metricsFile, nmseFields{k});
        nmse{c}(:, k) = values;
    end
end
assert(isequal(snr{1}, snr{2}), 'NOF1 与 NCS1 的 SNR 网格必须一致。');

scale = figureWidth / 3.5;
figureHeight = 2.6 * scale;
fig = figure('Color', 'white', 'Name', 'NOF1 / NCS1 Mean CSI NMSE-SNR', ...
    'NumberTitle', 'off', 'WindowStyle', 'normal', 'Units', 'inches', ...
    'Position', [1, 1, figureWidth, figureHeight], ...
    'PaperUnits', 'inches', 'PaperPositionMode', 'manual', ...
    'PaperPosition', [0, 0, figureWidth, figureHeight], ...
    'PaperSize', [figureWidth, figureHeight], 'InvertHardcopy', 'off');
ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [0.18, 0.19, 0.79, 0.62], ...
    'PositionConstraint', 'innerposition');
hold(ax, 'on');
handles = gobjects(1, 4);
for c = 1:numel(channels)
    for k = 1:numel(nmseFields)
        index = (c - 1) * 2 + k;
        handles(index) = plot(ax, snr{c}, nmse{c}(:, k), ...
            'LineStyle', lineStyles{k}, 'Color', colors(c, :), ...
            'Marker', markers{c, k}, 'MarkerSize', 4 * scale, ...
            'MarkerFaceColor', 'white', 'LineWidth', 1.1 * scale, ...
            'DisplayName', sprintf('%s: %s', channels{c}, states{k}));
    end
end
allNmse = vertcat(nmse{:});
padding = max(0.05 * (max(allNmse(:)) - min(allNmse(:))), 0.2);
yLimits = [min(allNmse(:)) - padding, max(allNmse(:)) + padding];
xPadding = 0.04 * (snr{1}(end) - snr{1}(1));
set(ax, 'FontName', 'Arial', 'FontSize', 8 * scale, ...
    'Box', 'on', 'LineWidth', 0.7 * scale, 'Layer', 'top', 'TickDir', 'out', ...
    'XTick', snr{1}, 'XLim', [snr{1}(1) - xPadding, snr{1}(end) + xPadding], ...
    'YScale', 'linear', 'YLim', yLimits, ...
    'YTick', 2 * ceil(yLimits(1) / 2):2:2 * floor(yLimits(2) / 2), ...
    'XGrid', 'on', 'YGrid', 'on', 'GridColor', [0.55, 0.55, 0.55], 'GridAlpha', 0.20);
xlabel(ax, 'SNR (dB)', 'FontName', 'Arial', 'FontSize', 9 * scale);
ylabel(ax, 'Mean CSI NMSE (dB)', 'FontName', 'Arial', 'FontSize', 9 * scale);
sharedLegend = legend(ax, handles, 'NumColumns', 2, 'Interpreter', 'none', ...
    'FontName', 'Arial', 'FontSize', 7.2 * scale, 'Box', 'off', ...
    'Units', 'normalized', 'Location', 'none');
drawnow;
legendPosition = sharedLegend.Position;
legendPosition(1:2) = [(1 - legendPosition(3)) / 2, 0.985 - legendPosition(4)];
sharedLegend.Position = legendPosition;
drawnow;
end
