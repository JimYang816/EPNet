function [fig, ax] = plot_nof1_ncs1_ber_icc(runDir, figureWidth)
%PLOT_NOF1_NCS1_BER_ICC 双栏 BER 主对比：左 NOF1、右 NCS1，共用图例。
%   在项目根目录运行：
%     addpath('scripts/visualize');
%     [fig, ax] = plot_nof1_ncs1_ber_icc;
%   默认读取 configs/evaluation/ber_comparison.json 的 output 目录。
%   也可传入包含 nof1/metrics.json 和 ncs1/metrics.json 的结果根目录。
%   默认按论文双栏宽度 7.16 in 绘制，共用图例置于两幅图上方。
%   两个信道使用相同 BER 坐标范围，颜色、标记和线型区分算法。
%   默认不保存；第二个参数可调整整图宽度 (in)，各元素同步缩放。
%   可在项目根目录导出等尺寸矢量 PDF 或 600 dpi PNG：
%     print(fig, 'outputs/paper/ber_snr_icc', '-dpdf', '-painters');
%     print(fig, 'outputs/paper/ber_snr_icc', '-dpng', '-r600');

projectDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if nargin < 1 || isempty(runDir)
    configFile = fullfile(projectDir, 'configs', 'evaluation', 'ber_comparison.json');
    config = jsondecode(fileread(configFile));
    runDir = fullfile(projectDir, config.output);
end
if nargin < 2 || isempty(figureWidth)
    figureWidth = 7.16;
end
validateattributes(figureWidth, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});

channels = {'NOF1', 'NCS1'};
algorithmIds = {'single_tap_mmse', 'full_matrix_lmmse', 'mmse_dfe', ...
    'udnet', 'oamp_net2', 'icr_mmse'};
algorithmNames = {'Single-tap LMMSE', 'Full-matrix LMMSE', 'MMSE-DFE', ...
    'UDNet', 'OAMP-Net2', 'CRU-Net'};
colors = [102, 112, 133; 36, 119, 179; 0, 151, 167; ...
    223, 128, 40; 21, 132, 91; 180, 42, 85] / 255;
markers = {'o', 's', 'x', '^', 'd', 'p'};
lineStyles = {'-', '--', '-.', '--', '-.', '-'};

% 先核对两个报告，再创建图窗；按 identifier 匹配曲线。
snr = cell(1, numel(channels));
ber = cell(1, numel(channels));
for c = 1:numel(channels)
    metricsFile = fullfile(runDir, lower(channels{c}), 'metrics.json');
    assert(isfile(metricsFile), '找不到指标文件：%s', metricsFile);
    report = jsondecode(fileread(metricsFile));
    assert(isfield(report, 'complete') && report.complete && ...
        isfield(report, 'input_view') && strcmp(report.input_view, 'compensated') && ...
        isfield(report, 'channel') && strcmp(report.channel, channels{c}), ...
        '需要 %s 已完成的 compensated CSI 指标报告：%s', channels{c}, metricsFile);
    snr{c} = report.snr_db(:);
    assert(numel(snr{c}) >= 2 && all(isfinite(snr{c})) && ...
        all(diff(snr{c}) > 0), 'SNR 必须有限且严格递增：%s', metricsFile);

    curveFields = fieldnames(report.curves);
    curveIds = cellfun(@(key) report.curves.(key).identifier, curveFields, ...
        'UniformOutput', false);
    ber{c} = zeros(numel(snr{c}), numel(algorithmIds));
    for k = 1:numel(algorithmIds)
        index = find(strcmp(curveIds, algorithmIds{k}));
        assert(isscalar(index), '算法 %s 必须恰好出现一次：%s', ...
            algorithmIds{k}, metricsFile);
        values = report.curves.(curveFields{index}).ber(:);
        assert(numel(values) == numel(snr{c}) && all(isfinite(values)) && ...
            all(values >= 0 & values <= 1), 'BER 长度或数值无效：%s / %s', ...
            metricsFile, algorithmIds{k});
        ber{c}(:, k) = values;
    end
end
assert(isequal(snr{1}, snr{2}), 'NOF1 与 NCS1 的 SNR 网格必须一致。');

allBer = vertcat(ber{:});
positiveBer = allBer(allBer > 0);
assert(~isempty(positiveBer), '全部 BER 为零，无法绘制对数 BER 坐标。');
% 零 BER 不替换为人为下限，在对数图上留空并提示。
if any(allBer(:) == 0)
    warning('EPNet:ZeroBER', '零 BER 点在对数坐标上留空，未使用人为正数替代。');
end
figureHeight = figureWidth * 3.2 / 7.16;
scale = figureWidth / 7.16;
fig = figure('Color', 'white', 'Name', 'NOF1 / NCS1 BER-SNR', ...
    'NumberTitle', 'off', 'WindowStyle', 'normal', 'Units', 'inches', ...
    'Position', [1, 1, figureWidth, figureHeight], ...
    'PaperUnits', 'inches', 'PaperPositionMode', 'manual', ...
    'PaperPosition', [0, 0, figureWidth, figureHeight], ...
    'PaperSize', [figureWidth, figureHeight], 'InvertHardcopy', 'off');
ax = gobjects(1, numel(channels));
curveHandles = gobjects(numel(channels), numel(algorithmIds));
positions = [0.085, 0.205, 0.395, 0.605; 0.585, 0.205, 0.395, 0.605];
logLimits = log10([min(positiveBer), max(positiveBer)]);
padding = max(0.06 * diff(logLimits), 0.02);
yLimits = 10 .^ (logLimits + [-padding, padding]);
yTicks = 10 .^ (ceil(log10(yLimits(1))):floor(log10(yLimits(2))));

for c = 1:numel(channels)
    ax(c) = axes('Parent', fig, 'Units', 'normalized', 'Position', positions(c, :), ...
        'PositionConstraint', 'innerposition');
    hold(ax(c), 'on');
    set(ax(c), 'YScale', 'log');
    for k = 1:numel(algorithmIds)
        values = ber{c}(:, k);
        values(values == 0) = NaN;
        curveHandles(c, k) = semilogy(ax(c), snr{c}, values, ...
            'Color', colors(k, :), 'LineStyle', lineStyles{k}, ...
            'Marker', markers{k}, 'MarkerSize', 4.2 * scale, ...
            'MarkerFaceColor', 'white', 'LineWidth', 1.15 * scale, ...
            'DisplayName', algorithmNames{k});
    end
    xlabel(ax(c), {'SNR (dB)', sprintf('(%c) %s', 'a' + c - 1, channels{c})}, ...
        'FontName', 'Arial', 'FontSize', 9 * scale);
    ylabel(ax(c), 'BER', 'FontName', 'Arial', 'FontSize', 9 * scale);
    % 横轴稍留余量，保证边界处的标记完整显示。
    xPadding = 0.05 * (snr{c}(end) - snr{c}(1));
    set(ax(c), 'FontName', 'Arial', 'FontSize', 8 * scale, ...
        'Box', 'on', 'LineWidth', 0.7 * scale, 'Layer', 'top', 'TickDir', 'out', ...
        'TickLength', [0.01, 0.025], ...
        'XTick', snr{c}, 'XLim', [snr{c}(1) - xPadding, snr{c}(end) + xPadding], ...
        'YLim', yLimits, 'YTick', yTicks, ...
        'XGrid', 'on', 'YGrid', 'on', 'YMinorGrid', 'on', ...
        'GridColor', [0.55, 0.55, 0.55], 'GridAlpha', 0.20, 'MinorGridAlpha', 0.10);
end
sharedLegend = legend(ax(1), curveHandles(1, :), algorithmNames, ...
    'Orientation', 'horizontal', 'NumColumns', 3, 'Interpreter', 'none', ...
    'FontName', 'Arial', 'FontSize', 8 * scale, 'Box', 'off', ...
    'Units', 'normalized', 'Location', 'none');
drawnow;
legendPosition = sharedLegend.Position;
legendPosition(1:2) = [(1 - legendPosition(3)) / 2, 0.985 - legendPosition(4)];
sharedLegend.Position = legendPosition;
linkaxes(ax, 'xy');
drawnow;
end
