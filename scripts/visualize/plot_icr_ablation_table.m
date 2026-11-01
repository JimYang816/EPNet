function [fig, summary] = plot_icr_ablation_table(runDir, figureWidth)
%PLOT_ICR_ABLATION_TABLE A-D 正文单栏表格：双信道平均 BER 与 30 dB BER。
%   [fig, summary] = plot_icr_ablation_table;
%   默认读取当前正式 A-D compensated CSI 测试结果，默认宽度 3.5 in。
%   平均 BER 为 0:5:30 dB 七个 SNR 点的算术均值（线性域、等权）。
%   summary 保存未经显示舍入的数值，表中每列最小值加粗，不自动保存。
%   A/B：固定 CSI；C/D：迭代修正 CSI。
%   A/C：Full-matrix LMMSE；B/D：MMSE Unfolding，D 为完整 CRU-Net。

if nargin < 1
    runDir = [];
end
if nargin < 2 || isempty(figureWidth)
    figureWidth = 3.5;
end
validateattributes(figureWidth, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});
[reports, runDir] = load_icr_ablation_reports(runDir);
groups = {'A'; 'B'; 'C'; 'D'};
values = zeros(4, 4);
for c = 1:numel(reports)
    report = reports{c};
    index30 = find(report.snr_db == 30);
    assert(isscalar(index30), '报告必须恰好包含一个 30 dB SNR 点。');
    for k = 1:numel(groups)
        ber = report.curves.(groups{k}).ber(:);
        values(k, 2 * c - 1) = mean(ber);
        values(k, 2 * c) = ber(index30);
    end
end
summary = table(groups, values(:, 1), values(:, 2), values(:, 3), values(:, 4), ...
    'VariableNames', {'Variant', 'NOF1_MeanBER', 'NOF1_BER30dB', ...
    'NCS1_MeanBER', 'NCS1_BER30dB'});
summary.Properties.Description = 'Compensated test CSI; equal-weight mean BER over 0:5:30 dB';

scale = figureWidth / 3.5;
figureHeight = 2.05 * scale;
fig = figure('Color', 'white', 'Name', 'A-D Ablation: Mean BER / 30 dB BER', ...
    'NumberTitle', 'off', 'WindowStyle', 'normal', 'Units', 'inches', ...
    'Position', [1, 1, figureWidth, figureHeight], ...
    'PaperUnits', 'inches', 'PaperPositionMode', 'manual', ...
    'PaperPosition', [0, 0, figureWidth, figureHeight], ...
    'PaperSize', [figureWidth, figureHeight], 'InvertHardcopy', 'off', ...
    'UserData', struct('summary', summary, 'source_run', runDir));
ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [0.02, 0.02, 0.96, 0.96], ...
    'XLim', [0, 1], 'YLim', [0, 1], 'Visible', 'off');
hold(ax, 'on');
centres = [0.065, 0.255, 0.465, 0.675, 0.885];
label(ax, centres(1), 0.845, 'Variant', 7.5 * scale, 'bold');
label(ax, mean(centres(2:3)), 0.90, 'NOF1', 8.5 * scale, 'bold');
label(ax, mean(centres(4:5)), 0.90, 'NCS1', 8.5 * scale, 'bold');
for c = 1:2
    label(ax, centres(2 * c), 0.78, 'Mean BER', 7.5 * scale, 'normal');
    label(ax, centres(2 * c + 1), 0.78, '30 dB BER', 7.5 * scale, 'normal');
end
plot(ax, [0.01, 0.99], [0.97, 0.97], 'k-', 'LineWidth', 0.85 * scale);
plot(ax, [0.16, 0.56], [0.845, 0.845], 'k-', 'LineWidth', 0.4 * scale);
plot(ax, [0.59, 0.99], [0.845, 0.845], 'k-', 'LineWidth', 0.4 * scale);
plot(ax, [0.01, 0.99], [0.71, 0.71], 'k-', 'LineWidth', 0.55 * scale);
minimums = min(values, [], 1);
for k = 1:numel(groups)
    y = 0.645 - (k - 1) * 0.105;
    weight = 'normal';
    if strcmp(groups{k}, 'D')
        weight = 'bold';
    end
    label(ax, centres(1), y, groups{k}, 8 * scale, weight);
    for column = 1:4
        weight = 'normal';
        if values(k, column) == minimums(column)
            weight = 'bold';
        end
        label(ax, centres(column + 1), y, sprintf('%.3e', values(k, column)), ...
            8 * scale, weight);
    end
end
plot(ax, [0.01, 0.99], [0.27, 0.27], 'k-', 'LineWidth', 0.85 * scale);
notes = {'A/B: Fixed CSI; C/D: Iterative CSI refinement.', ...
    'A/C: Full-matrix LMMSE; B/D: MMSE unfolding.', ...
    'Mean BER: equal-weight average over 0:5:30 dB.'};
for k = 1:numel(notes)
    text(ax, 0.01, 0.195 - (k - 1) * 0.065, notes{k}, ...
        'FontName', 'Arial', 'FontSize', 6.4 * scale, 'Interpreter', 'none', ...
        'VerticalAlignment', 'middle');
end
drawnow;
end

function label(ax, x, y, value, fontSize, weight)
text(ax, x, y, value, 'FontName', 'Arial', 'FontSize', fontSize, ...
    'FontWeight', weight, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'Interpreter', 'none');
end
