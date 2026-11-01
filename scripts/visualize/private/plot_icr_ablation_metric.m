function [fig, ax] = plot_icr_ablation_metric(runDir, figureWidth, metric)
%PLOT_ICR_ABLATION_METRIC A-D 全部逐 SNR 测量值，仅作补充材料。
validateattributes(figureWidth, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});
reports = load_icr_ablation_reports(runDir);
channels = {'NOF1', 'NCS1'};
groups = {'A', 'B', 'C', 'D'};
names = {'A: Fixed CSI + LMMSE', 'B: Fixed CSI + unfolding', ...
    'C: Refined CSI + LMMSE', 'D: CRU-Net'};
colors = [69, 69, 69; 0, 114, 178; 0, 158, 115; 213, 94, 0] / 255;
markers = {'o', 's', '^', 'd'};
styles = {'--', '-.', ':', '-'};
scale = figureWidth / 7.16;
figureHeight = 3.05 * scale;
fig = figure('Color', 'white', 'Name', ['Supplement: A-D ', metric], ...
    'NumberTitle', 'off', 'WindowStyle', 'normal', 'Units', 'inches', ...
    'Position', [1, 1, figureWidth, figureHeight], ...
    'PaperUnits', 'inches', 'PaperPositionMode', 'manual', ...
    'PaperPosition', [0, 0, figureWidth, figureHeight], ...
    'PaperSize', [figureWidth, figureHeight], 'InvertHardcopy', 'off');
positions = [0.095, 0.215, 0.385, 0.60; 0.595, 0.215, 0.385, 0.60];
ax = gobjects(1, 2);
handles = gobjects(2, 4);
for c = 1:numel(channels)
    report = reports{c};
    snr = report.snr_db(:);
    ax(c) = axes('Parent', fig, 'Units', 'normalized', 'Position', positions(c, :), ...
        'PositionConstraint', 'innerposition');
    hold(ax(c), 'on');
    allValues = [];
    for k = 1:numel(groups)
        values = report.curves.(groups{k}).(metric)(:);
        if strcmp(metric, 'ber')
            values(values == 0) = NaN;
        end
        handles(c, k) = plot(ax(c), snr, values, 'LineStyle', styles{k}, ...
            'Color', colors(k, :), 'Marker', markers{k}, 'MarkerSize', 4.2 * scale, ...
            'MarkerFaceColor', 'white', 'LineWidth', 1.15 * scale);
        if strcmp(metric, 'mean_nmse_db') && k <= 2
            % A/B 的重合曲线保留全部顶点，仅交替显示标记。
            handles(c, k).MarkerIndices = k:2:numel(snr);
        end
        allValues = [allValues; values]; %#ok<AGROW>
    end
    finiteValues = allValues(isfinite(allValues));
    assert(~isempty(finiteValues), '%s 没有可显示的正 BER。', channels{c});
    limits = [min(finiteValues), max(finiteValues)];
    if strcmp(metric, 'ber')
        set(ax(c), 'YScale', 'log');
        limits = log10(limits);
        padding = max(0.06 * diff(limits), 0.02);
        limits = 10 .^ (limits + [-padding, padding]);
        yName = 'BER';
    else
        padding = max(0.06 * diff(limits), 0.2);
        limits = limits + [-padding, padding];
        yName = 'Mean CSI NMSE (dB)';
    end
    xlabel(ax(c), {'SNR (dB)', sprintf('(%c) %s', 'a' + c - 1, channels{c})}, ...
        'FontName', 'Arial', 'FontSize', 9 * scale);
    ylabel(ax(c), yName, 'FontName', 'Arial', 'FontSize', 9 * scale);
    set(ax(c), 'FontName', 'Arial', 'FontSize', 8 * scale, ...
        'Box', 'on', 'LineWidth', 0.7 * scale, 'Layer', 'top', 'TickDir', 'out', ...
        'XTick', snr, 'XLim', [min(snr) - 1.2, max(snr) + 1.2], 'YLim', limits, ...
        'XGrid', 'on', 'YGrid', 'on', 'YMinorGrid', 'on', ...
        'GridColor', [0.55, 0.55, 0.55], 'GridAlpha', 0.20, 'MinorGridAlpha', 0.10);
end
sharedLegend = legend(ax(1), handles(1, :), names, 'NumColumns', 2, ...
    'Interpreter', 'none', 'FontName', 'Arial', 'FontSize', 8 * scale, ...
    'Box', 'off', 'Units', 'normalized', 'Location', 'none');
drawnow;
position = sharedLegend.Position;
position(1:2) = [(1 - position(3)) / 2, 0.985 - position(4)];
sharedLegend.Position = position;
linkaxes(ax, 'x');
drawnow;
end
