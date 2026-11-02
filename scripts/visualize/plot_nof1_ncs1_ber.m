% Plot NOF1 and NCS1 BER-SNR curves in two separate MATLAB figures.
% Run this script from any working directory. No figures or data are exported.
% Data source: the output directory in configs/evaluation/ber_comparison.json.

scriptDirectory = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(scriptDirectory));
comparisonConfig = jsondecode(fileread(fullfile(projectRoot, ...
    'configs', 'evaluation', 'ber_comparison.json')));
resultsRoot = fullfile(projectRoot, comparisonConfig.output);

channels = {'NOF1', 'NCS1'};
algorithmIds = {'single_tap_mmse', 'full_matrix_lmmse', 'mmse_dfe', ...
    'dd_ce_lmmse', 'udnet', 'oamp_net2', 'icr_mmse'};
algorithmLabels = {'Single-tap LMMSE', 'Full-matrix LMMSE', 'MMSE-DFE', ...
    'DD-DPSS', 'UDNet', 'OAMP-Net2', 'CRU-Net'};

% A dashed line and larger plus signs distinguish DD-DPSS where BER curves overlap.
style.colors = [0.40 0.44 0.53; 0.12 0.47 0.71; 0.00 0.62 0.68; ...
    0.50 0.28 0.65; 0.94 0.49 0.10; 0.04 0.52 0.36; 0.79 0.08 0.29];
style.lineStyles = {'-', '-', '-', '--', '-', '-', '-'};
style.markers = {'o', 's', 'x', '+', '^', 'd', 'p'};
style.markerSizes = [4.0 4.0 4.5 6.5 4.5 4.0 5.5];
style.lineWidth = 1.0;
style.figureSize = [3.5 2.8]; % Shared ICC single-column width and height, inches.
style.axesPosition = [0.16 0.15 0.81 0.81]; % Refined from label extents below.
style.fontName = 'Arial';
style.fontSize = 9;
style.legendFontSize = 8.0;
style.legendTokenSize = [10 6];
% Lower/upper margins as fractions of each channel's logarithmic BER span.
style.yPadding = [0.20 0.05; 0.05 0.10]; % Clear the seven-entry legend on NOF1.

% Read both complete test reports before opening any figure.
berData = cell(1, numel(channels));
for channelIndex = 1:numel(channels)
    reportPath = fullfile(resultsRoot, lower(channels{channelIndex}), 'metrics.json');
    berData{channelIndex} = load_ber_report(reportPath, ...
        channels{channelIndex}, algorithmIds);
end

figures = struct();
berAxes = gobjects(1, numel(channels));
axesPositions = zeros(numel(channels), 4);
for channelIndex = 1:numel(channels)
    channel = channels{channelIndex};
    data = berData{channelIndex};
    fig = figure('Name', [channel ' BER-SNR'], 'NumberTitle', 'off', ...
        'Color', 'w', 'WindowStyle', 'normal', 'Units', 'inches', ...
        'Position', [0.4 + 3.8 * (channelIndex - 1), 1.2, style.figureSize]);
    figures.(lower(channel)) = fig;
    ax = axes('Parent', fig, 'Units', 'normalized', 'Position', style.axesPosition);
    berAxes(channelIndex) = ax;
    hold(ax, 'on');
    set(ax, 'YScale', 'log', 'FontName', style.fontName, ...
        'FontSize', style.fontSize, 'LineWidth', 0.7, 'Box', 'on', ...
        'TickDir', 'out', 'XMinorTick', 'off', 'YMinorTick', 'on', ...
        'XGrid', 'on', 'XMinorGrid', 'off', 'YGrid', 'on', 'YMinorGrid', 'on', ...
        'GridLineStyle', '-', 'MinorGridLineStyle', ':', ...
        'GridColor', [0.65 0.65 0.65], 'GridAlpha', 0.80, ...
        'MinorGridColor', [0.60 0.60 0.60], 'MinorGridAlpha', 0.60, ...
        'Layer', 'bottom');

    curveHandles = gobjects(1, numel(algorithmIds));
    displayedBer = [];
    hasZeroErrors = false;
    for algorithmIndex = 1:numel(algorithmIds)
        ber = data.ber(algorithmIndex, :);
        zero = ber == 0;
        plottedBer = ber;
        plottedBer(zero) = NaN; % Break the line at a zero-error observation.
        curveHandles(algorithmIndex) = semilogy(ax, data.snrDb, plottedBer, ...
            'Color', style.colors(algorithmIndex, :), ...
            'LineStyle', style.lineStyles{algorithmIndex}, ...
            'Marker', style.markers{algorithmIndex}, ...
            'MarkerSize', style.markerSizes(algorithmIndex), ...
            'MarkerFaceColor', style.colors(algorithmIndex, :), ...
            'LineWidth', style.lineWidth, 'DisplayName', algorithmLabels{algorithmIndex});
        displayedBer = [displayedBer, ber(~zero)]; %#ok<AGROW>
        if any(zero)
            zeroLevel = 0.5 ./ data.totalBits(algorithmIndex, zero);
            semilogy(ax, data.snrDb(zero), zeroLevel, 'v', ...
                'Color', style.colors(algorithmIndex, :), 'MarkerSize', 8, ...
                'MarkerFaceColor', 'none', 'LineStyle', 'none', ...
                'LineWidth', 1.4, 'HandleVisibility', 'off');
            displayedBer = [displayedBer, zeroLevel]; %#ok<AGROW>
            hasZeroErrors = true;
        end
    end
    % Keep DD-DPSS visible where it nearly coincides with OAMP-Net2 on NOF1.
    uistack(curveHandles(strcmp(algorithmIds, 'dd_ce_lmmse')), 'top');

    xlim(ax, [min(data.snrDb) - 1.5, max(data.snrDb) + 1.5]);
    xticks(ax, data.snrDb);
    % Pad in logarithmic coordinates to match the reference's tight framing.
    logRange = max(log10(max(displayedBer) / min(displayedBer)), 0.5);
    lowerLimit = min(displayedBer) * 10 ^ (-style.yPadding(channelIndex, 1) * logRange);
    upperLimit = max(displayedBer) * 10 ^ (style.yPadding(channelIndex, 2) * logRange);
    % Reserve space for the lower-left legend; keep any zero-error markers visible.
    lowerLimit = min(1e-3, lowerLimit);
    ylim(ax, [lowerLimit, min(1, upperLimit)]);
    % Grid lines follow the ticks: SNR samples and the same BER values in both plots.
    % Logarithmic BER grid: 10^k major lines and (2:9)*10^k minor lines.
    % Minor lines have unequal vertical spacing on the logarithmic axis.
    decades = floor(log10(ax.YLim(1))):ceil(log10(ax.YLim(2)));
    majorBerTicks = 10 .^ decades;
    minorBerTicks = reshape((2:9).' * majorBerTicks, 1, []);
    yticks(ax, majorBerTicks(majorBerTicks >= ax.YLim(1) & majorBerTicks <= ax.YLim(2)));
    ax.YAxis.MinorTickValues = minorBerTicks( ...
        minorBerTicks > ax.YLim(1) & minorBerTicks < ax.YLim(2));
    xlabel(ax, 'SNR (dB)', 'FontSize', style.fontSize + 1);
    ylabel(ax, 'BER', 'FontSize', style.fontSize + 1);
    legendHandle = legend(ax, curveHandles, algorithmLabels, 'Location', 'southwest', ...
        'Interpreter', 'none', 'FontName', style.fontName, ...
        'FontSize', style.legendFontSize, 'NumColumns', 1, 'Box', 'on', ...
        'Color', 'w', 'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 0.7);
    legendHandle.ItemTokenSize = style.legendTokenSize;
    if hasZeroErrors
        text(ax, 0.03, 0.03, 'Open triangles: zero errors, shown at 0.5/N_{bits}', ...
            'Units', 'normalized', 'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'bottom', 'FontSize', style.fontSize - 2);
    end
    hold(ax, 'off');
    configure_icc_metric_figure(fig, ax);
    axesPositions(channelIndex, :) = ax.Position;
end
% Align the plotting frames as well as the PDF pages, preserving label margins.
lowerCorner = max(axesPositions(:, 1:2), [], 1);
upperCorner = min(axesPositions(:, 1:2) + axesPositions(:, 3:4), [], 1);
for channelIndex = 1:numel(channels)
    berAxes(channelIndex).Position = [lowerCorner, upperCorner - lowerCorner];
end
drawnow;

function data = load_ber_report(reportPath, channel, algorithmIds)
% Match algorithm identifiers, independent of JSON field-name conversion.
    assert(isfile(reportPath), 'BERPlot:MissingReport', ...
        'Report not found: %s', reportPath);
    report = jsondecode(fileread(reportPath));
    assert(report.complete && ~report.smoke && strcmp(report.channel, channel) ...
        && strcmp(report.input_view, 'compensated') ...
        && strcmp(report.nmse.split, 'test'), 'BERPlot:InvalidReport', ...
        'Expected a complete compensated-CSI test report for %s.', channel);
    data.snrDb = report.snr_db(:).';
    assert(isequal(data.snrDb, 0:5:30), 'BERPlot:InvalidSnr', ...
        'Expected the SNR grid 0:5:30 dB for %s.', channel);
    curves = struct2cell(report.curves);
    identifiers = cellfun(@(curve) curve.identifier, curves, 'UniformOutput', false);
    data.ber = zeros(numel(algorithmIds), numel(data.snrDb));
    data.totalBits = zeros(size(data.ber));
    for algorithmIndex = 1:numel(algorithmIds)
        if strcmp(algorithmIds{algorithmIndex}, 'dd_ce_lmmse')
            curve = load_soft_dd_dpss(report, reportPath);
        else
            matches = find(strcmp(identifiers, algorithmIds{algorithmIndex}));
            assert(isscalar(matches), 'BERPlot:MissingAlgorithm', ...
                'Expected one %s curve in %s.', algorithmIds{algorithmIndex}, reportPath);
            curve = curves{matches};
        end
        ber = curve.ber(:).';
        errors = curve.errors(:).';
        bits = curve.total_bits(:).';
        assert(numel(ber) == numel(data.snrDb) && numel(errors) == numel(ber) ...
            && numel(bits) == numel(ber), 'BERPlot:InvalidLength', ...
            'BER/count lengths do not match the SNR grid in %s.', reportPath);
        assert(all(isfinite(ber) & isfinite(errors) & isfinite(bits)) ...
            && all(bits > 0 & bits == fix(bits)) ...
            && all(errors >= 0 & errors <= bits & errors == fix(errors)) ...
            && all(abs(ber - errors ./ bits) < 1e-12), 'BERPlot:InvalidCounts', ...
            'BER does not match error/bit counts for %s in %s.', ...
            algorithmIds{algorithmIndex}, reportPath);
        data.ber(algorithmIndex, :) = ber;
        data.totalBits(algorithmIndex, :) = bits;
    end
end
