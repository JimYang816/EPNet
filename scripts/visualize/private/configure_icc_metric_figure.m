function configure_icc_metric_figure(fig, ax)
% Fit labels inside a 3.5-inch PDF page with only a 3-point outer margin.
% This configures manual printing; it does not export any files.
    fig.Units = 'inches';
    figureSize = fig.Position(3:4);
    assert(abs(figureSize(1) - 3.5) < 1e-9, ...
        'ICCPlot:InvalidWidth', 'ICC metric figures must be 3.5 inches wide.');
    padding = 3 / 72;
    ax.Units = 'inches';
    ax.PositionConstraint = 'innerposition';
    % Text extents can change after moving the axes, especially on log axes.
    for pass = 1:3
        drawnow;
        % TightInset can omit outward ticks on the top/right box edges.
        tickOverhang = ax.TickLength(1) * max(ax.Position(3:4));
        inset = max(ax.TightInset, tickOverhang);
        ax.Position = [padding + inset(1), padding + inset(2), ...
            figureSize(1) - 2 * padding - inset(1) - inset(3), ...
            figureSize(2) - 2 * padding - inset(2) - inset(4)];
    end
    set(fig, 'PaperUnits', 'inches', 'PaperSize', figureSize, ...
        'PaperPosition', [0 0 figureSize], 'PaperPositionMode', 'manual', ...
        'InvertHardcopy', 'off', 'Renderer', 'painters');
end
