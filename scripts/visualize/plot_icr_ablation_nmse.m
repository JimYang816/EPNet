function [fig, ax] = plot_icr_ablation_nmse(runDir, figureWidth)
%PLOT_ICR_ABLATION_NMSE 补充材料：A-D 完整平均 NMSE-SNR 曲线。
%   正文使用 plot_icr_ablation_table；本图保留全部 SNR 测量值。
%   A/B 固定 CSI 的 NMSE 曲线重合，仅交替显示标记，不偏移数据。
if nargin < 1
    runDir = [];
end
if nargin < 2 || isempty(figureWidth)
    figureWidth = 7.16;
end
[fig, ax] = plot_icr_ablation_metric(runDir, figureWidth, 'mean_nmse_db');
end
