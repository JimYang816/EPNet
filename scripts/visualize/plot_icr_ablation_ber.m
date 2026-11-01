function [fig, ax] = plot_icr_ablation_ber(runDir, figureWidth)
%PLOT_ICR_ABLATION_BER 补充材料：A-D 完整逐 SNR 的 BER 曲线。
%   正文使用 plot_icr_ablation_table；本图保留 0:5:30 dB 全部测量值。
%   默认宽度 7.16 in，左 NOF1、右 NCS1，共用图例，不自动保存。
if nargin < 1
    runDir = [];
end
if nargin < 2 || isempty(figureWidth)
    figureWidth = 7.16;
end
[fig, ax] = plot_icr_ablation_metric(runDir, figureWidth, 'ber');
end
