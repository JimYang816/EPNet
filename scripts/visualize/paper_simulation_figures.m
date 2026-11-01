function figures = paper_simulation_figures(outputDir, mainRun, ablationRun)
%PAPER_SIMULATION_FIGURES 仅用 MATLAB 导出正文结果和完整消融补充材料。
%   在项目根目录运行：
%     addpath('scripts/visualize');
%     figures = paper_simulation_figures;
%   或传入输出目录、主对比结果目录、正式 A-D 消融目录。
%   正文：BER-SNR（双栏）、NMSE-SNR（单栏四曲线）、ABL-table（单栏）。
%   补充：ABL-SNR、ABL-NMSE-SNR，保留所有逐 SNR 测量值。
%   默认导出到 outputs/paper，与 main.tex 同目录；不修改论文主稿。
%   图为矢量 PDF / 300 dpi PNG，表另存完整精度 CSV 和可直接引用的 TeX。

projectDir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(projectDir, 'outputs', 'paper');
end
if nargin < 2
    mainRun = [];
end
if nargin < 3
    ablationRun = [];
end
if ~isfolder(outputDir)
    mkdir(outputDir);
end
[figures.ber, ~] = plot_nof1_ncs1_ber_icc(mainRun);
[figures.nmse, ~] = plot_nof1_ncs1_nmse_icc(mainRun);
[figures.ablation, summary] = plot_icr_ablation_table(ablationRun);
[figures.supplement_ber, ~] = plot_icr_ablation_ber(ablationRun);
[figures.supplement_nmse, ~] = plot_icr_ablation_nmse(ablationRun);
names = {'ber', 'nmse', 'ablation', 'supplement_ber', 'supplement_nmse'};
stems = {'BER-SNR', 'NMSE-SNR', 'ABL-table', 'ABL-SNR', 'ABL-NMSE-SNR'};
for k = 1:numel(names)
    fig = figures.(names{k});
    print(fig, fullfile(outputDir, stems{k}), '-dpdf', '-painters');
    print(fig, fullfile(outputDir, stems{k}), '-dpng', '-r300');
end
writetable(summary, fullfile(outputDir, 'ABL-table.csv'));
writeTableTex(summary, fullfile(outputDir, 'ABL-table.tex'));
writeIncludes(outputDir);
disp(summary);
fprintf('MATLAB figures and table: %s\n', outputDir);
end

function writeTableTex(summary, path)
[file, message] = fopen(path, 'w', 'n', 'UTF-8');
assert(file >= 0, '无法写入表格：%s', message);
cleanup = onCleanup(@() fclose(file));
lines = {
    '% Main-text single-column table; generated from compensated test reports.'
    '\begin{table}[t]'
    '  \centering'
    '  \caption{A--D ablation: mean BER and BER at 30 dB on both WATERMARK channels.}'
    '  \label{tab:simulation_ablation}'
    '  \begingroup\small\setlength{\tabcolsep}{3pt}'
    '  \begin{tabular}{crrrr}'
    '    \hline'
    '    & \multicolumn{2}{c}{NOF1} & \multicolumn{2}{c}{NCS1} \\'
    '    Variant & Mean BER & 30 dB BER & Mean BER & 30 dB BER \\'
    '    \hline'
};
fprintf(file, '%s\n', lines{:});
values = summary{:, 2:5};
minimums = min(values, [], 1);
for k = 1:height(summary)
    cells = cell(1, 4);
    for column = 1:4
        cells{column} = sprintf('%.3e', values(k, column));
        if values(k, column) == minimums(column)
            cells{column} = ['\textbf{', cells{column}, '}'];
        end
    end
    row = ['    ', strjoin([{summary.Variant{k}}, cells], ' & '), ' \\'];
    fprintf(file, '%s\n', row);
end
lines = {
    '    \hline'
    '  \end{tabular}\endgroup'
    '  \par\smallskip'
    '  \begin{minipage}{\columnwidth}\footnotesize'
    '    A/B: fixed CSI; C/D: iterative CSI refinement.'
    '    A/C: Full-matrix LMMSE; B/D: MMSE unfolding (D: complete CRU-Net).'
    '    Mean BER is the equal-weight arithmetic mean over $0:5:30$ dB.'
    '    The lowest BER in each column is bold. Full SNR curves are supplementary.'
    '  \end{minipage}'
    '\end{table}'
};
fprintf(file, '%s\n', lines{:});
end

function writeIncludes(outputDir)
main = {
    '% MATLAB-generated main results. Copy these environments into the manuscript.'
    '\begin{figure*}[t]'
    '  \centering'
    '  \includegraphics[width=\textwidth]{BER-SNR.pdf}'
    '  \caption{BER comparison on WATERMARK channels: (a) NOF1; (b) NCS1.}'
    '  \label{fig:simulation_ber}'
    '\end{figure*}'
    '\begin{figure}[t]'
    '  \centering'
    '  \includegraphics[width=\columnwidth]{NMSE-SNR.pdf}'
    '  \caption{Mean full-matrix CSI NMSE: initial and CRU-Net refined CSI on NOF1 and NCS1.}'
    '  \label{fig:simulation_nmse}'
    '\end{figure}'
    '\input{ABL-table.tex}'
};
supplement = {
    '% Supplement only: complete per-SNR A--D curves.'
    '\begin{figure*}[t]'
    '  \centering'
    '  \includegraphics[width=\textwidth]{ABL-SNR.pdf}'
    '  \caption{Supplementary A--D BER curves: (a) NOF1; (b) NCS1.}'
    '  \label{fig:ablation_ber_supplement}'
    '\end{figure*}'
    '\begin{figure*}[t]'
    '  \centering'
    '  \includegraphics[width=\textwidth]{ABL-NMSE-SNR.pdf}'
    '  \caption{Supplementary A--D mean CSI NMSE curves: (a) NOF1; (b) NCS1. A/B coincide.}'
    '  \label{fig:ablation_nmse_supplement}'
    '\end{figure*}'
};
writeLines(fullfile(outputDir, 'simulation_results.tex'), main);
writeLines(fullfile(outputDir, 'simulation_supplement.tex'), supplement);
end

function writeLines(path, lines)
[file, message] = fopen(path, 'w', 'n', 'UTF-8');
assert(file >= 0, '无法写入 LaTeX 引用：%s', message);
cleanup = onCleanup(@() fclose(file));
fprintf(file, '%s\n', lines{:});
end
