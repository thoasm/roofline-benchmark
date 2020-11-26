function [] = plot_roofline(path_to_file)
if (nargin < 1)
    path_to_file = "../20201125_A100_roofline_d3.csv";
    file_postfix = "_d3_a100";
    %path_to_file = "../speedup/MI100/speedup_mi100.csv";
    %file_postfix = "_mi100";
end


% Makes it the default behavior to always fill out the graph window
set(0,'DefaultAxesLooseInset',[0,0,0,0]);

% set different color scheme
mycolors=[
         0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840
    0.2500    0.2500    0.2500
];
myblue    = mycolors(1,:);
myorange  = mycolors(2,:);
myyellow  = mycolors(3,:);
mymagenta = mycolors(4,:);
mygreen   = mycolors(5,:);
mycyan    = mycolors(6,:);
myred     = mycolors(7,:);
myblack   = mycolors(8,:);

outer_loop_idx = 7;
outer_loop_value = 4;


data = readtable(path_to_file, 'VariableNamingRule', 'preserve');
data_size = size(data);
num_rows = data_size(1);

prec = table2array(data(:, 1));

%type_strings = ["<float64>" "<float32>" "<float16>" "<int32>" "<int16>"];
outer_table_mask = table2array(data(:, outer_loop_idx)) == outer_loop_value;
double_mask = boolean(zeros(num_rows, 1));
float_mask = double_mask;
acc_dd_mask = double_mask;
acc_df_mask = double_mask;
%acc_ff_mask = double_mask;

for i = 1:num_rows
    double_mask(i) = strcmp("double", prec(i, 1));
    float_mask(i) = strcmp("float", prec(i, 1));
    acc_dd_mask(i) = strcmp("Ac<3, d, d>", prec(i, 1));
    acc_df_mask(i) = strcmp("Ac<3, d, f>", prec(i, 1));
    %acc_ff_mask(i) = strcmp("Acc<3, f, f>", prec(i, 1));
end

double_mask = outer_table_mask & double_mask;
float_mask = outer_table_mask & float_mask;
acc_dd_mask = outer_table_mask & acc_dd_mask;
acc_df_mask = outer_table_mask & acc_df_mask;


double_plot = [ table2array(data(double_mask, 5))./table2array(data(double_mask, 6)) table2array(data(double_mask, 2:3)) ];
float_plot = [ table2array(data(float_mask, 5))./table2array(data(float_mask, 6)) table2array(data(float_mask, 2:3)) ];
acc_dd_plot = [ table2array(data(acc_dd_mask, 5))./table2array(data(acc_dd_mask, 6)) table2array(data(acc_dd_mask, 2:3)) ];
acc_df_plot = [ table2array(data(acc_df_mask, 5))./table2array(data(acc_df_mask, 6)) table2array(data(acc_df_mask, 2:3)) ];


LW = 'linewidth';
lw = 1;
FS = 'fontsize';
fsla = 18; %14;
fsle = 18; %14;


[ figure1, axes1 ] = plot_helper(double_plot(:, 1:2), float_plot(:, 1:2), acc_dd_plot(:, 1:2), acc_df_plot(:, 1:2));
legend_str = string([]);

legend_str(end+1) = "double precision";
legend_str(end+1) = "single precision";
legend_str(end+1) = "Accessor<d, d>";
legend_str(end+1) = "Accessor<d, s>";
yline(19490.0, '--', 'Color', myblue, LW, lw);
legend_str(end+1) = "Peak single precision";
yline(9746.0, '--', 'Color', mygreen, LW, lw);
legend_str(end+1) = "Peak double precision";

legend(axes1, legend_str, FS, fsle, 'Location', 'SE');
xlabel(axes1, "FLOP / Byte", FS, fsla);
ylabel(axes1, "GFLOP / s", FS, fsla);

% Actually plot:
print_to_file(figure1, strcat('plots/', 'roofline_compute', file_postfix));
hold(axes1, 'off');

%%%%%%%%%%%%%%%%%%%%%%  Plot FLOP / Value  %%%%%%%%%%%%%%%%%%%%%%%%5

[ figure2, axes2 ] = plot_helper(double_plot(:, 1:2).*[8, 1], float_plot(:, 1:2).*[4, 1], ...
                                acc_dd_plot(:, 1:2).*[8, 1], acc_df_plot(:, 1:2).*[4, 1]);
legend_str = string([]);

legend_str(end+1) = "double precision";
legend_str(end+1) = "single precision";
legend_str(end+1) = "Accessor<d, d>";
legend_str(end+1) = "Accessor<d, s>";
yline(19490.0, '--', 'Color', myblue, LW, lw);
legend_str(end+1) = "Peak single precision";
yline(9746.0, '--', 'Color', mygreen, LW, lw);
legend_str(end+1) = "Peak double precision";

legend(axes2, legend_str, FS, fsle, 'Location', 'SE');
xlabel(axes2, "FLOP / Value", FS, fsla);
ylabel(axes2, "GFLOP / s", FS, fsla);

print_to_file(figure2, strcat('plots/', 'roofline_compute_pvalue', file_postfix));
hold(axes2, 'off');



%%%%%%%%%%%%%%%%%%%%%%  Plot Bandwidth  %%%%%%%%%%%%%%%%%%%%%%%%5

[ figure3, axes3 ] = plot_helper(double_plot(:, [1,3]), float_plot(:, [1,3]),...
                                acc_dd_plot(:, [1,3]), acc_df_plot(:, [1,3]));

legend_str = string([]);

legend_str(end+1) = "double precision";
legend_str(end+1) = "single precision";
legend_str(end+1) = "Accessor<d, d>";
legend_str(end+1) = "Accessor<d, s>";
yline(1555, '--', 'Color', myblue, LW, lw);
legend_str(end+1) = "Peak Bandwidth";

legend(axes3, legend_str, FS, fsle, 'Location', 'SW');
xlabel(axes3, "FLOP / Byte", FS, fsla);
ylabel(axes3, "GB / s", FS, fsla);

print_to_file(figure3, strcat('plots/', 'roofline_bandwidth', file_postfix));
hold(axes3, 'off');

%%%%%%%%%%%%%%%%%%%%%%  Plot Bandwidth / Value %%%%%%%%%%%%%%%%%%%%%%%%5

[ figure4, axes4 ] = plot_helper(double_plot(:, [1,3]).*[8,1], float_plot(:, [1,3]).*[4,1],...
                                acc_dd_plot(:, [1,3]).*[8,1], acc_df_plot(:, [1,3]).*[4,1]);

legend_str = string([]);

legend_str(end+1) = "double precision";
legend_str(end+1) = "single precision";
legend_str(end+1) = "Accessor<d, d>";
legend_str(end+1) = "Accessor<d, s>";
yline(1555, '--', 'Color', myblue, LW, lw);
legend_str(end+1) = "Peak Bandwidth";

legend(axes4, legend_str, FS, fsle, 'Location', 'SW');
xlabel(axes4, "FLOP / Value", FS, fsla);
ylabel(axes4, "GB / s", FS, fsla);

print_to_file(figure4, strcat('plots/', 'roofline_bandwidth_pvalue', file_postfix));
hold(axes4, 'off');

end

function [] = print_to_file(figure, plotname)
plot_width  = 1000;%1600;
plot_height =  400;% 600;
paper_scalar = 1.02;
output_format='-dpdf';

set(figure, 'position', [0 0 plot_width plot_height]);

% Actually plot:
set(figure, 'Color', 'white'); % white bckgr

% Makes it the default behavior to always fill out the graph window
figure.PaperPositionMode = 'auto';
fig_pos = figure.PaperPosition;
figure.PaperSize = [fig_pos(3)*paper_scalar fig_pos(4)*paper_scalar];
print(figure, plotname, output_format)
end

function [ figure1, axes1 ] = plot_helper(p1, p2, p3, p4)
% set different color scheme
mycolors=[
         0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840
    0.2500    0.2500    0.2500
];
myblue    = mycolors(1,:);
myorange  = mycolors(2,:);
myyellow  = mycolors(3,:);
mymagenta = mycolors(4,:);
mygreen   = mycolors(5,:);
mycyan    = mycolors(6,:);
myred     = mycolors(7,:);
myblack   = mycolors(8,:);



LW = 'linewidth';
lw = 1;
MS = 'markersize';
ms = 12;%10;
FS = 'fontsize';
fsaxis = 14;%12;


figure1 = figure('Color',[1 1 1]);
% Create axes
axes1 = axes('Parent',figure1);

hold(axes1,'on');

plt1 = plot(p1(:, 1), p1(:, 2),  's-', 'Color', mygreen, 'MarkerEdgeColor', mygreen, 'MarkerFaceColor', mygreen, LW, lw, MS, ms);
plt2 = plot(p2(:, 1), p2(:, 2),  '+-', 'Color', myblue, 'MarkerEdgeColor', myblue, 'MarkerFaceColor', myblue, LW, lw, MS, ms);
plt3 = plot(p3(:, 1), p3(:, 2),  '*-', 'Color', myorange, 'MarkerEdgeColor', myorange, 'MarkerFaceColor', myorange, LW, lw, MS, ms);
plt4 = plot(p4(:, 1), p4(:, 2),  'X-', 'Color', mymagenta, 'MarkerEdgeColor', mymagenta, 'MarkerFaceColor', mymagenta, LW, lw, MS, ms);

set(axes1,'XGrid','on', 'YGrid', 'on');
hold(axes1, 'on');
set(axes1, 'YScale', 'log', FS, fsaxis);
set(axes1, 'XScale', 'log', FS, fsaxis);
end