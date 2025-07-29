clc; clear;

param_name = "n_training"; 
% EbNo_dB_b,  EbNo_dB_e, BER_limit_dB, n_training

path_pre = '/Users/ufukaltun/antieavesdropping model/sim_results_eve/';

if param_name == "EbNo_dB_b"
    data = load(fullfile(path_pre, 'simulation_results_snr4.mat'));
    xlabel_display = 'SNR_{Bob}   (dB)';
elseif param_name == "EbNo_dB_e"
    data = load(fullfile(path_pre, 'simulation_results_snreve4.mat'));
    xlabel_display = 'SNR_{Eve}   (dB)';
elseif param_name == "BER_limit_dB"
    data = load(fullfile(path_pre, 'simulation_results_berlim4.mat'));
    xlabel_display = 'BER_{lim}   (dB)';
elseif param_name == "n_training"
    data = load(fullfile(path_pre, 'simulation_results_n_training4.mat'));
    xlabel_display = 'n_{training}';    
end
data = data.avg_results;


model_names = ["Contextual_OFDM_IM", "Contextual_OFDM", ...
             "Rule_Based_OFDM", ...
            "Contextual_Incesure_PCSI_OFDM", "Contextual_Incesure_ICSI_OFDM"];

model_names_print = ["CMAB OFDM-IM", "CMAB OFDM", ...
             "Rule Based OFDM", ...
            "CMAB Insecure PCSI", "CMAB Insecure ICSI"];


markers = {'o', 'o', '*', '+', 'x'}; % Different markers for clarity
lines = {'-', ':', ':', ':', ':'};
line_widths = {1.5, 1.5, 1.5, 0.5, 0.5};

param_values = data.(param_name);

% Initialize arrays for plotting
throughput = cell(length(model_names), 1);
BER_b = cell(length(model_names), 1);
BER_e = cell(length(model_names), 1);


% Extract results
for i = 1:length(model_names)
    model_name = model_names(i);
    throughput{i} = data.(model_name).throughput;
    BER_b{i} = data.(model_name).BER_b;
    BER_e{i} = data.(model_name).BER_e;
end

%% --- Throughput vs Parameter ---
figure;
set(gcf, 'Units', 'inches', 'Position', [1, 1, 5, 8]); % Set figure size to 12x5 inches
hold on;
for i = 1:length(model_names)    
    plot(param_values, throughput{i}, [lines{i}, markers{i}], 'Color', 'k', 'LineWidth', line_widths{i}, 'MarkerSize', 8, 'DisplayName', model_names_print{i});
end
if param_name == "n_training"
    xlim([20 280]);
else
    xlim([-2 28]);
end

xlabel(xlabel_display, 'FontSize', 14);
ylabel('$\tau_{\mathrm{avg}}$ \quad (bits/symbol)', 'FontSize', 14, 'Interpreter', 'latex');
legend('Location', 'best', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 12);

print(param_name+'_tau', '-depsc');
hold off;

%% --- BER of Bob & Eve (Same Y-axis) ---
figure;
set(gcf, 'Units', 'inches', 'Position', [1, 1, 5, 8]); % Set figure size to 12x5 inches

% --- BER of Bob ---
hold on;
for i = 1:length(model_names)
    plot(param_values, BER_b{i}, [lines{i}, markers{i}], 'Color', 'k', 'LineWidth', line_widths{i}, 'MarkerSize', 8, 'DisplayName', model_names_print{i});
end

% --- BER of Eve ---
if param_name == "BER_limit_dB"
    hold on;
    h = drawline('Position', [0 1e0; 30 1e-3], 'LineWidth', 1,'StripeColor','g');
    % Add text near the middle of the line
    text(11, 0.11, 'BER_{lim}', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'g', 'HorizontalAlignment', 'left', 'Rotation', -63);
end

hold on;
for i = 1:length(model_names)
    plot(param_values, BER_e{i}, [lines{i}, markers{i}], 'Color', 'r', 'LineWidth', line_widths{i}, 'MarkerSize', 8, 'DisplayName', model_names_print{i});
end
xlabel(xlabel_display, 'FontSize', 14);
ylabel('BER_{avg}', 'FontSize', 14);
if param_name == "n_training"
    xlim([20 280]);
else
    xlim([-2 28]);
end
ylim([3e-3 1e0]);
%yticks([1e-2 1e-1 0.5 1]); % Specify the tick positions

set(gca, 'YScale', 'log'); % Log scale for BER
lgd = legend('Location', 'best', 'FontSize', 10,'NumColumns', 2);
lgd.Position = [0.47, 0.85, 0.2, 0.1];
grid on;

set(gca, 'FontSize', 12);
print(param_name+'_ber', '-depsc');
hold off;





