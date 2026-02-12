%% 1. Configuration Parameters
fileName = 'C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\adc_data.bin';
numADCSamples = 256;       
numRX = 4;                 
numTX = 3;                 
numChirpLoops = 128;       
numChirpsPerFrame = numTX * numChirpLoops; 
rangeResolution = 0.044;   
minRangeBin = 5;
outputRangeBins = 100; 

%% 2. Read Binary File
fid = fopen(fileName, 'r');
adcData = fread(fid, 'int16');
fclose(fid);

% Convert interleaved complex data
adcData = adcData(1:2:end) + 1j * adcData(2:2:end); 

% Calculate Frames
totalComplexSamples = length(adcData);
samplesPerFrame = numRX * numADCSamples * numChirpsPerFrame;
numFrames = floor(totalComplexSamples / samplesPerFrame);
fprintf('Found %d Frames.\n', numFrames);

% Raw Data Cube
adcData = adcData(1:numFrames*samplesPerFrame);
rawCube = reshape(adcData, numRX, numADCSamples, numChirpsPerFrame, numFrames);

%% 3. Frame Processing Loop
RDT_Tensor = zeros(outputRangeBins, 128, numFrames);

fprintf('Starting RDT Processing...\n');

for fr = 1:numFrames
    % --- A. Prepare Data [ADC, Chirp, RX] ---
    frameData = rawCube(:, :, :, fr);
    frameData = permute(frameData, [2, 3, 1]); 
    
    tx0_data = frameData(:, 1:3:end, :); 
    tx1_data = frameData(:, 2:3:end, :); 
    
    % --- B. Range-Doppler Processing ---
    % 1. Range FFT
    rngWin = hanning(numADCSamples);
    tx0_rng = fft(tx0_data .* rngWin, numADCSamples, 1);
    
    % Remove DC
    %tx0_rng = tx0_rng - mean(tx0_rng, 2);
    
    % 2. Doppler FFT
    dopWin = hanning(size(tx0_rng, 2)).';
    rd_tx0 = fftshift(fft(tx0_rng .* dopWin, [], 2), 2);
    
    % 3. Magnitude Map
    magMapLinear = sum(abs(rd_tx0).^2, 3);
    magMap_dB = 10 * log10(magMapLinear + 1e-10);
    
    % --- C. CFAR Detection ---
    % CA-CFAR Parameters
    numGuard = 2;       
    numTrain = 4;       
    P_fa = 1e-3;  
    
    % Calculate Threshold Factor
    N_cells = (2*numTrain + 2*numGuard + 1)^2 - (2*numGuard + 1)^2;
    alpha = N_cells * (P_fa^(-1/N_cells) - 1);
    
    cfarMask = zeros(size(magMapLinear));
    offset = numTrain + numGuard;
    [nRows, nCols] = size(magMapLinear);
    
    % Loop through Range (r) and Doppler (d) bins
    for r = (offset + 1):(nRows - offset)
        for d = (offset + 1):(nCols - offset)
            
            % 1. Extract the local window
            window = magMapLinear(r-offset : r+offset, d-offset : d+offset);
            
            % 2. Calculate Noise Floor
            totalSum = sum(window(:));

            % Extract inner guard area
            guardRegion = window(numTrain+1 : end-numTrain, numTrain+1 : end-numTrain);
            noiseSum = totalSum - sum(guardRegion(:));
            noiseFloor = noiseSum / N_cells;
            
            % 3. Compare CUT (Cell Under Test) to Threshold
            if magMapLinear(r,d) > (noiseFloor * alpha)
                cfarMask(r,d) = 1;
            end
        end
    end
    
    % Mask out the DC bins to avoid false positives
    cfarMask(1:minRangeBin, :) = 0;
    
    % Get indices of detected targets
    [rIdx, dIdx] = find(cfarMask == 1);
    
    % Determine Dynamic Threshold
    if ~isempty(rIdx)
        current_peak_dB = max(magMap_dB(cfarMask == 1));
        threshold_dB = current_peak_dB - 20; 
    else
        threshold_dB = 0;
    end
    
    % --- D. RDT Generation ---
    tempRDT = zeros(outputRangeBins, 128);
    for i = 1:length(rIdx)
        r = rIdx(i); 
        d = dIdx(i);
        
        % Range Processing
        if r <= outputRangeBins
            
            % Intensity
            intensity = magMap_dB(r, d) - threshold_dB; 
            intensity = max(0, intensity); 
            
            tempRDT(r, d) = intensity;
        end
    end
    
    if max(tempRDT(:)) > 0
        tempRDT = tempRDT / max(tempRDT(:));
    end
    RDT_Tensor(:, :, fr) = tempRDT;
    
    % Progress Indicator
    if mod(fr, 50) == 0
        fprintf('Processed Frame %d / %d\n', fr, numFrames);
    end
end

%% 4. Visualization
%figure('Name', 'RDT Feature Map (100 Bins)');
%maxVal = max(RDT_Tensor(:)); if maxVal==0, maxVal=1; end

%for t = 1:size(RDT_Tensor, 3)
%    imagesc(RDT_Tensor(:, :, t));
%    axis xy;            
%    colormap(jet);      
%    colorbar;
%    title(['RDT Frame: ', num2str(t), ' (Max Range ~4.4m)']);
%    xlabel('Doppler (Velocity Bins)');
%    ylabel('Range Bins (0-100)');
%    caxis([0, maxVal * 0.8]); 
%    pause(0.1); 
%end

%% 5. Save File
actionName = 'Idle'; % ['Jumping', 'Waving', 'Idle']
savePath   = fullfile('C:\MyDataset_RDT', actionName);

% Create folder if it doesn't exist
if ~exist(savePath, 'dir')
    mkdir(savePath);
end

% Get existing trial files
files = dir(fullfile(savePath, sprintf('%s_trial_*.mat', actionName)));

if isempty(files)
    trialNum = 1;
else
    % Extract trial numbers directly
    trialNums = arrayfun(@(f) ...
        sscanf(f.name, [actionName '_trial_%d.mat']), files);
    trialNum = max(trialNums) + 1;
end

% Save the Tensor
fileName = sprintf('%s_trial_%04d.mat', actionName, trialNum);
save(fullfile(savePath, fileName), 'RDT_Tensor', '-v7.3');
