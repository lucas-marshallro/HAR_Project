%% 1. Configuration Parameters
fileName = 'C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\adc_data.bin';
numADCSamples = 256;       
numRX = 4;                 
numTX = 3;                 
numChirpLoops = 128;       
numChirpsPerFrame = numTX * numChirpLoops; 
rangeResolution = 0.044;   
minRangeBin = 5;

%% 2. Read Binary File
fid = fopen(fileName, 'r');
adcData = fread(fid, 'int16');
fclose(fid);

% Convert Interleaved Complex Data
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
DHT_Tensor = zeros(100, 128, numFrames);

fprintf('Starting DHT Processing...\n');

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
    tx1_rng = fft(tx1_data .* rngWin, numADCSamples, 1);
    
    % Remove DC (Static Clutter)
    %tx0_rng = tx0_rng - mean(tx0_rng, 2);
    %tx1_rng = tx1_rng - mean(tx1_rng, 2);
    
    % 2. Doppler FFT
    dopWin = hanning(size(tx0_rng, 2)).';
    rd_tx0 = fftshift(fft(tx0_rng .* dopWin, [], 2), 2);
    rd_tx1 = fftshift(fft(tx1_rng .* dopWin, [], 2), 2);
    
    % 3. Magnitude Map (Sum Power across 4 RX)
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
    
    % Determine Dynamic Threshold for DHT Intensity
    if ~isempty(rIdx)
        current_peak_dB = max(magMap_dB(cfarMask == 1));
        threshold_dB = current_peak_dB - 20; 
    else
        threshold_dB = 0;
    end
    
    % --- D. DHT Generation ---
    tempDHT = zeros(100, 128);
    for i = 1:length(rIdx)
        r = rIdx(i); 
        d = dIdx(i);
        
        % Angle Processing
        val_tx0 = sum(rd_tx0(r, d, :)); 
        val_tx1 = sum(rd_tx1(r, d, :));
        phaseDiff = angle(val_tx1 * conj(val_tx0));
        elevation = asin(phaseDiff / pi);
        
        % Height Calculation
        rangeVal = (r-1) * rangeResolution;
        heightVal = rangeVal * sin(elevation);
        
        % Map to Bin (0-100)
        hBin = round(((heightVal + 1.5) / 3) * 100);
        hBin = max(1, min(100, hBin));
        
        % Intensity
        intensity = magMap_dB(r, d) - threshold_dB; 
        intensity = max(0, intensity); 
        
        tempDHT(hBin, d) = tempDHT(hBin, d) + intensity;
    end
    
    if max(tempDHT(:)) > 0
        tempDHT = tempDHT / max(tempDHT(:));
    end
    DHT_Tensor(:, :, fr) = tempDHT;
    
    % Progress Indicator
    if mod(fr, 50) == 0
        fprintf('Processed Frame %d / %d\n', fr, numFrames);
    end
end

%% 4. Visualization
figure('Name', 'DHT Feature Map Sequence');
maxVal = max(DHT_Tensor(:)); if maxVal==0, maxVal=1; end

for t = 1:size(DHT_Tensor, 3)
    imagesc(DHT_Tensor(:, :, t));
    axis xy;            
    colormap(jet);      
    colorbar;
    title(['DHT Frame: ', num2str(t), ' of ', num2str(numFrames)]);
    xlabel('Doppler (Velocity Bins)');
    ylabel('Height Bins');
    caxis([0, maxVal * 0.8]); 
    pause(0.1); 
end

%% 5. Save File
%actionName = 'Idle'; % ['Jumping', 'Waving', 'Idle']
%savePath   = fullfile('C:\MyDataset_DHT', actionName);

% Create folder if it doesn't exist
%if ~exist(savePath, 'dir')
%    mkdir(savePath);
%end

% Get existing trial files
%files = dir(fullfile(savePath, sprintf('%s_trial_*.mat', actionName)));

%if isempty(files)
%    trialNum = 1;
%else
    % Extract trial numbers directly
%    trialNums = arrayfun(@(f) ...
%        sscanf(f.name, [actionName '_trial_%d.mat']), files);
%    trialNum = max(trialNums) + 1;
%end

% Save the Tensor
%fileName = sprintf('%s_trial_%04d.mat', actionName, trialNum);
%save(fullfile(savePath, fileName), 'DHT_Tensor', '-v7.3');
