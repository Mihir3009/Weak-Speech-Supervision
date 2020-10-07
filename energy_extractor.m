
function energy = energy_extractor(wav_path)

% Read Wav file
[y, fs] = audioread('./1.wav');

% Original values into integer
y = y * 2.^15;

% Assuming Fs is integer.
frame_length = fs/40; 

% Frame length of 400 ms (for Fs = 16 kHz)
n_samples = length(y);
trailing_samples = mod(n_samples, frame_length);
frames = reshape( y(1:end-trailing_samples), frame_length, []);
n_frames = length(frames(1,:));

%% -----------------------Hamming Windowing--------------------------------
h = hamming(frame_length);
h = repmat(h,1,n_frames);
w = frames.*h;

%% -----------------------Energy computation-------------------------------
energy = sum(w.^2);
