const FeatureExtractor = require('./featureExtractor');
var fs = require('fs');
var WaveFile = require('wavefile').WaveFile;

file_name = "/Users/jaejunlee/Documents/playground/audio_processing/sample_audio/go/0a9f9af7_nohash_0.wav"

let buffer = fs.readFileSync(file_name);
let wav = new WaveFile(buffer);
wav.toBitDepth("32f");
// in case the file has different sample rate
// wav.toSampleRate(this.sampleRate);

let x = [].slice.call(wav.getSamples());

let featureExtractor = new FeatureExtractor(
  sampleRate = 16000,
  spectrogram = null,
  n_fft = 480,
  hop_length = 160,
  win_length = null,
  window = 'hann',
  center = true,
  pad_mode = 'reflect',
  power = 2.0,
  n_mels= 40,
  f_min= 20,
  f_max= 4000
);

let mel_spectogram = featureExtractor.extract(x)
mel_spectogram.print()
