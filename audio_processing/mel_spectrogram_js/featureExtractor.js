const config = require('./config')
const spectogram = require('./spectogram');
const tf = require('@tensorflow/tfjs');
const filters = require('./filters');

function FeatureExtractor(
  sample_rate = 16000,
  spectrogram = null,
  n_fft = 2048,
  hop_length = 512,
  win_length = null,
  window = 'hann',
  center = true,
  pad_mode = 'reflect',
  power = 2.0,
  n_mels=128,
  f_min=0.0,
  f_max=null
) {
  // console.log("~~ arguments ~~");
  //
  // console.log("sample_rate: ", sample_rate);
  // console.log("spectrogram: ", spectrogram);
  // console.log("n_fft: ", n_fft);
  // console.log("hop_length: ", hop_length);
  // console.log("win_length: ", win_length);
  // console.log("window: ", window);
  // console.log("center: ", center);
  // console.log("pad_mode: ", pad_mode);
  // console.log("power: ", power);

  this.sample_rate = sample_rate;
  this.spectrogram = spectrogram;
  this.n_fft = n_fft;
  this.hop_length = hop_length;
  this.win_length = win_length;
  this.window = window;
  this.center = center;
  this.pad_mode = pad_mode;
  this.power = power;
  this.n_mels = n_mels; // mel filter
  this.f_min = f_min; // mel filter
  this.f_max = f_max; // mel filter
}


FeatureExtractor.prototype.extract = function(x) {

  console.log("~~ arguments ~~");

  console.log("sample_rate: ", this.sample_rate);
  console.log("spectrogram: ", this.spectrogram);
  console.log("n_fft: ", this.n_fft);
  console.log("hop_length: ", this.hop_length);
  console.log("win_length: ", this.win_length);
  console.log("window: ", this.window);
  console.log("center: ", this.center);
  console.log("pad_mode: ", this.pad_mode);
  console.log("power: ", this.power);
  console.log("n_mel: ", this.n_mel);
  console.log("f_min: ", this.f_min);
  console.log("f_max: ", this.f_max);

  if (!(x instanceof tf.Tensor)) {
    x = tf.tensor(x);
  }

  let result = spectogram(
    y=x,
    S=null,
    n_fft=this.n_fft,
    hop_length=this.hop_length,
    power=this.power,
    win_length=this.win_length,
    window=this.window,
    center=this.center,
    pad_mode=this.pad_mode
  );
  S = result['S'];
  n_fft = result['n_fft'];

  // Build a Mel filter
  let mel_basis = filters.mel(
    sr=this.sample_rate,
    n_fft=this.n_fft,
    n_mels=this.n_mels,
    fmin=this.f_min,
    fmax=this.f_max
  );

  return mel_basis.dot(S);
}

module.exports = FeatureExtractor
