const tf = require('@tensorflow/tfjs');
const filters = require('./filters');
const util = require('./util');

// https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py#L2316

function stft(
  y,
  n_fft=2048,
  hop_length=null,
  win_length=null,
  window='hann',
  center=True,
  pad_mode='reflect'
) {

  // By default, use the entire frame
  if (win_length == null) {
    win_length = n_fft
  }
  console.log('stft - win_length: ', win_length);

  // Set the default hop, if it's not already specified
  if (hop_length == null) {
    hop_length = Math.floor(win_length / 4);
  }
  console.log('stft - hop_length: ', hop_length);

  let fft_window = filters.get_window(window, win_length, fftbins=true);
  console.log('stft - fft_window: ', fft_window.shape);
  // fft_window.print();

  // Pad the window out to n_fft size
  fft_window = util.pad_center(fft_window, n_fft)
  console.log('stft - fft_window (padded): ', fft_window.shape);
  // fft_window.print();

  // Pad the time series so that frames are centered
  if (center) {
    y = util.pad_reflect(y, y.shape[0] + n_fft)
    // instead of padding with reflect, simply pad center with zeros
    console.log('stft - y (padded): ', y.shape);
  }

  // Window the time series.
  y_frames = tf.signal.frame(y, frameLength=n_fft, frameStep=hop_length)
  console.log('stft - y_frames: ', y_frames.shape);

  let windowed = y_frames.mul(fft_window)
  console.log('windowed: ', windowed.shape);

  transformed = tf.spectral.rfft(windowed)
  console.log('transformed: ', transformed.shape);

  return transformed;
}

function spectogram(
  y=null,
  S=null,
  n_fft=2048,
  hop_length=512,
  power=1,
  win_length=null,
  window='hann',
  center=true,
  pad_mode='reflect'
) {

  if (S != null) {
    n_fft = 2 * (S.shape[0] - 1)
  } else {
    transformed = stft(
      y,
      n_fft=n_fft,
      hop_length=hop_length,
      win_length=win_length,
      window=window,
      center=center,
      pad_mode=pad_mode
    )

    S = transformed.abs().pow(power).transpose();
  }

  let results = {
    'S': S,
    'n_fft': n_fft
  }

  return results
}

module.exports = spectogram
