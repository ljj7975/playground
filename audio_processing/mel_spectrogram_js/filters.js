const tf = require('@tensorflow/tfjs');
const util = require('./util');

module.exports.get_window = function(
  window,
  Nx,
  fftbins=True
) {
  // we only support hanning
  return get_hanning_window(Nx);
}

function get_hanning_window(
  M
) {
  M += 1
  let fac = tf.linspace(-Math.PI, Math.PI, M);

  let k = tf.scalar(0);
  let a1 = tf.scalar(0.5);
  let w1 = fac.mul(k).cos().mul(a1);

  k = tf.scalar(1);
  let a2 = tf.scalar(0.5);
  let w2 = fac.mul(k).cos().mul(a2);

  let w = w1.add(w2);

  w = tf.slice(w, 0, M-1);
  return w;
}


module.exports.mel = function(
  sr,
  n_fft,
  n_mels=128,
  fmin=0.0,
  fmax=null,
  htk=false,
) {
  if (fmax == null) {
    console.log('??')
    fmax = sr/2;
  }
  fmax = tf.scalar(fmax)

  // Center freqs of each FFT bin
  let fftfreqs = tf.linspace(0, sr/2, 1 + Math.floor(n_fft/2))
  console.log('mel - fftfreqs: ', fftfreqs.shape);
  fftfreqs.mean().print()

  // 'Center freqs' of mel bands - uniformly spaced between limits
  let min_mel = util.hz_to_mel(fmin, htk).dataSync()[0];
  console.log('mel - min_mel: ', min_mel);
  let max_mel = util.hz_to_mel(fmax, htk).dataSync()[0];
  console.log('mel - max_mel: ', max_mel);

  let mels = tf.linspace(min_mel, max_mel, n_mels+2);
  console.log('mel - mels: ', mels.shape);

  let mel_f = util.mel_to_hz(mels, htk);
  console.log('mel - mel_f: ', mel_f.shape);
  mel_f.mean().print()

  let former = mel_f.slice(0, mel_f.shape[0] - 1);
  let latter = mel_f.slice(1, mel_f.shape[0] - 1);
  let fdiff = latter.sub(former);
  console.log('mel - fdiff: ', fdiff.shape);
  fdiff.mean().print()

  let ramps = mel_f.reshape([-1, 1]).sub(fftfreqs)
  console.log('mel - ramps: ', ramps.shape);
  ramps.mean().print()

  let weights = []
  let zero_vec = tf.zeros([fftfreqs.shape[0]]);

  for (var i = 0; i < n_mels; i++) {
    // lower and upper slopes for all bins
    let lower = ramps.slice(i, 1).flatten().mul(-1).div(fdiff.slice(i, 1))
    let upper = ramps.slice(i+2, 1).flatten().div(fdiff.slice(i+1, 1))

    // .. then intersect them with each other and zero
    let indices = lower.greaterEqual(upper);
    let row = upper.where(indices, lower);
    indices = row.greaterEqual(zero_vec);
    row = row.where(indices, zero_vec);

    weights.push(row)
  }

  let enorm = tf.scalar(2.0).div(mel_f.slice(2, n_mels).sub(mel_f.slice(0, n_mels)))
  console.log('mel - enorm: ', enorm.shape);

  weights = tf.stack(weights).transpose().mul(enorm).transpose()
  console.log('mel - weights: ', weights.shape);
  weights.mean().print()

  return weights;
}
