# Adapted from https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md

import tensorflow as tf
import numpy as np
import resampy
import pkg_resources
import os
import hashlib
import tf_slim

from urllib.request import urlretrieve


def md5_file(fname):
    hsh = hashlib.md5(open(fname, 'rb').read())
    return hsh.hexdigest()


# Architectural constants.
NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.

# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 0.96     # with zero overlap.

# Parameters used for embedding postprocessing.
PCA_EIGEN_VECTORS_NAME = 'pca_eigen_vectors'
PCA_MEANS_NAME = 'pca_means'
QUANTIZE_MIN_VAL = -2.0
QUANTIZE_MAX_VAL = +2.0

# Hyperparameters used in training.
INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.

# Names of ops, tensors, and features.
INPUT_OP_NAME = 'vggish/input_features'
INPUT_TENSOR_NAME = INPUT_OP_NAME + ':0'
OUTPUT_OP_NAME = 'vggish/embedding'
OUTPUT_TENSOR_NAME = OUTPUT_OP_NAME + ':0'
AUDIO_EMBEDDING_FEATURE_NAME = 'audio_embedding'

START_TIME = 'start_time_seconds'
VIDEO_ID = 'video_id'
LABELS = 'labels'
TIME = 'time'

MD5_CHECKSUMS = {
    'vggish_model.ckpt': 'd1c7011e6366aa34176bb05c705e31a8',
    'vggish_pca_params.npz': 'c80cae691033abe7c7ecd11ea39fc834',
}

MODEL_PARAMS = pkg_resources.resource_filename(
    __name__, '../../../models/vggish_model.ckpt'
)
PCA_PARAMS = pkg_resources.resource_filename(
    __name__, '../../../models/vggish_pca_params.npz'
)

for fname in MODEL_PARAMS, PCA_PARAMS:
    if not os.path.exists(fname):
        print('Model data not found, dowloading it...')
        urlretrieve(
            'https://storage.googleapis.com/audioset/vggish_model.ckpt',
            MODEL_PARAMS,
        )
        urlretrieve(
            'https://storage.googleapis.com/audioset/vggish_pca_params.npz',
            PCA_PARAMS,
        )
        print('Download completed')

    fbase = os.path.basename(fname)
    if md5_file(fname) != MD5_CHECKSUMS[fbase]:
        raise RuntimeError(
            '### VGGish model checksums do not match! ###\n\n'
            'Re-run `./scripts/download-deps.sh`, and open an issue at \n'
            'https://github.com/cosmir/openmic-2018/issues/new if that \n'
            "doesn't resolve the problem.\n"
        )


class Postprocessor(object):
    """Post-processes VGGish embeddings.

    The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as
    well) and 8-bit quantization to the raw embedding output from VGGish, in
    order to stay compatible with the YouTube-8M project which provides visual
    embeddings in the same format for a large set of YouTube videos. This
    class implements the same PCA (with whitening) and quantization
    transformations.
    """

    def __init__(self, pca_params_npz_path):
        """Constructs a postprocessor.

        Args:
            pca_params_npz_path: Path to a NumPy-format .npz file that
            contains the PCA parameters used in postprocessing.
        """
        with np.load(pca_params_npz_path) as data:
            self._pca_matrix = data[PCA_EIGEN_VECTORS_NAME]
            # Load means into a column vector for easier broadcasting later.
            self._pca_means = data[PCA_MEANS_NAME].reshape(-1, 1)

        assert self._pca_matrix.shape == (
            EMBEDDING_SIZE,
            EMBEDDING_SIZE,
        ), 'Bad PCA matrix shape: %r' % (self._pca_matrix.shape,)
        assert self._pca_means.shape == (
            EMBEDDING_SIZE,
            1,
        ), 'Bad PCA means shape: %r' % (self._pca_means.shape,)

    def postprocess(self, embeddings_batch):
        """Applies postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An nparray of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          An nparray of the same shape as the input but of type uint8,
          containing the PCA-transformed and quantized version of the input.
        """
        assert (
            len(embeddings_batch.shape) == 2
        ), 'Expected 2-d batch, got %r' % (embeddings_batch.shape,)
        assert (
            embeddings_batch.shape[1] == EMBEDDING_SIZE
        ), 'Bad batch shape: %r' % (embeddings_batch.shape,)

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = np.dot(
            self._pca_matrix, (embeddings_batch.T - self._pca_means)
        ).T

        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = np.clip(
            pca_applied, QUANTIZE_MIN_VAL, QUANTIZE_MAX_VAL
        )
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = (clipped_embeddings - QUANTIZE_MIN_VAL) * (
            255.0 / (QUANTIZE_MAX_VAL - QUANTIZE_MIN_VAL)
        )
        # - cast 8-bit float to uint8
        quantized_embeddings = quantized_embeddings.astype(np.uint8)

        return quantized_embeddings


__pproc__ = Postprocessor(PCA_PARAMS)
postprocess = __pproc__.postprocess


def waveform_to_features(data, sample_rate, compress=True):
    """Converts an audio waveform to VGGish features, with or without
    PCA compression.

    Parameters
    ----------
    data : np.array of either one dimension (mono) or two dimensions (stereo)

    sample_rate:
        Sample rate of the audio data

    compress : bool
        If True, PCA and quantization are applied to the features.
        If False, the features are taken directly from the model output

    Returns
    -------
    time_points : np.ndarray, len=n
        Time points in seconds of the features

    features : np.ndarray, shape=(n, 128)
        The output features, with or without PCA compression and quantization.
    """

    examples = waveform_to_examples(data, sample_rate)

    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        time_points, features = transform(examples, sess)

        if compress:
            features_z = postprocess(features)
            return time_points, features_z

        return time_points, features


def waveform_to_examples(data, sample_rate):
    """Converts audio waveform into an array of examples for VGGish.

    Args:
        data: np.array of either one dimension (mono) or two dimensions
        (multi-channel, with the outer dimension representing channels).
        Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
        sample_rate: Sample rate of data.

    Returns:
        3-D np.array of shape [num_examples, num_frames, num_bands] which
        represents a sequence of examples, each of which contains a patch of
        log mel spectrogram, covering num_frames frames of audio and num_bands
        mel frequency bands, where the frame length is
        STFT_HOP_LENGTH_SECONDS.
    """
    # Convert to mono.
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    # Resample to the rate assumed by VGGish.
    if sample_rate != SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = log_mel_spectrogram(
        data,
        audio_sample_rate=SAMPLE_RATE,
        log_offset=LOG_OFFSET,
        window_length_secs=STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=NUM_MEL_BINS,
        lower_edge_hertz=MEL_MIN_HZ,
        upper_edge_hertz=MEL_MAX_HZ,
    )

    # Frame features into examples.
    features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS
    example_window_length = int(
        round(EXAMPLE_WINDOW_SECONDS * features_sample_rate)
    )
    example_hop_length = int(round(EXAMPLE_HOP_SECONDS * features_sample_rate))
    log_mel_examples = frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length,
    )
    return log_mel_examples


def frame(data, window_length, hop_length):
    """Convert array into a sequence of successive possibly overlapping frames.

    An n-dimensional array of shape (num_samples, ...) is converted into an
    (n+1)-D array of shape (num_frames, window_length, ...), where each frame
    starts hop_length points after the preceding one.

    This is accomplished using stride_tricks, so the original data is not
    copied.  However, there is no zero-padding, so any incomplete frames at the
    end are not included.

    Args:
      data: np.array of dimension N >= 1.
      window_length: Number of samples in each frame.
      hop_length: Advance (in samples) between each window.

    Returns:
      (N+1)-D np.array with as many rows as there are complete frames that can
      be extracted.
    """
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def periodic_hann(window_length):
    """Calculate a "periodic" Hann window.

    The classic Hann window is defined as a raised cosine that starts and
    ends on zero, and where every value appears twice, except the middle
    point for an odd-length window.  Matlab calls this a "symmetric" window
    and np.hanning() returns it.  However, for Fourier analysis, this
    actually represents just over one cycle of a period N-1 cosine, and
    thus is not compactly expressed on a length-N Fourier basis.  Instead,
    it's better to use a raised cosine that ends just before the final
    zero value - i.e. a complete cycle of a period-N cosine.  Matlab
    calls this a "periodic" window. This routine calculates it.

    Args:
      window_length: The number of points in the returned window.

    Returns:
      A 1D np.array containing the periodic hann window.
    """
    return 0.5 - (
        0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length))
    )


def stft_magnitude(signal, fft_length, hop_length=None, window_length=None):
    """Calculate the short-time Fourier transform magnitude.

    Args:
      signal: 1D np.array of the input time-domain signal.
      fft_length: Size of the FFT to apply.
      hop_length: Advance (in samples) between each frame passed to FFT.
      window_length: Length of each block of samples to pass to FFT.

    Returns:
      2D np.array where each row contains the magnitudes of the fft_length/2+1
      unique values of the FFT for the corresponding frame of input samples.
    """
    frames = frame(signal, window_length, hop_length)
    # Apply frame window to each frame. We use a periodic Hann (cosine of
    # period window_length) instead of the symmetric Hann of np.hanning (period
    # window_length-1).
    window = periodic_hann(window_length)
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


# Mel spectrum constants and functions.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def hertz_to_mel(frequencies_hertz):
    """Convert frequencies to mel scale using HTK formula.

    Args:
      frequencies_hertz: Scalar or np.array of frequencies in hertz.

    Returns:
      Object of same size as frequencies_hertz containing corresponding values
      on the mel scale.
    """
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ)
    )


def spectrogram_to_mel_matrix(
    num_mel_bins=20,
    num_spectrogram_bins=129,
    audio_sample_rate=8000,
    lower_edge_hertz=125.0,
    upper_edge_hertz=3800.0,
):
    """Return a matrix that can post-multiply spectrogram rows to make mel.

    Returns a np.array matrix A that can be used to post-multiply a matrix S of
    spectrogram values (STFT magnitudes) arranged as frames x bins to generate
    a "mel spectrogram" M of frames x num_mel_bins.  M = S A.

    The classic HTK algorithm exploits the complementarity of adjacent mel
    bands to multiply each FFT bin by only one mel weight, then add it, with
    positive and negative signs, to the two adjacent mel bands to which that
    bin contributes.  Here, by expressing this operation as a matrix multiply,
    we go from num_fft multiplies per frame (plus around 2*num_fft adds) to
    around num_fft^2 multiplies and adds.  However, because these are all
    presumably accomplished in a single call to np.dot(), it's not clear which
    approach is faster in Python.  The matrix multiplication has the
    attraction of being more general and flexible, and much easier to read.

    Args:
      num_mel_bins: How many bands in the resulting mel spectrum.  This is
        the number of columns in the output matrix.
      num_spectrogram_bins: How many bins there are in the source spectrogram
        data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
        only contains the nonredundant FFT bins.
      audio_sample_rate: Samples per second of the audio at the input to the
        spectrogram. We need this to figure out the actual frequencies for
        each spectrogram bin, which dictates how they are mapped into mel.
      lower_edge_hertz: Lower bound on the frequencies to be included in the
        mel spectrum.  This corresponds to the lower edge of the lowest
        triangular band.
      upper_edge_hertz: The desired top edge of the highest frequency band.

    Returns:
      An np.array with shape (num_spectrogram_bins, num_mel_bins).

    Raises:
      ValueError: if frequency edges are incorrectly ordered.
    """
    nyquist_hertz = audio_sample_rate / 2.0
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError(
            'lower_edge_hertz %.1f >= upper_edge_hertz %.1f'
            % (lower_edge_hertz, upper_edge_hertz)
        )
    spectrogram_bins_hertz = np.linspace(
        0.0, nyquist_hertz, num_spectrogram_bins
    )
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    # The i'th mel band (starting from i=1) has center frequency
    # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
    # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
    # the band_edges_mel arrays.
    band_edges_mel = np.linspace(
        hertz_to_mel(lower_edge_hertz),
        hertz_to_mel(upper_edge_hertz),
        num_mel_bins + 2,
    )
    # Matrix to post-multiply feature arrays whose rows are
    # num_spectrogram_bins of spectrogram values.
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i : i + 3]
        # Calculate lower and upper slopes for every spectrogram bin.
        # Line segments are linear in the *mel* domain, not hertz.
        lower_slope = (spectrogram_bins_mel - lower_edge_mel) / (
            center_mel - lower_edge_mel
        )
        upper_slope = (upper_edge_mel - spectrogram_bins_mel) / (
            upper_edge_mel - center_mel
        )
        # .. then intersect them with each other and zero.
        mel_weights_matrix[:, i] = np.maximum(
            0.0, np.minimum(lower_slope, upper_slope)
        )
    # HTK excludes the spectrogram DC bin; make sure it always gets a zero
    # coefficient.
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def log_mel_spectrogram(
    data,
    audio_sample_rate=8000,
    log_offset=0.0,
    window_length_secs=0.025,
    hop_length_secs=0.010,
    **kwargs
):
    """Convert waveform to a log magnitude mel-frequency spectrogram.

    Args:
      data: 1D np.array of waveform data.
      audio_sample_rate: The sampling rate of data.
      log_offset: Add this to values when taking log to avoid -Infs.
      window_length_secs: Duration of each window to analyze.
      hop_length_secs: Advance between successive analysis windows.
      **kwargs: Additional arguments to pass to spectrogram_to_mel_matrix.

    Returns:
      2D np.array of (num_frames, num_mel_bins) consisting of log mel
      filterbank magnitudes for successive frames.
    """
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    spectrogram = stft_magnitude(
        data,
        fft_length=fft_length,
        hop_length=hop_length_samples,
        window_length=window_length_samples,
    )
    mel_spectrogram = np.dot(
        spectrogram,
        spectrogram_to_mel_matrix(
            num_spectrogram_bins=spectrogram.shape[1],
            audio_sample_rate=audio_sample_rate,
            **kwargs
        ),
    )
    return np.log(mel_spectrogram + log_offset)


def transform(examples, sess):
    """Compute VGGish features for an iterable of examples.

    Parameters
    ----------
    examples : iterable of tf.Examples
        Examples to process by the model.
        See openmic.vggish.inputs.{soundfile_to_examples, waveform_to_examples}

    sess : tf.Session
        Open tensorflow session.

    Returns
    -------
    time_points : np.ndarray, len=n
        Time points in seconds of the feature vector.

    features : np.ndarray, shape=(n, 128), dtype=np.uint8
        VGGish feature array.
    """
    define_vggish_slim(training=False)
    load_vggish_slim_checkpoint(sess, MODEL_PARAMS)

    features_tensor = sess.graph.get_tensor_by_name(INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(OUTPUT_TENSOR_NAME)

    [features] = sess.run(
        [embedding_tensor], feed_dict={features_tensor: examples}
    )

    time_points = np.arange(len(features)) * EXAMPLE_HOP_SECONDS

    return time_points, features


def define_vggish_slim(training=False):
    """Defines the VGGish TensorFlow model.

    All ops are created in the current default graph, under the scope
    'vggish/'.

    The input is a placeholder named 'vggish/input_features' of type float32
    and shape [batch_size, num_frames, num_bands] where batch_size is variable
    and num_frames and num_bands are constants, and [num_frames, num_bands]
    represents a log-mel-scale spectrogram patch covering num_bands frequency
    bands and num_frames time frames (where each frame step is usually 10ms).
    This is produced by computing the stabilized log(mel-spectrogram +
    LOG_OFFSET). The output is an op named 'vggish/embedding' which
    produces the activations of a 128-D embedding layer, which is usually the
    penultimate layer when used as part of a full model with a final
    classifier layer.

    Args:
      training: If true, all parameters are marked trainable.

    Returns:
      The op 'vggish/embeddings'.
    """
    # Defaults:
    # - All weights are initialized to N(0, INIT_STDDEV).
    # - All biases are initialized to 0.
    # - All activations are ReLU.
    # - All convolutions are 3x3 with stride 1 and SAME padding.
    # - All max-pools are 2x2 with stride 2 and SAME padding.
    with tf_slim.arg_scope(
        [tf_slim.conv2d, tf_slim.fully_connected],
        weights_initializer=tf.compat.v1.truncated_normal_initializer(
            stddev=INIT_STDDEV
        ),
        biases_initializer=tf.compat.v1.zeros_initializer(),
        activation_fn=tf.nn.relu,
        trainable=training,
    ), tf_slim.arg_scope(
        [tf_slim.conv2d], kernel_size=[3, 3], stride=1, padding='SAME'
    ), tf_slim.arg_scope(
        [tf_slim.max_pool2d], kernel_size=[2, 2], stride=2, padding='SAME'
    ), tf.compat.v1.variable_scope(
        'vggish'
    ):

        # Input: a batch of 2-D log-mel-spectrogram patches.
        features = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None, NUM_FRAMES, NUM_BANDS),
            name='input_features',
        )
        # Reshape to 4-D so that we can convolve a batch with conv2d().
        net = tf.reshape(features, [-1, NUM_FRAMES, NUM_BANDS, 1])

        # The VGG stack of alternating convolutions and max-pools.
        net = tf_slim.conv2d(net, 64, scope='conv1')
        net = tf_slim.max_pool2d(net, scope='pool1')
        net = tf_slim.conv2d(net, 128, scope='conv2')
        net = tf_slim.max_pool2d(net, scope='pool2')
        net = tf_slim.repeat(net, 2, tf_slim.conv2d, 256, scope='conv3')
        net = tf_slim.max_pool2d(net, scope='pool3')
        net = tf_slim.repeat(net, 2, tf_slim.conv2d, 512, scope='conv4')
        net = tf_slim.max_pool2d(net, scope='pool4')

        # Flatten before entering fully-connected layers
        net = tf_slim.flatten(net)
        net = tf_slim.repeat(
            net, 2, tf_slim.fully_connected, 4096, scope='fc1'
        )
        # The embedding layer.
        net = tf_slim.fully_connected(net, EMBEDDING_SIZE, scope='fc2')
        return tf.identity(net, name='embedding')


def load_vggish_slim_checkpoint(session, checkpoint_path):
    """Loads a pre-trained VGGish-compatible checkpoint.

    This function can be used as an initialization function (referred to as
    init_fn in TensorFlow documentation) which is called in a Session after
    initializating all variables. When used as an init_fn, this will load
    a pre-trained checkpoint that is compatible with the VGGish model
    definition. Only variables defined by VGGish will be loaded.

    Args:
      session: an active TensorFlow session.
      checkpoint_path: path to a file containing a checkpoint that is
        compatible with the VGGish model definition.
    """
    # Get the list of names of all VGGish variables that exist in
    # the checkpoint (i.e., all inference-mode VGGish variables).
    with tf.Graph().as_default():
        define_vggish_slim(training=False)
        vggish_var_names = [v.name for v in tf.compat.v1.global_variables()]

    # Get the list of all currently existing variables that match
    # the list of variable names we just computed.
    vggish_vars = [
        v
        for v in tf.compat.v1.global_variables()
        if v.name in vggish_var_names
    ]

    # Use a Saver to restore just the variables selected above.
    saver = tf.compat.v1.train.Saver(
        vggish_vars, name='vggish_load_pretrained'
    )
    saver.restore(session, checkpoint_path)
