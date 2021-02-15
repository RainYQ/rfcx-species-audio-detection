# modified from
import random
import os
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import efficientnet.tfkeras as efn
import numpy as np
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from IPython.display import Audio
from classification_models.keras import Classifiers
import librosa.display
import matplotlib.patches as patches
from IPython.display import FileLink
from GroupNormalization import GroupNormalization
import keras.applications as kapplications

print(tf.__version__)

SEED = 42

train_tp = "./train_tp.csv"
train_fp = "./train_fp.csv"
train_data_tp = pd.read_csv(train_tp)
train_data_fp = pd.read_csv(train_fp)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(SEED)

# Configs

ResNet34, preprocess_input = Classifiers.get('resnet34')

cfg = {
    'parse_params': {
        'cut_time': 12,
    },
    'data_params': {
        'sample_time': 10,  # assert 60 % sample_time == 0
        'spec_fmax': 24000.0,
        'spec_fmin': 40.0,
        'spec_mel': 256,
        'mel_power': 2,
        'img_shape': (256, 768)
    },
    'model_params': {
        'batchsize_per_tpu': 8,
        'iteration_per_epoch': 256,
        'epoch': 150,
        'arch': ResNet34,
        'arch_preprocess': preprocess_input,
        'freeze_to': 0,  # Freeze to backbone.layers[:freeze_to]. If None, all layers in the backbone will be freezed.
        'loss': {
            # using Label Smooth
            'label_smooth': True,
            'eps': 0.05,
            'fn': tfa.losses.SigmoidFocalCrossEntropy,
            'params': {},
        },
        'optim': {
            'fn': tfa.optimizers.RectifiedAdam,
            'params': {'lr': 1e-4, 'total_steps': 150 * 256, 'warmup_proportion': 0.3, 'min_lr': 1e-7},
        },
        'mixup': True,  # False
        'multi-label': True
    }
}
AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_DS_PATH = './'
TRAIN_TFREC = "./tfrecords/train"
TEST_TFREC = "./tfrecords/test"
CUT = cfg['parse_params']['cut_time']  # CUT = 10
SR = 48000  # all wave's sample rate may be 48k
TIME = cfg['data_params']['sample_time']  # TIME = 6
FMAX = cfg['data_params']['spec_fmax']
FMIN = cfg['data_params']['spec_fmin']
N_MEL = cfg['data_params']['spec_mel']
HEIGHT, WIDTH = cfg['data_params']['img_shape']
CLASS_N = 24
raw_dataset = tf.data.TFRecordDataset([TRAIN_TFREC + '/00-148.tfrec'])
# 定义Feature结构, 告诉解码器每个Feature的类型和默认值
feature_description = {
    'recording_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'audio_wav': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label_info': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

parse_dtype = {
    'audio_wav': tf.float32,
    'recording_id': tf.string,
    'species_id': tf.int32,
    'songtype_id': tf.int32,
    't_min': tf.float32,
    'f_min': tf.float32,
    't_max': tf.float32,
    'f_max': tf.float32,
    'is_tp': tf.int32,
}


@tf.function
# 将TFRecord文件中的每一个序列化的tf.train.Example解码
def _parse_function(example_proto):
    sample = tf.io.parse_single_example(example_proto, feature_description)
    wav, _ = tf.audio.decode_wav(sample['audio_wav'], desired_channels=1)  # mono
    label_info = tf.strings.split(sample['label_info'], sep='"')[1]
    labels = tf.strings.split(label_info, sep=';')

    @tf.function
    def _cut_audio(label):
        items = tf.strings.split(label, sep=',')
        spid = tf.squeeze(tf.strings.to_number(items[0], tf.int32))
        soid = tf.squeeze(tf.strings.to_number(items[1], tf.int32))
        tmin = tf.squeeze(tf.strings.to_number(items[2]))
        fmin = tf.squeeze(tf.strings.to_number(items[3]))
        tmax = tf.squeeze(tf.strings.to_number(items[4]))
        fmax = tf.squeeze(tf.strings.to_number(items[5]))
        tp = tf.squeeze(tf.strings.to_number(items[6], tf.int32))

        tmax_s = tmax * tf.cast(SR, tf.float32)
        tmin_s = tmin * tf.cast(SR, tf.float32)
        cut_s = tf.cast(CUT * SR, tf.float32)
        all_s = tf.cast(60 * SR, tf.float32)
        tsize_s = tmax_s - tmin_s
        # cut_s 10s
        # [tmin_s - (cut_s - tsize_s) / 2, tmax_s + (cut_s - tsize_s) / 2]
        # center crop
        # 右端点不能超出边界 tf.minimum(tmax_s + (cut_s - tsize_s) / 2, all_s)
        # 必须保证可以cut到10s tf.minimum(tmin_s - (cut_s - tsize_s) / 2, tf.minimum(tmax_s + (cut_s - tsize_s) / 2, all_s) - cut_s)
        # 在保证中心裁切可以cut到10s的情况下使用center crop, 两边各扩展(cut_s - tsize_s) / 2
        # [cut_min, cut_max]
        cut_min = tf.cast(
            tf.maximum(0.0,
                       tf.minimum(tmin_s - (cut_s - tsize_s) / 2,
                                  tf.minimum(tmax_s + (cut_s - tsize_s) / 2, all_s) - cut_s)
                       ), tf.int32
        )
        cut_max = cut_min + CUT * SR
        _sample = {
            'audio_wav': tf.reshape(wav[cut_min:cut_max], [CUT * SR]),
            'recording_id': sample['recording_id'],
            'species_id': spid,
            'songtype_id': soid,
            't_min': tmin - tf.cast(cut_min, tf.float32) / tf.cast(SR, tf.float32),
            'f_min': fmin,
            't_max': tmax - tf.cast(cut_min, tf.float32) / tf.cast(SR, tf.float32),
            'f_max': fmax,
            'is_tp': tp,
        }
        return _sample

    samples = tf.map_fn(_cut_audio, labels, dtype=parse_dtype)
    return samples


parsed_dataset = raw_dataset.map(_parse_function).unbatch()


@tf.function
def _cut_wav(x):
    # random cut in training
    # 在训练阶段随机裁剪到6s
    # CUT: 10s TIME: 6s
    # delta: 4s
    # 找一个0-4s的随机数
    cut_min = tf.random.uniform([], maxval=(CUT - TIME) * SR, dtype=tf.int32)
    cut_max = cut_min + TIME * SR
    cutwave = tf.reshape(x['audio_wav'][cut_min:cut_max], [TIME * SR])
    y = {}
    # 更新x中的audio_wav
    y.update(x)
    y['audio_wav'] = cutwave
    y['t_min'] = tf.maximum(0.0, x['t_min'] - tf.cast(cut_min, tf.float32) / SR)
    y['t_max'] = tf.maximum(0.0, x['t_max'] - tf.cast(cut_min, tf.float32) / SR)
    return y


@tf.function
def _cut_wav_val(x):
    # 验证集永远是中心裁切
    # center crop in validation
    cut_min = (CUT - TIME) * SR // 2
    cut_max = cut_min + TIME * SR
    cutwave = tf.reshape(x['audio_wav'][cut_min:cut_max], [TIME * SR])
    y = {}
    y.update(x)
    y['audio_wav'] = cutwave
    y['t_min'] = tf.maximum(0.0, x['t_min'] - tf.cast(cut_min, tf.float32) / SR)
    y['t_max'] = tf.maximum(0.0, x['t_max'] - tf.cast(cut_min, tf.float32) / SR)
    return y


@tf.function
def _filtTP(x):
    return x['is_tp'] == 1


@tf.function
def _filtFP(x):
    return x['is_tp'] == 0


def show_wav(sample, ax):
    wav = sample["audio_wav"].numpy()
    rate = SR
    ax.plot(np.arange(len(wav)) / rate, wav)
    ax.set_title(
        sample["recording_id"].numpy().decode()
        + ("/%d" % sample["species_id"])
        + ("TP" if sample["is_tp"] else "FP"))

    return Audio((wav * 2 ** 15).astype(np.int16), rate=rate)


# fig, ax = plt.subplots(figsize=(15, 3))
# show_wav(next(iter(parsed_dataset)), ax)


@tf.function
def _wav_to_spec(x):
    mel_power = cfg['data_params']['mel_power']

    stfts = tf.signal.stft(x["audio_wav"], frame_length=2048, frame_step=512, fft_length=2048)
    spectrograms = tf.abs(stfts) ** mel_power

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = FMIN, FMAX, N_MEL

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, SR, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    y = {
        'audio_spec': tf.transpose(log_mel_spectrograms),  # (num_mel_bins, frames)
    }
    y.update(x)
    return y


spec_dataset = parsed_dataset.filter(_filtTP).map(_cut_wav).map(_wav_to_spec)


# plt.figure(figsize=(12, 5))
# for i, s in enumerate(spec_dataset.take(3)):
#     plt.subplot(1, 3, i + 1)
#     plt.imshow(s['audio_spec'])
# plt.show()


def show_spectrogram(sample, ax, showlabel=False):
    S_dB = sample["audio_spec"].numpy()
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=SR,
                                   fmax=FMAX, fmin=FMIN, ax=ax, cmap='magma')
    ax.set(title=f'Mel-frequency spectrogram of {sample["recording_id"].numpy().decode()}')
    sid, fmin, fmax, tmin, tmax, istp = (
        sample["species_id"], sample["f_min"], sample["f_max"], sample["t_min"], sample["t_max"], sample["is_tp"])
    ec = '#00ff00' if istp == 1 else '#0000ff'
    ax.add_patch(
        patches.Rectangle(xy=(tmin, fmin), width=tmax - tmin, height=fmax - fmin, ec=ec, fill=False)
    )

    if showlabel:
        ax.text(tmin, fmax,
                f"{sid.numpy().item()} {'tp' if istp == 1 else 'fp'}",
                horizontalalignment='left', verticalalignment='bottom', color=ec, fontsize=16)


# fig, ax = plt.subplots(figsize=(15, 3))
# show_spectrogram(next(iter(spec_dataset)), ax, showlabel=True)

# in validation, annotations will come to the center
# fig, ax = plt.subplots(figsize=(15, 3))
# show_spectrogram(next(iter(parsed_dataset.filter(_filtTP).map(_cut_wav_val).map(_wav_to_spec))), ax, showlabel=True)

# for sample in spec_dataset.take(5):
#     fig, ax = plt.subplots(figsize=(15, 3))
#     show_spectrogram(sample, ax, showlabel=True)


# create labels
# RainYQ
# support multi-label
@tf.function
def _create_annot(x, training=False):
    targ = tf.one_hot(x["species_id"], CLASS_N, on_value=x["is_tp"], off_value=0)
    if training:
        if cfg['model_params']['loss']['label_smooth']:
            eps = cfg['model_params']['loss']['eps']
            targ = tf.cast(targ, tf.float32)
            targ = targ * (1 - eps) + eps / CLASS_N

    return {
        'input': x["audio_spec"],
        'target': tf.cast(targ, tf.float32)
    }


@tf.function
def _create_annot_train(x):
    return _create_annot(x, training=True)


@tf.function
def _create_annot_val(x):
    return _create_annot(x, training=False)


# annot_dataset = spec_dataset.map(_create_annot)


# Preprocessing and data augmentation
# 
# In training, I use
# 
# * gaussian noise
# * random flip left & right (NEW)
# * random brightness
# * specaugment


@tf.function
def _preprocess_img(x, training=False, test=False):
    image = tf.expand_dims(x, axis=-1)
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image = tf.image.per_image_standardization(image)

    @tf.function
    def _specaugment(image):
        ERASE_TIME = 50
        ERASE_MEL = 16
        image = tf.expand_dims(image, axis=0)
        xoff = tf.random.uniform([2], minval=ERASE_TIME // 2, maxval=WIDTH - ERASE_TIME // 2, dtype=tf.int32)
        xsize = tf.random.uniform([2], minval=ERASE_TIME // 2, maxval=ERASE_TIME, dtype=tf.int32)
        yoff = tf.random.uniform([2], minval=ERASE_MEL // 2, maxval=HEIGHT - ERASE_MEL // 2, dtype=tf.int32)
        ysize = tf.random.uniform([2], minval=ERASE_MEL // 2, maxval=ERASE_MEL, dtype=tf.int32)
        if xsize[0] % 2 != 0:
            a = tf.add(xsize[0], 1)
        else:
            a = xsize[0]
        if xsize[1] % 2 != 0:
            b = tf.add(xsize[1], 1)
        else:
            b = xsize[1]
        if ysize[0] % 2 != 0:
            c = tf.add(ysize[0], 1)
        else:
            c = ysize[0]
        if ysize[1] % 2 != 0:
            d = tf.add(ysize[1], 1)
        else:
            d = ysize[1]
        xsize = tf.stack([a, b])
        ysize = tf.stack([c, d])
        image = tfa.image.cutout(image, [HEIGHT, xsize[0]], offset=[HEIGHT // 2, xoff[0]])
        image = tfa.image.cutout(image, [HEIGHT, xsize[1]], offset=[HEIGHT // 2, xoff[1]])
        image = tfa.image.cutout(image, [ysize[0], WIDTH], offset=[yoff[0], WIDTH // 2])
        image = tfa.image.cutout(image, [ysize[1], WIDTH], offset=[yoff[1], WIDTH // 2])
        image = tf.squeeze(image, axis=0)
        return image

    if training:
        # gaussian
        gau = tf.keras.layers.GaussianNoise(0.3)
        image = tf.cond(tf.random.uniform([]) < 0.5, lambda: gau(image, training=True), lambda: image)
        # brightness
        image = tf.image.random_brightness(image, 0.2)
        # random left right flip (NEW)
        image = tf.image.random_flip_left_right(image)
        # specaugment
        image = tf.cond(tf.random.uniform([]) < 0.5, lambda: _specaugment(image), lambda: image)

    if test:
        # Insert augmentations for TTA here
        # image = tf.cond(tf.random.uniform([]) < 0.5, lambda: _specaugment(image), lambda: image)
        pass

    image = (image - tf.reduce_min(image)) / (
            tf.reduce_max(image) - tf.reduce_min(image)) * 255.0  # rescale to [0, 255]
    image = tf.image.grayscale_to_rgb(image)
    image = cfg['model_params']['arch_preprocess'](image)

    return image


@tf.function
def _preprocess(x):
    image = _preprocess_img(x['input'], training=True, test=False)
    return (image, x["target"])


@tf.function
def _preprocess_val(x):
    image = _preprocess_img(x['input'], training=False, test=False)
    return (image, x["target"])


@tf.function
def _preprocess_test(x):
    image = _preprocess_img(x['audio_spec'], training=False, test=True)
    return (image, x["recording_id"])


# for inp, targ in annot_dataset.map(_preprocess).take(2):
#     plt.imshow(inp.numpy()[:, :, 0])
#     t = targ.numpy()
#     if t.sum() == 0:
#         plt.title(f'FP')
#     else:
#         plt.title(f'{t.nonzero()[0]}')
#     plt.colorbar()
#     plt.show()


# Model

def create_model():
    # backbone = cfg['model_params']['arch'](include_top=False, weights='imagenet')
    # backbone = cfg['model_params']['arch']((HEIGHT, WIDTH, 3), include_top=False, weights='imagenet')
    # using NASNet
    backbone = tf.keras.applications.ResNet50V2(weights="imagenet", include_top=False,
                                                input_shape=(HEIGHT, WIDTH, 3), classes=24)
    # backbone = efn.EfficientNetB0(weights="imagenet", include_top=False,
    #                               input_shape=(HEIGHT, WIDTH, 3), classes=24)
    if cfg['model_params']['freeze_to'] is None:
        for layer in backbone.layers:
            layer.trainable = False
    else:
        for layer in backbone.layers[:cfg['model_params']['freeze_to']]:
            layer.trainable = False

    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalAveragePooling2D(),
        GroupNormalization(group=32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()),
        GroupNormalization(group=32),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(CLASS_N, bias_initializer=tf.keras.initializers.Constant(-2.))])
    return model


model = create_model()
model.summary()

tf.keras.utils.plot_model(model,
                          to_file='model.png',  # 模型结构图保存名字
                          show_layer_names=True,  # 是否显示层名
                          show_shapes=True)  # 是否显示层形状


@tf.function
def _mixup(inp, targ):
    indice = tf.range(len(inp))
    indice = tf.random.shuffle(indice)
    sinp = tf.gather(inp, indice, axis=0)
    starg = tf.gather(targ, indice, axis=0)

    alpha = 0.2
    t = tf.compat.v1.distributions.Beta(alpha, alpha).sample([len(inp)])
    tx = tf.reshape(t, [-1, 1, 1, 1])
    ty = tf.reshape(t, [-1, 1])
    x = inp * tx + sinp * (1 - tx)
    y = targ * ty + starg * (1 - ty)
    #     y = tf.minimum(targ + starg, 1.0) # for multi-label???
    return x, y


tfrecs = sorted(tf.io.gfile.glob(TRAIN_TFREC + '/*.tfrec'))
parsed_trainval = (tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)
                   .map(_parse_function, num_parallel_calls=AUTOTUNE).unbatch()
                   .filter(_filtTP).enumerate())

# Stratified 5-Fold


indices = []
spid = []
recid = []

for i, sample in tqdm(parsed_trainval.prefetch(AUTOTUNE)):
    indices.append(i.numpy())
    spid.append(sample['species_id'].numpy())
    recid.append(sample['recording_id'].numpy().decode())

table = pd.DataFrame({'indices': indices, 'species_id': spid, 'recording_id': recid})

skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
splits = list(skf.split(table.index, table.species_id))


# plt.hist([table.loc[splits[0][0], 'species_id'], table.loc[splits[0][1], 'species_id']], bins=CLASS_N, stacked=True)
# plt.show()


def create_idx_filter(indice):
    @tf.function
    def _filt(i, x):
        return tf.reduce_any(indice == i)

    return _filt


@tf.function
def _remove_idx(i, x):
    return x


def create_train_dataset(batchsize, train_idx):
    global parsed_trainval
    parsed_train = (parsed_trainval
                    .filter(create_idx_filter(train_idx))
                    .map(_remove_idx))

    dataset = (parsed_train.cache()
               .shuffle(len(train_idx))
               .repeat()
               .map(_cut_wav, num_parallel_calls=AUTOTUNE)
               .map(_wav_to_spec, num_parallel_calls=AUTOTUNE)
               .map(_create_annot_train, num_parallel_calls=AUTOTUNE)
               .map(_preprocess, num_parallel_calls=AUTOTUNE)
               .batch(batchsize))

    if cfg['model_params']['mixup']:
        dataset = (dataset.map(_mixup, num_parallel_calls=AUTOTUNE)
                   .prefetch(AUTOTUNE))
    else:
        dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def create_val_dataset(batchsize, val_idx):
    global parsed_trainval
    parsed_val = (parsed_trainval
                  .filter(create_idx_filter(val_idx))
                  .map(_remove_idx))

    vdataset = (parsed_val
                .map(_cut_wav_val, num_parallel_calls=AUTOTUNE)
                .map(_wav_to_spec, num_parallel_calls=AUTOTUNE)
                .map(_create_annot_val, num_parallel_calls=AUTOTUNE)
                .map(_preprocess_val, num_parallel_calls=AUTOTUNE)
                .batch(batchsize)
                .cache())
    return vdataset


@tf.function
def _one_sample_positive_class_precisions(example):
    y_true, y_pred = example

    retrieved_classes = tf.argsort(y_pred, direction='DESCENDING')
    class_rankings = tf.argsort(retrieved_classes)
    retrieved_class_true = tf.gather(y_true, retrieved_classes)
    retrieved_cumulative_hits = tf.math.cumsum(tf.cast(retrieved_class_true, tf.float32))

    idx = tf.where(y_true)[:, 0]
    i = tf.boolean_mask(class_rankings, y_true)
    r = tf.gather(retrieved_cumulative_hits, i)
    c = 1 + tf.cast(i, tf.float32)
    precisions = r / c

    dense = tf.scatter_nd(idx[:, None], precisions, [y_pred.shape[0]])
    return dense


class LWLRAP(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='lwlrap'):
        super().__init__(name=name)

        self._precisions = self.add_weight(
            name='per_class_cumulative_precision',
            shape=[num_classes],
            initializer='zeros',
        )

        self._counts = self.add_weight(
            name='per_class_cumulative_count',
            shape=[num_classes],
            initializer='zeros',
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        precisions = tf.map_fn(
            fn=_one_sample_positive_class_precisions,
            elems=(y_true, y_pred),
            dtype=(tf.float32),
        )

        increments = tf.cast(precisions > 0, tf.float32)
        total_increments = tf.reduce_sum(increments, axis=0)
        total_precisions = tf.reduce_sum(precisions, axis=0)

        self._precisions.assign_add(total_precisions)
        self._counts.assign_add(total_increments)

    def result(self):
        per_class_lwlrap = self._precisions / tf.maximum(self._counts, 1.0)
        per_class_weight = self._counts / tf.reduce_sum(self._counts)
        overall_lwlrap = tf.reduce_sum(per_class_lwlrap * per_class_weight)
        return overall_lwlrap

    def reset_states(self):
        self._precisions.assign(self._precisions * 0)
        self._counts.assign(self._counts * 0)


# Testset and Inference function


def _parse_function_test(example_proto):
    sample = tf.io.parse_single_example(example_proto, feature_description)
    wav, _ = tf.audio.decode_wav(sample['audio_wav'], desired_channels=1)  # mono

    @tf.function
    def _cut_audio(i):
        _sample = {
            'audio_wav': tf.reshape(wav[i * SR * TIME:(i + 1) * SR * TIME], [SR * TIME]),
            'recording_id': sample['recording_id']
        }
        return _sample

    return tf.map_fn(_cut_audio, tf.range(60 // TIME), dtype={
        'audio_wav': tf.float32,
        'recording_id': tf.string
    })


def inference(model):
    tdataset = (tf.data.TFRecordDataset(tf.io.gfile.glob(TEST_TFREC + '/*.tfrec'), num_parallel_reads=AUTOTUNE)
                .map(_parse_function_test, num_parallel_calls=AUTOTUNE).unbatch()
                .map(_wav_to_spec, num_parallel_calls=AUTOTUNE)
                .map(_preprocess_test, num_parallel_calls=AUTOTUNE)
                .batch(1 * (60 // TIME)).prefetch(AUTOTUNE))

    rec_ids = []
    probs = []
    for inp, rec_id in tqdm(tdataset):
        pred = model.predict_on_batch(tf.reshape(inp, [-1, HEIGHT, WIDTH, 3]))
        prob = tf.sigmoid(pred)
        prob = tf.reduce_max(tf.reshape(prob, [-1, 60 // TIME, CLASS_N]), axis=1)
        rec_id_stack = tf.reshape(rec_id, [-1, 60 // TIME])
        for rec in rec_id.numpy():
            assert len(np.unique(rec)) == 1
        rec_ids.append(rec_id_stack.numpy()[:, 0])
        probs.append(prob.numpy())

    crec_ids = np.concatenate(rec_ids)
    cprobs = np.concatenate(probs)

    sub = pd.DataFrame({
        'recording_id': list(map(lambda x: x.decode(), crec_ids.tolist())),
        **{f's{i}': cprobs[:, i] / 5 for i in range(CLASS_N)}
    })
    sub = sub.sort_values('recording_id')
    return sub


# Start training!
def plot_history(history, name):
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.title("loss")
    # plt.yscale('log')
    plt.subplot(1, 2, 2)
    plt.plot(history.history["lwlrap"])
    plt.plot(history.history["val_lwlrap"])
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.title("metric")
    plt.savefig(name)


def train_and_inference(splits, split_id):
    batchsize = cfg['model_params']['batchsize_per_tpu']
    print("batchsize", batchsize)
    loss_fn = cfg['model_params']['loss']['fn'](from_logits=True, **cfg['model_params']['loss']['params'])
    optimizer = cfg['model_params']['optim']['fn'](**cfg['model_params']['optim']['params'])
    model = create_model()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[LWLRAP(CLASS_N)])
    idx_train_tf = tf.cast(tf.constant(splits[split_id][0]), tf.int64)
    idx_val_tf = tf.cast(tf.constant(splits[split_id][1]), tf.int64)
    dataset = create_train_dataset(batchsize, idx_train_tf)
    vdataset = create_val_dataset(batchsize, idx_val_tf)
    history = model.fit(dataset,
                        batch_size=cfg['model_params']['batchsize_per_tpu'],
                        steps_per_epoch=cfg['model_params']['iteration_per_epoch'],
                        epochs=cfg['model_params']['epoch'],
                        validation_data=vdataset,
                        callbacks=[
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath='./model/model_best_%d.h5' % split_id,
                                save_weights_only=True,
                                monitor='val_lwlrap',
                                mode='max',
                                save_best_only=True),
                        ])
    plot_history(history, 'history_%d.png' % split_id)
    # inference
    model.load_weights('./model/model_best_%d.h5' % split_id)
    return inference(model)


# N-fold ensemble
sub = sum(
    map(
        lambda i: train_and_inference(splits, i).set_index('recording_id'),
        range(len(splits))
    )
).reset_index()
sub.describe()
sub.to_csv("submission.csv", index=False)
FileLink(r'submission.csv')
