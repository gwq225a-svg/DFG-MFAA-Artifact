"""Implementation of  multi-Feature attention attack."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from absl import flags
from tensorflow.keras.utils import to_categorical

import utils

FLAGS = flags.FLAGS

flags.DEFINE_string('model_name', 'resnet_v1_152', 'The Model used to generate adv.')
flags.DEFINE_string('attack_method', 'MFA', 'The name of attack method.')
flags.DEFINE_integer('layer_num', '3', 'The number of layer used to generate adv.')
flags.DEFINE_string('layer_name','resnet_v1_152/block2/unit_7/bottleneck_v1/Relu','The layer to be attacked.')
flags.DEFINE_string('GPU_ID', '1', 'which GPU to use.')
flags.DEFINE_string('input_dir', './dataset/images/', 'Input directory with images.')
flags.DEFINE_string('output_dir', './adv/MFAA-resnet_v1_152/','Output directory with images.')
flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')
flags.DEFINE_integer('num_iter', 10, 'Number of iterations.')
flags.DEFINE_float('alpha', 1.6, 'Step size.')
flags.DEFINE_integer('batch_size', 20, 'How many images process at one time.')
flags.DEFINE_float('momentum', 1.0, 'Momentum.')

"""parameter for DIM"""
flags.DEFINE_integer('image_size', 224, 'size of each input images.')
flags.DEFINE_integer('image_resize', 250, 'size of each diverse images.')
flags.DEFINE_float('prob', 0.7, 'Probability of using diverse inputs.')

"""parameter for TIM"""
flags.DEFINE_integer('Tkern_size', 15, 'Kernel size of TIM.')

"""parameter for PIM"""
flags.DEFINE_float('amplification_factor', 2.5, 'To amplifythe step size.')
flags.DEFINE_float('gamma', 0.5, 'The gamma parameter.')
flags.DEFINE_integer('Pkern_size', 3, 'Kernel size of PIM.')

"""parameter for MFA"""
flags.DEFINE_float('ens', 30.0, 'Number of random mask input.')
flags.DEFINE_float('probb', 0.8, 'keep probability = 1 - drop probability.')
flags.DEFINE_float('keep_low', 0.65, 'The low-frequency keep ratio for frequency mask.')
EPS = 1e-12

"""Dynamic Frequency Mask Parameters"""
flags.DEFINE_float('low_ratio_base', 0.40, 'Initial low frequency ratio.')
flags.DEFINE_float('low_ratio_max', 0.55, 'Final low frequency ratio.')
flags.DEFINE_float('sigma_ratio', 0.06, 'Soft mask smoothing parameter.')

flags.DEFINE_float('rfia_lambda', 0.15, 'Residual fusion weight for RFIA.')

"""the loss function for FDA"""
def get_fda_loss(opt_operations):
    loss = 0
    for layer in opt_operations:
        if layer is None:
            continue
        batch_size = FLAGS.batch_size
        tensor = layer[:batch_size]
        mean_tensor = tf.stack([tf.reduce_mean(tensor, -1), ] * tensor.shape[-1], -1)
        wts_good = tf.cast(tensor < mean_tensor,tf.float32)
        wts_bad = tf.cast(tensor >= mean_tensor,tf.float32)
        loss += tf.log(tf.nn.l2_loss(wts_good * (layer[batch_size:]) / tf.cast(tf.size(layer),tf.float32)))
        loss -= tf.log(tf.nn.l2_loss(wts_bad * (layer[batch_size:]) / tf.cast(tf.size(layer),tf.float32)))
    return loss / len(opt_operations)

"""the loss function for NRDM"""
def get_nrdm_loss(opt_operations):
    loss = 0
    for layer in opt_operations:
        if layer is None:
            continue
        ori_tensor = layer[:FLAGS.batch_size]
        adv_tensor = layer[FLAGS.batch_size:]
        loss+=tf.norm(ori_tensor-adv_tensor)/tf.cast(tf.size(layer),tf.float32)
    return loss / len(opt_operations)

"""the loss function for MFA"""
def get_fia_loss(opt_operations,weights):
    loss = 0
    for layer in opt_operations:
        if layer is None:
            continue
        adv_tensor = layer[FLAGS.batch_size:]
        loss += tf.reduce_sum(adv_tensor*weights) / tf.cast(tf.size(layer), tf.float32)
    return loss / len(opt_operations)

# ── Normalization Utilities ─────────────────────────────
def tfnormalize(grad, opt=2):
    if opt==0:
        return grad
    elif opt==1:
        return grad / (tf.reduce_sum(tf.abs(grad), axis=(1,2,3), keepdims=True)+EPS)
    elif opt==2:
        return grad / (tf.sqrt(tf.reduce_sum(tf.square(grad), axis=(1,2,3), keepdims=True))+EPS)

def normalize(grad,opt=2):
    if opt==0:
        return grad
    elif opt==1:
        return grad / (np.sum(np.abs(grad), axis=(1,2,3), keepdims=True)+EPS)
    elif opt==2:
        return grad / (np.sqrt(np.sum(np.square(grad), axis=(1,2,3), keepdims=True))+EPS)

# ── Kernel Projection ───────────────────────────────────
def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size**2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0

    # 构造 depthwise conv kernel: [k, k, 3, 1]
    kern = kern[:, :, None, None]          # [k, k, 1, 1]
    stack_kern = np.tile(kern, (1, 1, 3, 1))  # [k, k, 3, 1]

    return stack_kern.astype(np.float32), kern_size // 2

def project_noise(x, stack_kern, kern_size):
    x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = tf.nn.depthwise_conv2d(x, stack_kern, strides=[1,1,1,1], padding='VALID')
    return x

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    return stack_kernel


def input_diversity(input_tensor):
    rnd = tf.compat.v1.random_uniform((), FLAGS.image_size, FLAGS.image_resize, dtype=tf.int32)
    rescaled = tf.image.resize(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = FLAGS.image_resize - rnd
    w_rem = FLAGS.image_resize - rnd
    pad_top = tf.compat.v1.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.compat.v1.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
    ret=tf.cond(tf.compat.v1.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
    ret = tf.image.resize(ret, [FLAGS.image_size, FLAGS.image_size],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return ret

def _is_4d_tensor(tensor):
    return tensor.shape.ndims == 4

def _find_tensor_by_candidates(operations, candidates):
    # 先精确匹配
    for name in candidates:
        for op in operations:
            if op.name == name:
                for out in op.outputs:
                    if _is_4d_tensor(out):
                        print("Found exact layer:", op.name, out)
                        return out

    # 再做包含匹配
    for name in candidates:
        for op in operations:
            if name in op.name:
                for out in op.outputs:
                    if _is_4d_tensor(out):
                        print("Found fuzzy layer:", op.name, out)
                        return out
    return None

def get_opt_layers(model_name):
    operations = tf.compat.v1.get_default_graph().get_operations()

    if model_name == 'inception_v3':
        layer_groups = [
            ['InceptionV3/InceptionV3/Mixed_7a/concat',
             'InceptionV3/Mixed_7a/concat',
             'Mixed_7a/concat'],
            ['InceptionV3/InceptionV3/Mixed_6a/concat',
             'InceptionV3/Mixed_6a/concat',
             'Mixed_6a/concat'],
            ['InceptionV3/InceptionV3/Mixed_5b/concat',
             'InceptionV3/Mixed_5b/concat',
             'Mixed_5b/concat'],
            ['InceptionV3/InceptionV3/Conv2d_4a_3x3/Relu',
             'InceptionV3/Conv2d_4a_3x3/Relu',
             'Conv2d_4a_3x3/Relu'],
            ['InceptionV3/InceptionV3/Conv2d_2b_3x3/Relu',
             'InceptionV3/Conv2d_2b_3x3/Relu',
             'Conv2d_2b_3x3/Relu'],
            ['InceptionV3/InceptionV3/Conv2d_1a_3x3/Relu',
             'InceptionV3/Conv2d_1a_3x3/Relu',
             'Conv2d_1a_3x3/Relu'],
        ]
    elif model_name == 'vgg_19':
        layer_groups = [
            ['vgg_19/conv5/conv5_4/Relu', 'vgg_19/conv5_4/Relu'],
            ['vgg_19/conv5/conv5_3/Relu', 'vgg_19/conv5_3/Relu'],
            ['vgg_19/conv4/conv4_4/Relu', 'vgg_19/conv4_4/Relu'],
            ['vgg_19/conv4/conv4_3/Relu', 'vgg_19/conv4_3/Relu'],
            ['vgg_19/conv3/conv3_4/Relu', 'vgg_19/conv3_4/Relu'],
            ['vgg_19/conv2/conv2_2/Relu', 'vgg_19/conv2_2/Relu'],
        ]
    elif model_name == 'vgg_16':
        layer_groups = [
            ['vgg_16/conv5/conv5_3/Relu', 'vgg_16/conv5_3/Relu'],
            ['vgg_16/conv4/conv4_3/Relu', 'vgg_16/conv4_3/Relu'],
            ['vgg_16/conv3/conv3_3/Relu', 'vgg_16/conv3_3/Relu'],
            ['vgg_16/conv2/conv2_2/Relu', 'vgg_16/conv2_2/Relu'],
            ['vgg_16/conv1/conv1_2/Relu', 'vgg_16/conv1_2/Relu'],
            ['vgg_16/conv1/conv1_1/Relu', 'vgg_16/conv1_1/Relu'],
        ]
    elif model_name == 'resnet_v1_152':
        layer_groups = [
            ['resnet_v1_152/block3/unit_29/bottleneck_v1/Relu'],
            ['resnet_v1_152/block3/unit_19/bottleneck_v1/Relu'],
            ['resnet_v1_152/block3/unit_9/bottleneck_v1/Relu'],
            ['resnet_v1_152/block2/unit_7/bottleneck_v1/Relu'],
            ['resnet_v1_152/block1/unit_3/bottleneck_v1/Relu'],
            ['resnet_v1_152/conv1/Relu'],
        ]
    else:
        raise ValueError("Unsupported model_name: {}".format(model_name))

    found = []
    shapes = []

    for candidates in layer_groups:
        tensor = _find_tensor_by_candidates(operations, candidates)
        if tensor is None:
            raise ValueError("Cannot find layer from candidates: {}".format(candidates))
        found.append([tensor])
        shapes.append(tensor[:FLAGS.batch_size].shape)

    return (
        found[0], found[1], found[2], found[3], found[4], found[5],
        shapes[0], shapes[1], shapes[2], shapes[3], shapes[4], shapes[5]
    )

# """obtain the feature map of the target layer"""
# def get_opt_layers(layer_name):
#     opt_operations = []
#     opt_operations1 = []
#     opt_operations2 = []
#     opt_operations3 = []
#     opt_operations4 = []
#     opt_operations5 = []
#     #shape=[FLAGS.batch_size,FLAGS.image_size,FLAGS.image_size,3]
#     operations = tf.compat.v1.get_default_graph().get_operations()
#     for op in operations:
#         if 'resnet_v1_152/block4/unit_3/bottleneck_v1/Relu' == op.name:
#             print(op.name, op.outputs)
#             opt_operations.append(op.outputs[0])
#             shape = op.outputs[0][:FLAGS.batch_size].shape
#         elif 'resnet_v1_152/block3/unit_29/bottleneck_v1/Relu' == op.name:
#             print(op.name, op.outputs)
#             opt_operations1.append(op.outputs[0])
#             shape1 = op.outputs[0][:FLAGS.batch_size].shape
#         elif 'resnet_v1_152/block3/unit_19/bottleneck_v1/Relu' == op.name:
#             print(op.name, op.outputs)
#             opt_operations2.append(op.outputs[0])
#             shape2 = op.outputs[0][:FLAGS.batch_size].shape
#         elif 'resnet_v1_152/block3/unit_9/bottleneck_v1/Relu' == op.name:
#             print(op.name, op.outputs)
#             opt_operations3.append(op.outputs[0])
#             shape3 = op.outputs[0][:FLAGS.batch_size].shape
#         elif 'resnet_v1_152/block2/unit_7/bottleneck_v1/Relu' == op.name:
#             print(op.name, op.outputs)
#             opt_operations4.append(op.outputs[0])
#             shape4 = op.outputs[0][:FLAGS.batch_size].shape
#         elif 'resnet_v1_152/block1/unit_3/bottleneck_v1/Relu' == op.name:
#             print(op.name, op.outputs)
#             opt_operations5.append(op.outputs[0])
#             shape5 = op.outputs[0][:FLAGS.batch_size].shape
#
#     return opt_operations, opt_operations1, opt_operations2, opt_operations3, opt_operations4, opt_operations5, shape, shape1, shape2, shape3, shape4, shape5

def apply_frequency_mask_dynamic_noise(
        noise_np,          # (N,H,W,C)
        t, T,
        low_ratio_base=0.45,
        low_ratio_max=0.65,
        sigma_ratio=0.06,
        energy_thresh=1.2,
        momentum=0.8,
        prev_state=None,
        debug=False
):


    N, H, W, C = noise_np.shape
    minHW = min(H, W)
    cy = (H - 1) / 2
    cx = (W - 1) / 2
    F = np.fft.fftshift(
        np.fft.fft2(noise_np, axes=(1, 2)),
        axes=(1, 2)
    )
    power = np.mean(np.abs(F) ** 2, axis=(0, 3))  # (H,W)

    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if prev_state is None:
        low_ratio = low_ratio_base
    else:
        low_ratio = prev_state["low_ratio"]

    r_low = low_ratio * minHW
    r_mid = 0.45 * minHW

    low_mask = dist <= r_low
    mid_mask = (dist > r_low) & (dist <= r_mid)

    E_low = power[low_mask].mean() if np.any(low_mask) else 0.0
    mid_band = power[(dist > r_low) & (dist <= r_mid)]
    if mid_band.size < 10:
        E_mid = np.percentile(power, 60)  # fallback
    else:
        E_mid = mid_band.mean()

    energy_ratio = np.clip(E_low / (E_mid + 1e-12), 0.0, 10.0)

    if energy_ratio > energy_thresh:
        expand = min(
            (energy_ratio - energy_thresh) / energy_thresh,
            1.0
        )
    else:
        expand = 0.0

    target_low_ratio = (
        low_ratio_base
        + expand * (low_ratio_max - low_ratio_base)
    )

    if prev_state is None:
        low_ratio = low_ratio_base
    else:
        low_ratio_prev = prev_state["low_ratio"]
        low_ratio = (
            momentum * low_ratio_prev
            + (1.0 - momentum) * target_low_ratio
        )

    low_ratio = np.clip(low_ratio, low_ratio_base, low_ratio_max)
    r_t = low_ratio * minHW

    sigma = max(1.0, sigma_ratio * minHW)
    final_mask = 1.0 / (1.0 + np.exp((dist - r_t) / sigma))
    final_mask = final_mask.astype(np.float32)

    F *= final_mask[None, :, :, None]
    noise_filtered = np.fft.ifft2(
        np.fft.ifftshift(F, axes=(1, 2)),
        axes=(1, 2)
    ).real

    state = {
        "low_ratio": float(low_ratio),
        "energy_ratio": float(energy_ratio),
        "mask_mean": float(final_mask.mean())
    }

    if debug:
        print(
            f"[Noise-Freq] t={t}/{T} | "
            f"E_low/E_mid={energy_ratio:.3f} | "
            f"low_ratio={low_ratio:.3f} | "
            f"mask_mean={final_mask.mean():.3f}"
        )

    return noise_filtered, state

def run_mfaa_weights_once(
    sess,
    images_tmp2,
    labels,
    ori_input,
    adv_input,
    label_ph,
    weights_tensor,
    weights_tensor1,
    weights_tensor2,
    weights_tensor3,
    weights_tensor4,
    weights_tensor5
):
    feed = {
        ori_input: images_tmp2,
        adv_input: images_tmp2,
        label_ph: labels
    }

    w, w1, w2, w3, w4, w5 = sess.run(
        [weights_tensor,
         weights_tensor1,
         weights_tensor2,
         weights_tensor3,
         weights_tensor4,
         weights_tensor5],
        feed_dict=feed
    )
    return w, w1, w2, w3, w4, w5

def main(_):
    # ────────────── 环境与参数初始化 ──────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU_ID
    P_kern, kern_size = project_kern(FLAGS.Pkern_size)
    T_kern = gkern(FLAGS.Tkern_size)

    # ε 和 α 设置
    if FLAGS.model_name in ['vgg_16','vgg_19', 'resnet_v1_50','resnet_v1_152']:
        eps = FLAGS.max_epsilon
        alpha = FLAGS.alpha
    else:
        eps = 2.0 * FLAGS.max_epsilon / 255.0
        alpha = FLAGS.alpha * 2.0 / 255.0

    num_iter = FLAGS.num_iter
    momentum = FLAGS.momentum

    # 图像预处理函数
    image_preprocessing_fn = utils.normalization_fn_map[FLAGS.model_name]
    inv_image_preprocessing_fn = utils.inv_normalization_fn_map[FLAGS.model_name]
    batch_shape = [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3]
    checkpoint_path = utils.checkpoint_paths[FLAGS.model_name]
    layer_name = FLAGS.layer_name

    # ────────────── 构建图 ──────────────
    with tf.Graph().as_default():
        # 输入占位符
        ori_input = tf.compat.v1.placeholder(tf.float32, shape=batch_shape)
        adv_input = tf.compat.v1.placeholder(tf.float32, shape=batch_shape)
        num_classes = 1000 + utils.offset[FLAGS.model_name]
        label_ph = tf.compat.v1.placeholder(tf.float32, shape=[FLAGS.batch_size*2, num_classes])
        accumulated_grad_ph = tf.compat.v1.placeholder(tf.float32, shape=batch_shape)
        amplification_ph = tf.compat.v1.placeholder(tf.float32, shape=batch_shape)

        # 网络
        network_fn = utils.nets_factory.get_network_fn(FLAGS.model_name, num_classes=num_classes, is_training=False)
        x = tf.concat([ori_input, adv_input], axis=0)

        # 使用 DIM 或不使用
        if 'DI' in FLAGS.attack_method:
            logits, end_points = network_fn(input_diversity(x))
        else:
            logits, end_points = network_fn(x)

            # 基本分类信息
        # problity = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(logits, axis=1)
        one_hot = tf.one_hot(pred, num_classes)
        entropy_loss = tf.compat.v1.losses.softmax_cross_entropy(one_hot[:FLAGS.batch_size], logits[FLAGS.batch_size:])

        (opt_operations, opt_operations1, opt_operations2,
         opt_operations3, opt_operations4, opt_operations5,
         shape, shape1, shape2, shape3, shape4, shape5) =  get_opt_layers(FLAGS.model_name)

        weights_ph = tf.compat.v1.placeholder(tf.float32, shape=shape)
        weights_ph1 = tf.compat.v1.placeholder(tf.float32, shape=shape1)
        weights_ph2 = tf.compat.v1.placeholder(tf.float32, shape=shape2)
        weights_ph3 = tf.compat.v1.placeholder(tf.float32, shape=shape3)
        weights_ph4 = tf.compat.v1.placeholder(tf.float32, shape=shape4)
        weights_ph5 = tf.compat.v1.placeholder(tf.float32, shape=shape5)

        # ────────────── 选择损失函数 ──────────────
        if 'FDA' in FLAGS.attack_method:
            loss = get_fda_loss(opt_operations)
        elif 'NRDM' in FLAGS.attack_method:
            loss = get_nrdm_loss(opt_operations)
        elif 'MFA' in FLAGS.attack_method:
            # 计算 6 层梯度
            weights_tensor = tf.gradients(logits * label_ph, opt_operations[0])[0]
            weights_tensor1 = tf.gradients(logits * label_ph, opt_operations1[0])[0]
            weights_tensor2 = tf.gradients(logits * label_ph, opt_operations2[0])[0]
            weights_tensor3 = tf.gradients(logits * label_ph, opt_operations3[0])[0]
            weights_tensor4 = tf.gradients(logits * label_ph, opt_operations4[0])[0]
            weights_tensor5 = tf.gradients(logits * label_ph, opt_operations5[0])[0]

            quanzhi = 1  # 系数

            # 逐层 FIA 损失传递
            loss2 = get_fia_loss(opt_operations, weights_ph)
            w1_fromnext = tf.gradients(loss2, opt_operations1[0])[0]
            loss1 = get_fia_loss(opt_operations1, quanzhi * tfnormalize(w1_fromnext[FLAGS.batch_size:]) + weights_ph1)
            w2_fromnext = tf.gradients(loss1, opt_operations2[0])[0]
            loss = get_fia_loss(opt_operations2, quanzhi * tfnormalize(w2_fromnext[FLAGS.batch_size:]) + weights_ph2)
            w3_fromnext = tf.gradients(loss, opt_operations3[0])[0]
            lossx1 = get_fia_loss(opt_operations3, quanzhi * tfnormalize(w3_fromnext[FLAGS.batch_size:]) + weights_ph3)
            w4_fromnext = tf.gradients(lossx1, opt_operations4[0])[0]
            lossx2 = get_fia_loss(opt_operations4, quanzhi * tfnormalize(w4_fromnext[FLAGS.batch_size:]) + weights_ph4)
            w5_fromnext = tf.gradients(lossx2, opt_operations5[0])[0]
            lossx3 = get_fia_loss(opt_operations5, quanzhi * tfnormalize(w5_fromnext[FLAGS.batch_size:]) + weights_ph5)
            if FLAGS.model_name == 'inception_v3':
                loss = lossx1
            elif FLAGS.model_name == 'resnet_v1_152':
                loss = lossx1
            elif FLAGS.model_name == 'vgg_16':
                loss = loss
            elif FLAGS.model_name == 'vgg_19':
                loss = lossx2
            else:
                loss = lossx2
        else:
            loss = entropy_loss

        # ────────────── 计算梯度 ──────────────
        gradient = tf.gradients(loss, adv_input)[0]
        noise = gradient
        adv_input_update = adv_input
        amplification_update = amplification_ph

        # TIM 卷积
        if 'TI' in FLAGS.attack_method:
            noise = tf.nn.depthwise_conv2d(noise, T_kern, strides=[1, 1, 1, 1], padding='SAME')

        # momentum 更新
        noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keepdims=True)
        noise = momentum * accumulated_grad_ph + noise

        # PIM/PI 优化
        if 'PI' in FLAGS.attack_method:
            alpha_beta = alpha * FLAGS.amplification_factor
            gamma = FLAGS.gamma * alpha_beta
            amplification_update += alpha_beta * tf.sign(noise)
            cut_noise = tf.clip_by_value(abs(amplification_update) - eps, 0.0, 10000.0) * tf.sign(amplification_update)
            projection = gamma * tf.sign(project_noise(cut_noise, P_kern, kern_size))
            amplification_update += projection
            adv_input_update = adv_input_update + alpha_beta * tf.sign(noise) + projection
        else:
            adv_input_update = adv_input_update + alpha * tf.sign(noise)

        # ────────────── Session 与迭代生成 ──────────────
        saver = tf.compat.v1.train.Saver()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        with tf.compat.v1.Session(config=config) as sess:
            saver.restore(sess, checkpoint_path)
            count = 0

            for images, names, labels in utils.load_image(FLAGS.input_dir, FLAGS.image_size, FLAGS.batch_size):
                prev_layer_freq_in = [None] * 6
                mu = 0.4
                count += FLAGS.batch_size
                if count % 100 == 0:
                    print("Generating:", count)

                images_tmp = image_preprocessing_fn(np.copy(images))
                if FLAGS.model_name in ['resnet_v1_50','resnet_v1_152','vgg_16','vgg_19']:
                    labels -= 1

                # 构造标签
                labels = to_categorical(np.concatenate([labels, labels], axis=-1), num_classes)

                # 初始化 adversarial image
                if 'NRDM' in FLAGS.attack_method:
                    images_adv = images + np.random.normal(0, 0.1, size=np.shape(images))
                else:
                    images_adv = images
                images_adv = image_preprocessing_fn(np.copy(images_adv))

                # 初始化梯度与权重
                grad_np = np.zeros(shape=batch_shape)
                amplification_np = np.zeros(shape=batch_shape)
                weight_np = np.zeros(shape=shape)
                weight_np1 = np.zeros(shape=shape1)
                weight_np2 = np.zeros(shape=shape2)
                weight_np3 = np.zeros(shape=shape3)
                weight_np4 = np.zeros(shape=shape4)
                weight_np5 = np.zeros(shape=shape5)

                freq_state = None
                # ─────── 迭代优化 ───────
                for i in range(num_iter):
                    # MFA 权重计算（仅第 0 轮初始化）
                    if i == 0 and 'MFA' in FLAGS.attack_method:
                        # 原图权重
                        if FLAGS.ens == 0:
                            images_tmp2 = image_preprocessing_fn(np.copy(images))
                            w, w1, w2, w3, w4, w5 = run_mfaa_weights_once(
                                sess, images_tmp2, labels,
                                ori_input, adv_input, label_ph,
                                weights_tensor, weights_tensor1, weights_tensor2,
                                weights_tensor3, weights_tensor4, weights_tensor5
                            )
                            weight_np, weight_np1, weight_np2 = w[:FLAGS.batch_size], w1[:FLAGS.batch_size], w2[
                                                                                                             :FLAGS.batch_size]
                            weight_np3, weight_np4, weight_np5 = w3[:FLAGS.batch_size], w4[:FLAGS.batch_size], w5[
                                                                                                               :FLAGS.batch_size]
                        # ensemble masked 权重
                        for l in range(int(FLAGS.ens)):
                            mask = np.random.binomial(1, FLAGS.probb,
                                                      size=(batch_shape[0], batch_shape[1], batch_shape[2],
                                                            batch_shape[3]))
                            images_tmp2 = images * mask
                            images_tmp2 = image_preprocessing_fn(np.copy(images_tmp2))
                            w, w1, w2, w3, w4, w5 = run_mfaa_weights_once(
                                sess, images_tmp2, labels,
                                ori_input, adv_input, label_ph,
                                weights_tensor, weights_tensor1, weights_tensor2,
                                weights_tensor3, weights_tensor4, weights_tensor5
                            )
                            weight_np += w[FLAGS.batch_size:]
                            weight_np1 += w1[FLAGS.batch_size:]
                            weight_np2 += w2[FLAGS.batch_size:]
                            weight_np3 += w3[FLAGS.batch_size:]
                            weight_np4 += w4[FLAGS.batch_size:]
                            weight_np5 += w5[FLAGS.batch_size:]

                        # 动态频域初始化
                        # 6 分支动态频域
                        def normalize_np(x):
                            x = x.astype(np.float32)
                            norms = np.sqrt(np.sum(x * x, axis=(1, 2, 3), keepdims=True)) + 1e-12
                            out = x / norms
                            return out

                        for idx, w in enumerate(
                                [weight_np, weight_np1, weight_np2, weight_np3, weight_np4, weight_np5]):
                            if w is not None and w.size > 0:
                                ## print(f"\n--- 处理第{idx}层 ---"
                                sensitivity = np.exp(-0.15 * idx)
                                Rmin_ratio = 0.18 + 0.40 * (1 - sensitivity)
                                beta_val = 0.15 * sensitivity + 0.02

                                ## print(f"层级参数: sensitivity={sensitivity:.3f}, "
                                ##      f"Rmin_ratio={Rmin_ratio:.3f}, beta={beta_val:.3f}")

                                freq_in = -normalize_np(w)
                                # --------- 动态调度核心: g / g_prev 使用频域 ---------
                                g = freq_in
                                if prev_layer_freq_in[idx] is None:
                                    g_prev = None
                                else:
                                    g_prev = prev_layer_freq_in[idx]

                                w_new = freq_in
                                if idx == 0: weight_np = w_new
                                if idx == 1: weight_np1 = w_new
                                if idx == 2: weight_np2 = w_new
                                if idx == 3: weight_np3 = w_new
                                if idx == 4: weight_np4 = w_new
                                if idx == 5: weight_np5 = w_new
                                prev_layer_freq_in[idx] = (1.0 - mu) * prev_layer_freq_in[idx] + mu * g if \
                                prev_layer_freq_in[idx] is not None else g.copy()
                        ## prev_layer_freq_in[idx] = freq_in.copy()

                    # 优化更新
                    images_adv, grad_np, amplification_np = sess.run(
                        [adv_input_update, noise, amplification_update],
                        feed_dict={ori_input: images_tmp, adv_input: images_adv,
                                   weights_ph: weight_np, weights_ph1: weight_np1, weights_ph2: weight_np2,
                                   weights_ph3: weight_np3, weights_ph4: weight_np4, weights_ph5: weight_np5,
                                   label_ph: labels, accumulated_grad_ph: grad_np, amplification_ph: amplification_np})
                    noise_np = grad_np
                    if i < 3:
                        # 前 3 轮：纯低频
                        grad_np, _ = apply_frequency_mask_dynamic_noise(
                            grad_np,
                            t=i,
                            T=num_iter,
                            low_ratio_base=0.45,
                            low_ratio_max=0.45,
                            energy_thresh=10.0,
                            debug=False
                        )
                    else:
                        # 后面才自适应
                        raw_grad = grad_np.copy()
                        grad_np, freq_state = apply_frequency_mask_dynamic_noise(
                            raw_grad,
                            t=i,
                            T=num_iter,
                            low_ratio_base=0.3,
                            low_ratio_max=0.75,
                            energy_thresh=1.2,
                            momentum=0.8,
                            prev_state=freq_state,
                            debug=(i > 3)
                        )
                    # grad_np, _ = apply_frequency_mask_dynamic_noise(
                    #     grad_np,
                    #     t=i,
                    #     T=num_iter,
                    #     low_ratio_base=FLAGS.low_ratio_base,
                    #     low_ratio_max=FLAGS.low_ratio_base,  # 关键：和 base 保持一致
                    #     energy_thresh=10.0,  # 给大一点，基本不触发扩张
                    #     momentum=0.0,  # 纯固定时这个其实无所谓
                    #     prev_state=None,  # 不继承历史状态
                    #     debug=False
                    # )

                    images_adv = np.clip(images_adv, images_tmp - eps, images_tmp + eps)

                    # 保存生成对抗样本
                images_adv = inv_image_preprocessing_fn(images_adv)
                utils.save_image(images_adv, names, FLAGS.output_dir)

if __name__ == '__main__':
    from absl import app

    app.run(main)