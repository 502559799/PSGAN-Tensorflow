'''
@Author: Dongbao, Yan
@Date: 2020-03-09 16:46:45
@LastEditTime: 2020-03-31 21:50:10
@LastEditors: Do not edit
@Description: Do not edit
'''
import tensorflow as tf
import numpy as np
from scipy.misc import imsave
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import pickle
import dlib
import cv2
import os
import time
import random

from utils import *
from layers import *
import vgg16

to_train = False
to_test = True
to_restore = False
save_training_imgs = False

check_dir = "./output/checkpoints/"

class PSGAN(object):
    def __init__(self, sess, args):
        self.model_name = 'PSGAN'
        self.sess = sess
        self.checkpointer_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.dataset_path = os.path.join('./dataset/', self.dataset_name)
        print("dataset name : ", self.dataset_path)
        #self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.use_segs_flag = args.use_segs_flag
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.pool_size = args.pool_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.init_lr = args.lr
        self.ch = args.ch
        #self.selected_attrs = args.selected_attrs
        #self.custom_label = np.expand_dims(args.custom_label, axis=0)
        #self.c_dim = len(self.selected_attrs)

        """ Weight """
        #self.adv_weight = args.adv_weight
        #self.rec_weight = args.rec_weight
        #self.cls_weight = args.cls_weight
        self.adv_ld = args.adv_ld
        self.per_ld = args.per_ld
        self.cyc_ld = args.cyc_ld
        self.make_ld = args.make_ld

        self.vfs_weight = args.vfs_weight

        """Generator"""
        self.mdn_n_res = args.mdn_n_res
        self.mdn_ch = args.mdn_ch

        """Discriminator"""
        self.man_n_res = args.man_n_res
        self.man_ch = args.man_ch

        self.img_height = args.img_height
        self.img_width = args.img_width
        self.img_ch = args.img_ch

        print()

        print("##### Information #####")
        print("gan type: ", self.gan_type)
        #print("# selected_attrs : ", self.selected_attrs)
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        #print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.mdn_n_res)
        print("# channels : ", self.man_ch)

        print("##### Discriminator #####")
        print("# residual blocks : ", self.man_n_res)
        print("# channels : ", self.mdn_ch)




    ##################################################################################
    # MDNet
    ##################################################################################

    def mdnet_module(self, x_init, reuse=False, scope="mdnet_module"):
        channel = self.mdn_ch
        #c = tf.cast(tf.reshape(c, shape=[-1, 1, 1, c.shape[-1]]), tf.float32)
        #c = tf.tile(c, [1, x_init.shape[1], x_init.shape[2], 1])
        #x = tf.concat([x_init, c], axis=-1)
        with tf.variable_scope(scope, reuse=reuse):
            x = tf.pad(x_init, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            #x = tf.pad(x_init, [[0, 0], [3, 3], [3, 3], [0, 0]])
            res = general_conv2d(x, channel, kernel=7, stride=1, padding="VALID",
                                 name="MDNet_conv", do_norm=True, do_relu=True)  # x: 1*256*256*64

            # Down-Sampling
            for i in range(2):
                res = tf.pad(res, [[0,0], [1, 1], [1, 1], [0, 0]], "REFLECT")
                #x = tf.pad(x_init, [[0, 0], [3, 3], [3, 3], [0, 0]])
                res = general_conv2d(res, channel*2, kernel=4, stride=2, padding="VALID", name="MDNet_conv_"+str(i), do_norm=True, do_relu=True)

                channel = channel * 2   # x: 1*64*64*256

            # Bottleneck
            for i in range(1, self.mdn_n_res + 1):
                res = resblock(res, channel, name='MDNet_resblock_' + str(i))  #  res: 1*64*64*256
                print( "mdn res : ", res)
            return res

    ##################################################################################
    # AMM Module
    ##################################################################################

    def amm_module(self, mdn_fm, man_fm, x_rs, y_rs, fm_height, fm_width, fm_channel,
                   ldmk_x, ldmk_y, mask, reuse=False, scope="amm_module"):
        # makeup matrices gamma and beta
        ## dim of mat_gamma and mat_beta equal to 1 x H x W
        gamma_t = general_conv2d(mdn_fm, 1, kernel=1, stride=1, padding="VALID", name="mat_gamma", do_norm=False, do_relu=False) # 1 * 64 * 64 * 1
        beta_t = general_conv2d(mdn_fm, 1, kernel=1, stride=1, padding="VALID", name = "mat_beta", do_norm=False, do_relu=False) # 1 * 64 * 64 * 1
        print("beta_t : ", beta_t)

        # Here, mask is a 0-1 array

        with tf.variable_scope(scope, reuse=reuse):
            # reshape  from C x H x W to C x HW
            vs_s = tf.reshape(man_fm, (fm_channel, fm_height * fm_width))  # C x HW
            vs_r = tf.reshape(mdn_fm, (fm_channel, fm_height * fm_width))  # C x HW

            # weight of visual features
            vfs_weight = self.vfs_weight

            # construct relative position features mat
            # x_rs : 1 * 256 * 256 * 3
            # y_rs : 1 * 256 * 256 * 3
            imageX_H = x_rs.get_shape().as_list()[1] # 255
            imageX_W = x_rs.get_shape().as_list()[2]
            imageX_C = x_rs.get_shape().as_list()[3]
            imageY_H = y_rs.get_shape().as_list()[1]
            imageY_W = y_rs.get_shape().as_list()[2]
            imageY_C = y_rs.get_shape().as_list()[3]

            print("imageX_H : ", imageX_H)
            print("imageX_W : ", imageX_W)
            print("landmark_x : ", ldmk_x)  # 68 * 2
            print("landmark_y : ", ldmk_y)

            # p_i = tf.constant(0, shape=[imageX_H, imageX_W, 136]) #256*256*136
            # p_j = tf.constant(0, shape=[imageY_H, imageX_W, 136])

            # use meshgrid to construct grid_x gridy
            grid_y, grid_x = tf.meshgrid(tf.range(imageX_H), tf.range(imageX_W))
            grid_x = tf.tile(tf.reshape(grid_x, (1, 1, imageX_H, imageX_W)), [1, 68, 1, 1])  # 1 * 68 * 256 * 256
            grid_y = tf.tile(tf.reshape(grid_y, (1, 1, imageX_H, imageX_W)), [1, 68, 1, 1])

            p_i_x = grid_x - tf.reshape(ldmk_x[:, 0], (1, 68, 1, 1))
            p_i_y = grid_y - tf.reshape(ldmk_x[:, 1], (1, 68, 1, 1))  # 1 * 68 * 256 * 256
            p_i = tf.concat([p_i_x, p_i_y], axis=1)  # 1 * 136 * 256 * 256

            p_j_x = grid_x - tf.reshape(ldmk_y[:, 0], (1, 68, 1, 1))
            p_j_y = grid_y - tf.reshape(ldmk_y[:, 1], (1, 68, 1, 1))  # 1 * 68 * 256 * 256
            p_j = tf.concat([p_j_x, p_j_y], axis=1)  # 1 * 136 * 256 * 256

            p_i = tf.reshape(tf.image.resize_images(tf.transpose(p_i, [0, 2, 3, 1]), [fm_height, fm_width]),
                                (p_i.get_shape().as_list()[1], fm_height * fm_width))  # [1 136 256 256] -> [1 64 64 136] -> [136 64*64]
            p_j = tf.reshape(tf.image.resize_images(tf.transpose(p_j, [0, 2, 3, 1]), [fm_height, fm_width]),
                                (p_j.get_shape().as_list()[1], fm_height * fm_width))  # [1 136 256 256] -> [1 64 64 136] -> [136 64*64]


            # p_i = tf.Variable(tf.constant(0, shape=[imageX_H, imageX_W, 136]), name="p_i")
            # p_j = tf.Variable(tf.constant(0, shape=[imageX_H, imageX_W, 136]), name="p_j")

            # for i in range(imageX_H):  # 256
            #     for j in range(imageX_W):  # 256
            #         for k in range(136):
            #             if k < 68:
            #                 p_i[i, j, k].assign(tf.subtract(tf.constant(i), ldmk_x[k, 0]))
            #                 p_j[i, j, k].assign(tf.subtract(tf.constant(i), ldmk_y[k, 0]))
            #                 #print("ydb*******")
            #             else:
            #                 p_i[i, j, k].assign(tf.subtract(tf.constant(j), ldmk_x[k - 68, 1]))
            #                 p_j[i, j, k].assign(tf.subtract(tf.constant(j), ldmk_y[k - 68, 1]))

            ## resize p to same size of feature map
            # p_i = tf.reshape(tf.transpose(tf.image.resize_images(p_i, [fm_height, fm_width]), [2, 0, 1]), (136, fm_height * fm_width)) # [256, 256, 136] --> [64, 64, 136] --> [136, 64, 64] --> [136, 64*64]
            # p_j = tf.reshape(tf.transpose(tf.image.resize_images(p_j, [fm_height, fm_width]), [2, 0, 1]), (136, fm_height * fm_width))
            print("p_i", p_i)
            print("p_j", p_j)
            # normalization
            p_i_nm = tf.div(p_i, tf.norm(p_i, ord=2, axis=None, keep_dims=False))
            p_j_nm = tf.div(p_j, tf.norm(p_j, ord=2, axis=None, keep_dims=False))

            # concat
            concat_i = tf.concat([vfs_weight * vs_s, p_i_nm], 0)
            concat_j = tf.concat([vfs_weight * vs_r, p_j_nm], 0)

            # exp(matmul results)
            matrix_temp1 = tf.exp(tf.matmul(tf.transpose(concat_i, [1,0]), concat_j)) # HW * HW (64*64 * 64*64)

            # introduce visual similarities consideration
            ## sum results of lip + eye + face
            matrix_temp2 = tf.multiply(matrix_temp1, mask[0]) + tf.multiply(matrix_temp1, mask[1]) + tf.multiply(matrix_temp1, mask[2]) # HW * HW

            # final result of attentive makeup matrix A
            ## use  A/A:,j to get norm along each row
            matrix_A = matrix_temp2 / tf.reduce_sum(matrix_temp2, 1, keep_dims=True)  # Hw * HW
            print("matrix_A : ", matrix_A)

            # apply softmax on matrix_A
            matrix_A = tf.nn.softmax(matrix_A)

            # mt_gamma mt_beta
            mm_gamma = tf.reshape(tf.matmul(matrix_A, tf.reshape(gamma_t, (fm_height * fm_width, 1))), (1, fm_height, fm_width)) # (HW * HW) * (HW * 1) = (HW * 1) ---> 1 x H x W
            mm_beta = tf.reshape(tf.matmul(matrix_A, tf.reshape(beta_t, (fm_height * fm_width, 1))), (1, fm_height, fm_width)) # (HW * HW) * (HW * 1) = (HW * 1) ---> 1 x H x W
            ## expand dims
            mt_gamma = tf.tile(mm_gamma, [fm_channel, 1, 1])  # C x H x W
            mt_beta = tf.tile(mm_beta, [fm_channel, 1, 1]) # C x H x W

            return mt_gamma, mt_beta

    ##################################################################################
    # MANet
    ##################################################################################

    def manet_module(self, x_init, y_init, amm_callback, x_68_coord, y_68_coord, x_mask, y_mask, reuse=False, scope="manet_model"):
        channel = self.man_ch
        #c = tf.cast(tf.reshape(c, shape=[-1, 1, 1, c.shape[-1]]), tf.float32)
        #c = tf.tile(c, [1, x_init.shape[1], x_init.shape[2], 1])
        #x = tf.concat([x_init, c], axis=-1)

        # The shape of x_init and y_init  are : [1, 256, 256, 3]
        # The shape of x_68_coord, y_68_coord are: [68, 2]
        # The shape of x_mask, y_mask are : [3, 256, 256] ----0/1 type from A/B_input_mask

        with tf.variable_scope(scope, reuse=reuse):
            # ------------------------------------------------------------------------------- #
            # Firstly, call MDNet to generate input of AMM: the feature map of 3rd bottleneck
            # ------------------------------------------------------------------------------- #
            mdn_o = self.mdnet_module(y_init, reuse=True, scope='mdnet') # mdn_o: 1*64*64*256
            print("mdn mdn_o : ", mdn_o)

            # ------------------------------------------------------------------------------- #
            # Secondly, do the encoder-bottleneck part of MANet
            # ------------------------------------------------------------------------------- #
            x_conv = tf.pad(x_init, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            man_o = general_conv2d(x_conv, channel, kernel=7, stride=1, padding="VALID",
                                    name="MANet_conv", do_norm=True, do_relu=True)    # conv: 1*256*256*64
            print("man conv : ", man_o)

            ## Down-Sampling
            for i in range(2):
                man_o = tf.pad(man_o, [[0,0], [1, 1], [1, 1], [0, 0]], "REFLECT")
                man_o = general_conv2d(man_o, channel*2, kernel=4, stride=2, padding="VALID", name="MANet_conv_"+str(i), do_norm=True, do_relu=True)

                channel = channel * 2   # x_conv: 1*64*64*256
            print("man donwsampling : ", man_o)

            ## Bottleneck 1-3
            for i in range(1, self.man_n_res + 1):
                man_o = resblock(man_o, channel, name='MANet_resblock_' + str(i))  # man_o : 1*64*64*256
            print("man man_o : ", man_o)

            ## feature map shape
            fm_H = man_o.get_shape().as_list()[1]  #64
            fm_W = man_o.get_shape().as_list()[2]  #64
            fm_C = man_o.get_shape().as_list()[3]  #256

            # ------------------------------------------------------------------------------- #
            # Thirdly, callback AMM to get morphed-makeup matrices mmt_gamma and mmt_beta
            # ------------------------------------------------------------------------------- #
            ## mask shape
            mx_H = x_mask.get_shape().as_list()[1]
            mx_W = x_mask.get_shape().as_list()[2]
            my_H = y_mask.get_shape().as_list()[1]
            my_W = y_mask.get_shape().as_list()[2]

            ## Scaling mask to the same size of feature map: H x W
            ###lip
            x_lip_tmp = tf.image.resize_images(tf.reshape(tf.cast(x_mask, tf.float32)[0,:,:], (mx_H, mx_W, 1)), [fm_H, fm_W]) # [256, 256, 1] --->  [64, 64, 1]
            y_lip_tmp = tf.image.resize_images(tf.reshape(tf.cast(y_mask, tf.float32)[0,:,:], (my_H, my_W, 1)), [fm_H, fm_W])
            ###eye
            x_eye_tmp = tf.image.resize_images(tf.reshape(tf.cast(x_mask, tf.float32)[1,:,:], (mx_H, mx_W, 1)), [fm_H, fm_W]) # [256, 256, 1] --->  [64, 64, 1]
            y_eye_tmp = tf.image.resize_images(tf.reshape(tf.cast(y_mask, tf.float32)[1,:,:], (my_H, my_W, 1)), [fm_H, fm_W])
            ###skin
            x_skin_tmp = tf.image.resize_images(tf.reshape(tf.cast(x_mask, tf.float32)[2,:,:], (mx_H, mx_W, 1)), [fm_H, fm_W]) # [256, 256, 1] ---> [64, 64, 1]
            y_skin_tmp = tf.image.resize_images(tf.reshape(tf.cast(y_mask, tf.float32)[2,:,:], (my_H, my_W, 1)), [fm_H, fm_W])

            ## reshape each mask part from H x W to HW x 1
            x_mask_lip = tf.reshape(x_lip_tmp, (fm_H * fm_W, 1)) # [64*64, 1]
            y_mask_lip = tf.reshape(y_lip_tmp, (1, fm_H * fm_W)) # [1, 64*64]
            lip_mask = tf.matmul(x_mask_lip, y_mask_lip)  # 64*64 * 64*64

            x_mask_eye = tf.reshape(x_eye_tmp, (fm_H * fm_W, 1))
            y_mask_eye = tf.reshape(y_eye_tmp, (1, fm_H * fm_W))
            eye_mask = tf.matmul(x_mask_eye, y_mask_eye)

            x_mask_skin = tf.reshape(x_skin_tmp, (fm_H * fm_W, 1))
            y_mask_skin = tf.reshape(y_skin_tmp, (1, fm_H * fm_W))
            skin_mask = tf.matmul(x_mask_skin, y_mask_skin)
            ##
            mask_full = []
            mask_full = [lip_mask, eye_mask, skin_mask]
            print("mask_full : ", mask_full)

            # resize origin image to the same size of feature map to calculate relative postion feature
            x_init_temp = tf.image.resize_images(x_init, [mx_H, mx_W]) # 1*64*64*3
            y_init_temp = tf.image.resize_images(y_init, [my_H, my_W])
            print("x_init_temp : ", x_init_temp)
            mmt_gm, mmt_bt = amm_callback(mdn_o, man_o, x_init, y_init, fm_H, fm_W, fm_C,
                                        x_68_coord, y_68_coord, mask_full, reuse=True, scope='amm_callback')  # C x H x W, C x H x W
            print("mmt_gm : ", mmt_gm) # 256, 64, 64
            print("mmt_bt : ", mmt_bt)
            # --------------------------------------------------------------------------------- #
            # apply morphed makeup matrices mmt_gm and mmt_bt to MANet #
            # --------------------------------------------------------------------------------- #
            v_x = man_o    # C x H x W (1 * 64 * 64 * 256)
            v_x = tf.reshape(mmt_gm, (1, mmt_gm.get_shape().as_list()[1], mmt_gm.get_shape().as_list()[2],
                                mmt_gm.get_shape().as_list()[0])) * v_x + tf.reshape(mmt_bt,(1,
                                mmt_bt.get_shape().as_list()[1], mmt_bt.get_shape().as_list()[2],
                                mmt_bt.get_shape().as_list()[0])) # 1 * 64 * 64 * 256

            # Bottleneck 4-6
            for i in range(4, self.man_n_res + 4):
                v_x = resblock(v_x, channel, name='MANet_resblock_' + str(i))  # C x H x W (1 * 64 * 64 * 256)
            print("man bottleneck : ", v_x)

            # Up-Sampling
            for i in range(2):
                #v_x = tf.pad(v_x, [[0,0], [1, 1], [1, 1], [0, 0]], "REFLECT")
                v_x = general_deconv2d(v_x, channel // 2, kernel=4, stride=2, padding="SAME", name="MANet_deconv_"+str(i), do_norm=True, do_relu=True)

                channel = channel // 2 # 1 * 256 * 256 * 64
                print("channel : ", channel)
            print("man upsampling deconv : ", v_x)

            res = tf.pad(v_x, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            res = general_conv2d(res, channels=3, kernel=7, stride=1, padding="VALID", name="MANet_G_logit")  # 1 * 256 * 256 * 3
            print("man upsampling conv : ", res)
            res = tf.tanh(res, name="MANet_tanh")

            return res


    ##################################################################################
    # Generator
    ##################################################################################

    def build_generator(self, x_init, y_init, x_ldmk, y_ldmk, x_mask, y_mask, reuse=False, scope='generator'):
        with tf.variable_scope(scope, reuse=reuse):
            out_gen = self.manet_module(x_init, y_init, self.amm_module, x_ldmk, y_ldmk, x_mask, y_mask, reuse=True, scope='MAN')
            print("generator output : ", out_gen)
            return out_gen # 1*256*256*3

    ##################################################################################
    # Discriminator
    ##################################################################################

    def build_discriminator(self, input_gen, reuse=False, scope='discriminator'):
        """
        :param inputdis: 1*256*256*3
        :param name:
        :return:
        """
        with tf.variable_scope(scope, reuse=False):
            f = 4
            # 目前的spectral normlization 有点问题
            print("discriminator input : ", input_gen)
            oc_1 = general_conv2d(input_gen,64,f,2,"SAME",name="dis_c1",do_norm=False,relufactor=0.2)  # 1*128*128*64
            oc_2 = general_conv2d(oc_1,128,f,2,"SAME",name="dis_c2",do_norm=False,do_relu=True,relufactor=0.2)  # 1*64*64*128
            oc_3 = general_conv2d(oc_2,256,f,2,"SAME",name="dis_c3",do_norm=False,do_relu=True,relufactor=0.2)  # 1*32*32*256
            oc_4 = general_conv2d(oc_3,512,f,1,"SAME",name="dis_c4",do_norm=False,do_relu=True,relufactor=0.2)  # 1*32*32*512
            oc_5 = general_conv2d(oc_4,1,f,1,"SAME",name="dis_c5",do_norm=False,do_relu=False)  # 1*32*32*1
            print("discriminator res : ", oc_5)
            return oc_5

    ##################################################################################
    # Model
    ##################################################################################

    def model_setup(self):

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.input_S = tf.placeholder(tf.float32,[self.batch_size, self.img_height, self.img_width, self.img_ch],name="input_S")
        self.input_R = tf.placeholder(tf.float32,[self.batch_size, self.img_height, self.img_width, self.img_ch],name="input_R")

        self.input_S_mask = tf.placeholder(tf.bool, [3, self.img_height, self.img_width], name="input_S_mask")
        self.input_R_mask = tf.placeholder(tf.bool, [3, self.img_height, self.img_width], name="input_R_mask")

        # self.S_mask_G = tf.placeholder(tf.bool, [self.img_height, img_width, 1], name='S_mask_G')  # a full mask for genrator [256, 256, 1]
        # self.R_mask_G = tf.placeholder(tf.bool, [self.img_height, img_width, 1], name='S_mask_R')

        self.fake_pool_S = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_ch], name="fake_pool_S")
        self.fake_pool_R = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_ch], name="fake_pool_R")

        self.num_fake_inputs = 0
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # self.landmarks_S = tf.compat.v1.placeholder(tf.int32, [self.batch_size, 68, 2], name='landmarks_S')
        # self.landmarks_R = tf.compat.v1.placeholder(tf.int32, [self.batch_size, 68, 2], name='landmarks_R')
        self.landmarks_S = tf.placeholder(tf.int32, [68, 2], name='landmarks_S')
        self.landmarks_R = tf.placeholder(tf.int32, [68, 2], name='landmarks_R')

        # self.landmarks_fake_pool_S = tf.placeholder(tf.float32, [None, 68, 2], name='landmarks_fake_pool_S')
        # self.landmarks_fake_pool_R = tf.placeholder(tf.float32, [None, 68, 2], name='landmarks_fake_pool_R')

        # self.predictor = dlib.shape_predictor("./preTrainedModel/shape_predictor_68_face_landmarks.dat")
        # self.detector = dlib.get_frontal_face_detector()

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)  # beta1 衰减率/* ref BeautyGAN */
        with tf.variable_scope("Model") as scope:
            self.fake_Ss = []
            self.fake_Rs = []

            self.fake_S = self.build_generator(self.input_S, self.input_R, self.landmarks_S, self.landmarks_R,
                                                self.input_S_mask, self.input_R_mask, reuse=True, scope="g_S")  # G(x,y)
            self.fake_R = self.build_generator(self.input_R, self.input_S, self.landmarks_R, self.landmarks_S,
                                                self.input_R_mask, self.input_S_mask, reuse=True, scope="g_R")  # G(y,x)
            self.fake_Ss.append(self.fake_S)   #  data parallelism
            self.fake_Rs.append(self.fake_R)
            self.rec_S = self.build_discriminator(self.input_S, scope='d_S')                        # D(x)
            self.rec_R = self.build_discriminator(self.input_R, scope='d_R')                        # D(y)

            scope.reuse_variables()

            self.fake_rec_S = self.build_discriminator(self.fake_S, scope='d_S')                    # D(G(x,y))
            self.fake_rec_R = self.build_discriminator(self.fake_R, scope='d_R')                    # D(G(y,x))
            self.cyc_S = self.build_generator(self.fake_S, self.input_S, self.landmarks_S, self.landmarks_S, self.input_S_mask, self.input_S_mask, reuse=True, scope='g_S')         # G(G(x,y), x)
            self.cyc_R = self.build_generator(self.fake_R, self.input_R, self.landmarks_R, self.landmarks_R, self.input_R_mask, self.input_R_mask, reuse=True, scope='g_R')         # G(G(y,x), y)

            scope.reuse_variables()

            self.fake_pool_rec_S = self.build_discriminator(self.fake_pool_S, 'd_S')
            self.fake_pool_rec_R = self.build_discriminator(self.fake_pool_R, 'd_R')

            self.perc_S = tf.cast(tf.image.resize_images((self.input_S+1)*127.5,[224,224]),tf.float32)  # resize to 224 for inputs requiement of vgg-16
            self.perc_R = tf.cast(tf.image.resize_images((self.input_R+1)*127.5, [224, 224]), tf.float32)
            self.perc_fake_S = tf.cast(tf.image.resize_images((self.fake_S+1)*127.5, [224, 224]), tf.float32)
            self.perc_fake_R = tf.cast(tf.image.resize_images((self.fake_R+1)*127.5, [224, 224]), tf.float32)
            self.perc = self.perc_loss_cal(tf.concat([self.perc_S,self.perc_R,self.perc_fake_S,self.perc_fake_R],axis=0))
            percep_norm,var = tf.nn.moments(self.perc, [1, 2], keep_dims=True)
            self.perc = tf.divide(self.perc,tf.add(percep_norm,1e-5))


    ##################################################################################
    # Loss Functions
    ##################################################################################

    def perc_loss_cal(self, input_tensor):
        vgg = vgg16.Vgg16("./preTrainedModel/vgg16.npy")
        vgg.build(input_tensor)
        return vgg.conv4_1

    def histogram_loss_cal(self, source, template, source_mask, template_mask):
        shape = tf.shape(source)   # shape [x y z ...]
        source = tf.reshape(source, [1, -1]) # 拉平
        template = tf.reshape(template, [1, -1])
        source_mask = tf.reshape(source_mask, [-1, 256 * 256])
        template_mask = tf.reshape(template_mask, [-1, 256 * 256])

        source = tf.boolean_mask(source, source_mask) # 值计算当前mask为True的区域(lip or eye/ or face)
        template = tf.boolean_mask(template, template_mask) #save True part同上
        his_bins = 255

        max_value = tf.reduce_max([tf.reduce_max(source), tf.reduce_max(template)])  # find max value(hist) with fake_S/R + input_S/R range
        min_value = tf.reduce_min([tf.reduce_min(source), tf.reduce_min(template)])

        hist_delta = (max_value - min_value) / his_bins  #极差/bins
        hist_range = tf.range(min_value, max_value, hist_delta)
        hist_range = tf.add(hist_range, tf.divide(hist_delta, 2))

        s_hist = tf.histogram_fixed_width(source, [min_value, max_value], his_bins, dtype=tf.int32)
        t_hist = tf.histogram_fixed_width(template, [min_value, max_value], his_bins, dtype=tf.int32)

        s_quantiles = tf.cumsum(s_hist) #计算fake_S/R累计直方图
        s_last_element = tf.subtract(tf.size(s_quantiles), tf.constant(1)) # 256*256 - 1
        s_quantiles = tf.divide(s_quantiles, tf.gather(s_quantiles, s_last_element))  # 把累计直方图计算结果除以最后一排累计数据 来得到归一化结果

        t_quantiles = tf.cumsum(t_hist) #计算input_S/R累计直方图
        t_last_element = tf.subtract(tf.size(t_quantiles), tf.constant(1))
        t_quantiles = tf.divide(t_quantiles, tf.gather(t_quantiles, t_last_element))

        nearest_indices = tf.map_fn(lambda x: tf.argmin(tf.abs(tf.subtract(t_quantiles, x))), s_quantiles,
                                    dtype=tf.int64)  ## 找每列最小值的坐标索引（|t_qualities - s_qualities|）
        s_bin_index = tf.to_int64(tf.divide(source, hist_delta))
        s_bin_index = tf.clip_by_value(s_bin_index, 0, 254)

        matched_to_t = tf.gather(hist_range, tf.gather(nearest_indices, s_bin_index))
        # Using the same normalization as Gatys' style transfer: A huge variation--the normalization scalar is different according to different image
        # normalization includes variation constraints may be better
        matched_to_t = tf.subtract(tf.div(matched_to_t,127.5),1)
        source = tf.subtract(tf.divide(source,127.5),1)
        return tf.reduce_mean(tf.squared_difference(matched_to_t,source))


    def loss_calc(self):
        "Adversarial loss"
        '''BeautyGAN replace negative log likelihood objective in adv_loss by a least square loss
           to stablize the training procedure and generate high quality images'''
        g_loss_S = tf.reduce_mean(tf.squared_difference(self.fake_rec_S, 1))  # train the G to minimize  (D(G(x,y)) - 1)^2
        g_loss_R = tf.reduce_mean(tf.squared_difference(self.fake_rec_R, 1))

        d_loss_S = (tf.reduce_mean(tf.square(self.fake_pool_rec_S)) + tf.reduce_mean(tf.squared_difference(self.rec_S, 1))) / 2.0
        d_loss_R = (tf.reduce_mean(tf.square(self.fake_pool_rec_R)) + tf.reduce_mean(tf.squared_difference(self.rec_R, 1))) / 2.0
        d_loss = (d_loss_S + d_loss_R)

        "Cycle consistency loss"
        #cyc_loss = tf.reduce_mean(tf.abs(G(G(x,y),x) - x)) + tf.reduce_mean(tf.abs(G(G(y,x),y) - y))
        cyc_loss = tf.reduce_mean(tf.abs(self.cyc_S - self.input_S)) + tf.reduce_mean(tf.abs(self.cyc_R - self.input_R))

        "Perceptual loss"
        #perceptual_loss = tf.reduce_mean(tf.squared_difference(F(G(x,y)), F(x))) + tf.reduce_mean(tf.squared_difference(F(G(y,x)), F(y)))
        perceptual_loss = tf.reduce_mean(tf.squared_difference(self.perc[2], self.perc[0])) + tf.reduce_mean(tf.squared_difference(self.perc[3], self.perc[1]))

        "Makeup loss"
        '''calculate histogram loss on rgb channel respectively and recombine them'''
        '''||G(x,y) - HW(x,y)||_2'''
        temp_source = tf.cast((self.fake_S[0, :, :, 0] + 1) * 127.5, dtype=tf.float32)
        temp_template = tf.cast((self.input_S[0, :, :, 0] + 1) * 127.5, dtype=tf.float32)
        histogram_loss_lip_r = self.histogram_loss_cal(temp_source, temp_template, self.input_S_mask[0], self.input_R_mask[0])
        histogram_loss_eye_r = self.histogram_loss_cal(temp_source, temp_template, self.input_S_mask[1], self.input_R_mask[1])
        histogram_loss_face_r = self.histogram_loss_cal(temp_source, temp_template, self.input_S_mask[2], self.input_R_mask[2])
        S_histogram_loss_r = histogram_loss_lip_r + histogram_loss_eye_r + histogram_loss_face_r

        temp_source = tf.cast((self.fake_S[0, :, :, 1] + 1) * 127.5, dtype=tf.float32)
        temp_template = tf.cast((self.input_S[0, :, :, 1] + 1) * 127.5, dtype=tf.float32)
        histogram_loss_lip_g = self.histogram_loss_cal(temp_source, temp_template, self.input_S_mask[0], self.input_R_mask[0])
        histogram_loss_eye_g = self.histogram_loss_cal(temp_source, temp_template, self.input_S_mask[1], self.input_R_mask[1])
        histogram_loss_face_g = self.histogram_loss_cal(temp_source, temp_template, self.input_S_mask[2], self.input_R_mask[2])
        S_histogram_loss_g = histogram_loss_lip_g + histogram_loss_eye_g + histogram_loss_face_g

        temp_source = tf.cast((self.fake_S[0, :, :, 2] + 1) * 127.5, dtype=tf.float32)
        temp_template = tf.cast((self.input_S[0, :, :, 0] + 1) * 127.5, dtype=tf.float32)
        histogram_loss_lip_b = self.histogram_loss_cal(temp_source, temp_template, self.input_S_mask[0], self.input_R_mask[0])
        histogram_loss_eye_b = self.histogram_loss_cal(temp_source, temp_template, self.input_S_mask[1], self.input_R_mask[1])
        histogram_loss_face_b = self.histogram_loss_cal(temp_source, temp_template, self.input_S_mask[2], self.input_R_mask[2])
        S_histogram_loss_b = histogram_loss_lip_b + histogram_loss_eye_b + histogram_loss_face_b

        S_histogram_loss = S_histogram_loss_r + S_histogram_loss_g + S_histogram_loss_b

        '''||G(y,x) - HW(y,x)||_2'''
        temp_source = tf.cast((self.fake_R[0, :, :, 0] + 1) * 127.5, dtype=tf.float32)
        temp_template = tf.cast((self.input_R[0, :, :, 0] + 1) * 127.5, dtype=tf.float32)
        histogram_loss_lip_r = self.histogram_loss_cal(temp_source, temp_template, self.input_R_mask[0], self.input_S_mask[0])
        histogram_loss_eye_r = self.histogram_loss_cal(temp_source, temp_template, self.input_R_mask[1], self.input_S_mask[1])
        histogram_loss_face_r = self.histogram_loss_cal(temp_source, temp_template, self.input_R_mask[2], self.input_S_mask[2])
        R_histogram_loss_r = histogram_loss_lip_r + histogram_loss_eye_r + histogram_loss_face_r

        temp_source = tf.cast((self.fake_R[0, :, :, 1] + 1) * 127.5, dtype=tf.float32)
        temp_template = tf.cast((self.input_R[0, :, :, 1] + 1) * 127.5, dtype=tf.float32)
        histogram_loss_lip_g = self.histogram_loss_cal(temp_source, temp_template, self.input_R_mask[0], self.input_S_mask[0])
        histogram_loss_eye_g = self.histogram_loss_cal(temp_source, temp_template, self.input_R_mask[1], self.input_S_mask[1])
        histogram_loss_face_g = self.histogram_loss_cal(temp_source, temp_template, self.input_R_mask[2], self.input_S_mask[2])
        R_histogram_loss_g = histogram_loss_lip_g + histogram_loss_eye_g + histogram_loss_face_g

        temp_source = tf.cast((self.fake_R[0, :, :, 2] + 1) * 127.5, dtype=tf.float32)
        temp_template = tf.cast((self.input_R[0, :, :, 0] + 1) * 127.5, dtype=tf.float32)
        histogram_loss_lip_b = self.histogram_loss_cal(temp_source, temp_template, self.input_R_mask[0], self.input_S_mask[0])
        histogram_loss_eye_b = self.histogram_loss_cal(temp_source, temp_template, self.input_R_mask[1], self.input_S_mask[1])
        histogram_loss_face_b = self.histogram_loss_cal(temp_source, temp_template, self.input_R_mask[2], self.input_S_mask[2])
        R_histogram_loss_b = histogram_loss_lip_b + histogram_loss_eye_b + histogram_loss_face_b

        R_histogram_loss = R_histogram_loss_r + R_histogram_loss_g + R_histogram_loss_b
        makeup_loss = S_histogram_loss + R_histogram_loss

        "Total loss"
        g_loss = self.adv_ld * (-1) * (g_loss_S + g_loss_R) + self.cyc_ld * cyc_loss + self.per_ld * perceptual_loss + self.make_ld * makeup_loss
        d_loss = self.adv_ld * (-1) * d_loss

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        ''' define training variables '''
        d_S_vars = [var for var in self.model_vars if "d_S" in var.name]
        d_R_vars = [var for var in self.model_vars if "d_R" in var.name]
        g_S_vars = [var for var in self.model_vars if "g_S" in var.name]
        g_R_vars = [var for var in self.model_vars if "g_R" in var.name]

        self.d_S_trainer = optimizer.minimize(d_loss_S, var_list=d_S_vars)
        self.d_R_trainer = optimizer.minimize(d_loss_R, var_list=d_R_vars)
        self.g_S_trainer = optimizer.minimize(g_loss_S, var_list=g_S_vars)
        self.g_R_trainer = optimizer.minimize(g_loss_R, var_list=g_R_vars)

        # g_vars = [var for var in self.model_vars if "generator" in var.name]
        # d_vars = [var for var in self.model_vars if "discriminator" in var.name]

        # self.g_trainer = optimizer.minimize(g_loss, var_list=g_vars)
        # self.d_trainer = optimizer.minimize(d_loss, var_list=d_vars)

        for var in self.model_vars:
            print(var.name)

        # Summary variables for tensorboard
        self.g_S_loss_sum = tf.summary.scalar('g_S_loss', g_loss_S)
        self.g_R_loss_sum = tf.summary.scalar('g_R_loss', g_loss_R)
        self.cyc_loss_sum = tf.summary.scalar('cyc_loss', cyc_loss)
        self.makeup_loss_sum = tf.summary.scalar('makeup_loss', makeup_loss)
        self.percep_loss_sum = tf.summary.scalar('perceptual_loss', perceptual_loss)
        self.g_loss_sum = tf.summary.scalar('g_loss', g_loss)

        self.g_summary = tf.summary.merge([
            self.g_S_loss_sum, self.g_R_loss_sum, self.cyc_loss_sum, self.makeup_loss_sum, self.percep_loss_sum, self.g_loss_sum,
        ], "g_summary")

        self.d_S_loss_sum = tf.summary.scalar('d_S_loss', d_loss_S)
        self.d_R_loss_sum = tf.summary.scalar('d_R_loss', d_loss_R)
        #self.d_S_loss_sum = tf.summary.scalar('d_loss', d_loss)


    def save_training_images(self, sess, epoch):
        if not os.path.exists("./output/images"):
            os.makedirs("./output/images")

        for i in range(10):
            fake_S_tmp, fake_R_tmp, cyc_S_tmp, cyc_R_tmp = sess.run([self.fake_S, self.fake_R, self.cyc_S, self.cyc_R], feed_dict={
                self.input_S:self.input_S[i],
                self.input_R:self.input_R[i]
            })
        imsave("./output/images/fakeS_"+str(epoch)+"_"+str(i)+".jpg", ((fake_S_tmp[0] + 1) * 127.5).astype(np.uint8))
        imsave("./output/images/fakeR_"+str(epoch)+"_"+str(i)+".jpg", ((fake_R_tmp[0] + 1) * 127.5).astype(np.uint8))
        imsave("./output/images/cycS_"+str(epoch)+"_"+str(i)+".jpg", ((cyc_S_tmp[0] + 1) * 127.5).astype(np.uint8))
        imsave("./output/images/cycR_"+str(epoch)+"_"+str(i)+".jpg", ((cyc_R_tmp[0] + 1) * 127.5).astype(np.uint8))

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        ''' This function saves the generated image to corresponding pool of images.
        In starting. It keeps on filling the pool till it is full and then randomly selects an
        already-stored image and replace it with a new one.'''
        if num_fakes < self.pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()  # p is a float number in [0,1)
            if p > 0.5:
                random_id = random.randint(0, self.pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    def train(self):

        ''' Training Functions '''

        # initialize all variables
        # tf.global_variables_initializer().run()
        # tf.local_variables_initializer().run()

        # Load dataset from the dataset folder
        Input_img_class = InputData(img_height=self.img_height, img_width=self.img_width, img_channels=self.img_ch,
                batch_size=self.batch_size, pool_size=self.pool_size, data_path = self.dataset_path, use_segs=self.use_segs_flag)
        Input_img_class.input_setup()

        # Build the network
        self.model_setup()

        # Loss function calculations
        self.loss_calc()

        # Initialize the global variables
        init = (tf.local_variables_initializer(), tf.global_variables_initializer())
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            #sess.run(tf.local_variables_initializer())
            # Read input to nd array
            Input_img_class.input_read(sess)

            #Restore the model to run the model from list checkpoint
            if to_restore:
                checkpoints_fn = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, checkpoints_fn)

            writer = tf.summary.FileWriter("./output/2")

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            # Training Loop
            for epoch in range(sess.run(self.global_step), self.epoch):
                print(" In the epoch ", epoch)
                saver.save(sess, os.path.join(check_dir, "PSGAN"), global_step=epoch)

                # use linear decay ?
                if self.decay_flag:
                    if epoch < self.epoch:
                        curr_lr = self.init_lr
                    else:
                        #curr_lr = self.init_lr - self.init_lr * (epoch-self.epoch)/$decay_rate
                        curr_lr = self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch)

                if save_training_imgs:
                    self.save_training_images(sess, epoch)

                for ptr in range(0, Input_img_class.train_num):
                    print(" In the iteration", ptr)
                    print(time.ctime())

                    # Optimizing the Generator S network
                    _, fake_S_tmp, summary_str = sess.run([self.g_S_trainer, self.fake_S, self.g_summary], feed_dict={
                        self.input_S:Input_img_class.A_input[ptr],
                        self.input_R:Input_img_class.B_input[ptr],
                        self.lr:curr_lr,
                        self.landmarks_S:Input_img_class.A_landmark[ptr],
                        self.landmarks_R:Input_img_class.B_landmark[ptr],
                        self.input_S_mask:Input_img_class.A_input_mask[ptr],
                        self.input_R_mask:Input_img_class.B_input_mask[ptr],
                    })
                    writer.add_summary(summary_str, epoch * Input_img_class.train_num + ptr)

                    # Optimize the Generator R network
                    _, fake_R_tmp, summary_str = sess.run([self.g_R_trainer, self.fake_R, self.g_summary], feed_dict={
                        self.input_R:Input_img_class.B_input[ptr],
                        self.input_S:Input_img_class.A_input[ptr],
                        self.lr:curr_lr,
                        self.landmarks_R:Input_img_class.B_landmark[ptr],
                        self.landmarks_S:Input_img_class.A_landmark[ptr],
                        self.input_R_mask:Input_img_class.B_input_mask[ptr],
                        self.input_S_mask:Input_img_class.A_input_mask[ptr],
                    })
                    writer.add_summary(summary_str, epoch * Input_img_class.train_num + ptr)

                    fake_S_tmp1 = self.fake_image_pool(self.num_fake_inputs, fake_S_tmp, Input_img_class.fake_images_A)
                    fake_R_tmp1 = self.fake_image_pool(self.num_fake_inputs, fake_R_tmp, Input_img_class.fake_images_B)

                    # Optimize the Discriminator S network
                    _, summary_str = sess.run([self.d_S_trainer, self.d_S_loss_sum], feed_dict={
                        self.input_S:Input_img_class.A_input[ptr],
                        self.input_R:Input_img_class.B_input[ptr],
                        self.lr:curr_lr,
                        self.fake_pool_S:fake_S_tmp1
                    })
                    writer.add_summary(summary_str, epoch * Input_img_class.train_num + ptr)

                    # Optimize the Discriminator R network
                    _, summary_str = sess.run([self.d_R_trainer, self.d_R_loss_sum], feed_dict={
                        self.input_S:Input_img_class.A_input[ptr],
                        self.input_R:Input_img_class.B_input[ptr],
                        self.lr:curr_lr,
                        self.fake_pool_R:fake_R_tmp1
                    })
                    writer.add_summary(summary_str, epoch * Input_img_class.train_num + ptr)

                    self.num_fake_inputs += 1
                sess.run(tf.assign(self.global_step, epoch + 1))

















