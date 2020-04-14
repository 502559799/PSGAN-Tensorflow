import numpy as np
import os
import scipy.misc
from scipy import misc
import cv2
import pickle
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import dlib

max_images = 900
load_dir = "all_imgs.txt"
lips_ft = True
eyes_ft = True
skin_ft = True
use_np = True

class InputData:

    def __init__(self, img_height, img_width, img_channels, batch_size, pool_size, data_path, use_segs, augment_flag=False):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = img_channels
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.use_segs_flag = use_segs
        self.augment_flag = augment_flag
        #self.selected_attrs = selected_attrs

        self.data_path_A = os.path.join(data_path, 'train/images/non-makeup/')
        check_folder(self.data_path_A)
        self.data_path_B = os.path.join(data_path, 'train/images/makeup/')
        check_folder(self.data_path_B)
        self.data_path_C = os.path.join(data_path, 'train/segs/non-makup/')
        check_folder(self.data_path_C)
        self.data_path_D = os.path.join(data_path, 'train/segs/makeup/')
        check_folder(self.data_path_D)

        #self.lines = open(os.path.join(data_path, ''))

        self.train_dataset = []
        self.train_dataset_label = []
        self.train_dataset_fix_label = []

        self.test_dataset = []
        self.test_dataset_label = []
        self.test_dataset_fix_label = []


    def input_setup(self):
        """
        dataset_A:non-makeup
        dataset_B:makeup
        :return:

        This function basically setup variables for taking image input.

        filenames_A/filenames_B -> takes the list of all training images
        filenames_C/filenames_D -> takes the list of all training segs
        self.image_A/self.image_B -> Input image with each values ranging from [-1,1]
        self.image_C/self.image_D -> Input seg with each values ranging from [-1,1]
        """
        filenames_A = tf.train.match_filenames_once(os.path.join(self.data_path_A, "*.png"))
        self.queue_length_A = tf.size(filenames_A)
        filenames_B = tf.train.match_filenames_once(os.path.join(self.data_path_B, "*.png"))
        self.queue_length_B = tf.size(filenames_B)

        filenames_C = tf.train.match_filenames_once(os.path.join(self.data_path_C, "*.png"))
        self.queue_length_C = tf.size(filenames_C)
        filenames_D = tf.train.match_filenames_once(os.path.join(self.data_path_D, "*.png"))
        self.queue_length_D = tf.size(filenames_D)

        #tensorflow中特有的文件名队列 string_input_producer将传入的文件名list转化成文件名队列
        filenames_queue_A = tf.train.string_input_producer(filenames_A, shuffle=False)
        filenames_queue_B = tf.train.string_input_producer(filenames_B, shuffle=False)

        filenames_queue_C = tf.train.string_input_producer(filenames_C, shuffle=False)
        filenames_queue_D = tf.train.string_input_producer(filenames_D, shuffle=False)
        print("fn_A : ", filenames_A)
        print("fn_B : ", filenames_B)
        print("fn_C : ", filenames_C)
        print("fn_D : ", filenames_D)

        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filenames_queue_A)
        _, image_file_B = image_reader.read(filenames_queue_B)
        _, image_file_C = image_reader.read(filenames_queue_C)
        _, image_file_D = image_reader.read(filenames_queue_D)

        self.image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A), [self.img_height, self.img_width]), 127.5), 1)  # [256, 256, 3]
        self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B), [self.img_height, self.img_width]), 127.5), 1)
        self.mask_A = tf.image.resize_images(tf.image.decode_jpeg(image_file_C), [self.img_height, self.img_width])
        self.mask_B = tf.image.resize_images(tf.image.decode_jpeg(image_file_D), [self.img_height, self.img_width])  # [256, 256, 1]
        print("input_setup image_A : ", self.image_A)
        print("input_setup mask_A : ", self.mask_A)
        print("input_setup mask_B : ", self.mask_B)


    def input_read(self, sess):

        '''
        It reads the input into from the image folder.

        self.fake_images_A/self.fake_images_B -> List of generated images used for calculation of loss function of Discriminator
        self.A_input/self.B_input -> Stores all the training images in python list
        self.A_input_mask/self.B_input_mask -> Stores all the training face mask in python list
        '''

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        num_files_A = sess.run(self.queue_length_A)
        num_files_B = sess.run(self.queue_length_B)
        num_files_C = sess.run(self.queue_length_C)
        num_files_D = sess.run(self.queue_length_D)
        print("num_files A/B/C/D : ", num_files_A, num_files_B, num_files_C, num_files_D)

        self.fake_images_A = np.zeros((self.pool_size, 1, self.img_height, self.img_width, self.channels))
        self.fake_images_B = np.zeros((self.pool_size, 1, self.img_height, self.img_width, self.channels))

        self.A_input = np.zeros((max_images, self.batch_size, self.img_height, self.img_width, self.channels))
        self.B_input = np.zeros((max_images, self.batch_size, self.img_height, self.img_width, self.channels))

        self.A_input_mask = np.zeros((max_images, 3, self.img_height, self.img_width))  # lip / eye / face
        self.B_input_mask = np.zeros((max_images, 3, self.img_height, self.img_width))

        self.A_mask_G = np.zeros((max_images, self.img_height, self.img_width, 1))    #only pass to genrator directly, not for loss calc
        self.B_mask_G = np.zeros((max_images, self.img_height, self.img_width, 1))

        self.A_landmark = np.zeros((max_images, 68, 2))  # landmarks of input A
        self.B_landmark = np.zeros((max_images, 68, 2))  # landmarks of input B

        self.predictor = dlib.shape_predictor("./preTrainedModel/shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()

        # get 68-landmarks and target mask region from inputs
        cnt_A = 0
        cnt_B = 0
        if self.use_segs_flag:
            if not os.path.exists(load_dir):
                for i in range(max_images):
                    # mask attribute: 0:background 1:face 2:left-eyebrown 3:right-eyebrown 4:left-eye 5: right-eye 6: nose
                    # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck
                    ## lips
                    if lips_ft is True:
                        mask_A_lip = tf.cast(tf.equal(self.mask_A, 7), tf.float32) + tf.cast(tf.equal(self.mask_A, 9), tf.float32)  #   lip regions are 1 others are 0    size : [256 256 1]
                        mask_B_lip = tf.cast(tf.equal(self.mask_B, 7), tf.float32) + tf.cast(tf.equal(self.mask_B, 9), tf.float32)
                        mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)
                        #print("mask_A_lip : ", mask_A_lip.eval(sess))
                        #print("mask_A_lip check nonzero : ", tf.count_nonzero(mask_A_lip).eval(sess))
                        #print("mask_B_lip : ", mask_B_lip.eval(sess))
                    ## skin
                    if skin_ft is True:
                        mask_A_skin = tf.cast(tf.equal(self.mask_A, 1), tf.float32) + tf.cast(tf.equal(self.mask_A, 6), tf.float32) + tf.cast(tf.equal(self.mask_A, 13), tf.float32)
                        mask_B_skin = tf.cast(tf.equal(self.mask_B, 1), tf.float32) + tf.cast(tf.equal(self.mask_B, 6), tf.float32) + tf.cast(tf.equal(self.mask_B, 13), tf.float32)
                        mask_A_skin, mask_B_skin, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_skin, mask_B_skin)
                        #print("mask_A_skin : ", mask_A_skin.eval(sess))
                        #print("mask_B_skin : ", mask_B_skin.eval(sess))
                    ## eyes
                    if eyes_ft is True:
                        mask_A_eye_left = tf.cast(tf.equal(self.mask_A, 4), tf.float32)
                        mask_A_eye_right = tf.cast(tf.equal(self.mask_A, 5), tf.float32)

                        mask_B_eye_left = tf.cast(tf.equal(self.mask_B, 4), tf.float32)
                        mask_B_eye_right = tf.cast(tf.equal(self.mask_B, 5), tf.float32)

                        mask_A_face = tf.cast(tf.equal(self.mask_A, 1), tf.float32) + tf.cast(tf.equal(self.mask_A, 6), tf.float32)
                        mask_B_face = tf.cast(tf.equal(self.mask_B, 1), tf.float32) + tf.cast(tf.equal(self.mask_B, 6), tf.float32)
                        # avoid the situation that images with closed eyes
                        # if not ((mask_A_eye_left.eval() > 0.0).any() and (mask_B_eye_left.eval() > 0.0).any() and (mask_A_eye_right.eval() > 0.0).any() and (mask_B_eye_right.eval() > 0.0).any()):
                        #     continue
                        # tf.cond(tf.equal(tf.count_nonzero(mask_A_eye_left), 0) and tf.equal(tf.count_nonzero(mask_B_eye_left), 0) and
                        #         tf.equal(tf.count_nonzero(mask_A_eye_right), 0) and tf.equal(tf.count_nonzero(mask_B_eye_right), 0)):
                        #     continue
                        mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left, mask_A_eye_right, mask_A_face)
                        mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left, mask_B_eye_right, mask_B_face)

                        sess.run(mask_A_eye_left.initializer)
                        sess.run(mask_A_eye_right.initializer)
                        sess.run(mask_B_eye_left.initializer)
                        sess.run(mask_B_eye_right.initializer)
                        mask_A_eye = mask_A_eye_left + mask_A_eye_right
                        mask_B_eye = mask_B_eye_left + mask_B_eye_right

                        mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = \
                            self.mask_preprocess(mask_A_eye_left, mask_B_eye_left)

                        mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = \
                            self.mask_preprocess(mask_A_eye_right, mask_B_eye_right)

                        mask_A_eye, mask_B_eye, index_A_eye, index_B_eye = \
                            self.mask_preprocess(mask_A_eye, mask_B_eye)
                        print("mask_A_eye_left : ", mask_A_eye_left)
                        print("mask_A_eye_right : ", mask_A_eye_right)
                        print("mask_B_eye_left : ", mask_B_eye_left)
                        print("mask_B_eye_right : ", mask_B_eye_right)

                    #  convert image and mask tensor to nd array
                    image_tensor_A = sess.run(self.image_A)
                    mask_tensor_A = sess.run(self.mask_A)
                    # lip_tensor_A = sess.run(mask_A_lip) # bool nd array
                    # eye_tensor_A = sess.run(mask_A_eye)
                    # skin_tensor_A = sess.run(mask_A_skin)
                    if image_tensor_A.size == self.img_height * self.img_width * self.channels and mask_tensor_A.size == self.img_height * self.img_width * 1:
                        temp_image = ((image_tensor_A + 1) * 127.5).astype(np.uint8)
                        temp_mask = mask_tensor_A.astype(np.uint8)
                        self.A_input[cnt_A] = image_tensor_A.reshape(self.batch_size, self.img_height, self.img_width, self.channels)
                        self.A_landmark[cnt_A] = self.get_keypoints(temp_image, self.detector, self.predictor)
                        self.A_input_mask[cnt_A][0] = sess.run(mask_A_lip)[:,:,0] # [256, 256, 1] ---> [256, 256]   type: bool
                        self.A_input_mask[cnt_A][1] = sess.run(mask_A_eye)[:,:,0]
                        self.A_input_mask[cnt_A][2] = sess.run(mask_A_skin)[:,:,0]
                        self.A_mask_G[cnt_A] = temp_mask  # [256, 256, 1]
                        cnt_A += 1

                    image_tensor_B = sess.run(self.image_B)
                    mask_tensor_B = sess.run(self.mask_B)
                    # lip_tensor_B = sess.run(mask_B_lip) # bool nd array  [256, 256, 1]
                    # eye_tensor_B = sess.run(mask_B_eye)
                    # skin_tensor_B = sess.run(mask_B_skin)
                    if image_tensor_B.size == self.img_height * self.img_width * self.channels and mask_tensor_B.size == self.img_height * self.img_width * 1:
                        temp_image = ((image_tensor_B + 1) * 127.5).astype(np.uint8)
                        temp_mask = mask_tensor_B.astype(np.uint8)
                        self.B_input[cnt_B] = image_tensor_B.reshape(self.batch_size, self.img_height, self.img_width, self.channels)
                        self.B_landmark[cnt_B] = self.get_keypoints(temp_image, self.detector, self.predictor)
                        self.B_input_mask[cnt_B][0] = sess.run(mask_B_lip)[:,:,0]
                        self.B_input_mask[cnt_B][1] = sess.run(mask_B_eye)[:,:,0]
                        self.B_input_mask[cnt_B][2] = sess.run(mask_B_skin)[:,:,0]
                        self.B_mask_G[cnt_B] = temp_mask  # [256, 256, 1]
                        cnt_B += 1

                #os.mknod(load_dir)
                #open(load_dir, 'w').close()
                fw = open(load_dir, "wb")
                pickle.dump(self.A_landmark, fw)
                pickle.dump(self.B_landmark, fw)
                pickle.dump(self.A_input_mask, fw)
                pickle.dump(self.B_input_mask, fw)
                #pickle.dump(self.A_mask_G, fw)
                #pickle.dump(self.B_mask_G, fw)
                pickle.dump(cnt_A, fw)
                pickle.dump(cnt_B, fw)

            else:
                fr = open(load_dir, "rb")
                #self.mask_A = pickle.load(fr)
                #self.mask_B = pickle.load(fr)
                self.A_landmark = pickle.load(fr)
                self.B_landmark = pickle.load(fr)
                self.A_input_mask = pickle.load(fr)
                self.B_input_mask = pickle.load(fr)
                #self.A_mask_G = pickle.load(fr)
                #self.B_mask_G = pickle.load(fr)
                cnt_A = pickle.load(fr)
                cnt_B = pickle.load(fr)

            self.train_num = min(cnt_A, cnt_B)
            print("face mask num: ", self.train_num)

        else:
            # obtain face mask from input images (source/ref)
            if not os.path.exists(load_dir):
                cnt_A = 0
                for i in range(max_images):
                    image_tensor = sess.run(self.image_A)
                    if image_tensor == self.img_height * self.img_width * self.channels:
                        tmp = ((image_tensor + 1) * 127.5).astype(np.uint8)
                        res = self.get_mask(tmp, self.detector, self.predictor)
                        if res != None:
                            self.A_input[cnt_A] = image_tensor.reshape((self.batch_size, self.img_height, self.img_width, self.channels))
                            self.A_input_mask[cnt_A][0] = np.equal(res[0], 255)
                            self.A_input_mask[cnt_A][1] = np.equal(res[0], 255)
                            self.A_input_mask[cnt_A][2] = np.equal(res[0], 255)
                            cnt_A += 1

                cnt_B = 0
                for i in range(max_images):
                    image_tensor = sess.run(self.image_B)
                    if image_tensor.size == self.img_height * self.img_width * self.channels:
                        self.B_input[i] = image_tensor.reshape((self.batch_size, self.img_height, self.img_width, self.channels))  # ?
                        tmp = ((image_tensor + 1) * 127.5).astype(np.uint8)
                        res = self.get_mask(tmp, self.detector, self.predictor)
                        if res != None:
                            self.B_input[cnt_B] = image_tensor.reshape((self.batch_size, self.image_height, self.img_width, self.channels))
                            self.B_input_mask[cnt_B][0] = np.equal(res[0], 255)
                            self.B_input_mask[cnt_B][1] = np.equal(res[1], 255)
                            self.B_input_mask[cnt_B][2] = np.equal(res[2], 255)
                            cnt_B += 1

                os.mknod(load_dir)
                fw = open(load_dir, "wb")
                pickle.dump(self.A_input, fw)
                pickle.dump(self.B_input, fw)
                pickle.dump(self.A_input_mask, fw)
                pickle.dump(self.B_input_mask, fw)
                pickle.dump(cnt_A, fw)
                pickle.dump(cnt_B, fw)

            else:
                fr = open(load_dir, "rb")
                self.A_input = pickle.load(fr)
                self.B_input = pickle.load(fr)
                self.A_input_mask = pickle.load(fr)
                self.B_input_mask = pickle.load(fr)
                cnt_A = pickle.load(fr)
                cnt_B = pickle.load(fr)

            self.train_num = min(cnt_A, cnt_B)
            print("68 benchmark face number: ", self.train_num)

        coord.request_stop()
        coord.join(threads)


    def get_mask(self, input_face, detector, predictor,window=5):
        gray = cv2.cvtColor(input_face, cv2.COLOR_BGR2GRAY)
        dets = detector(gray, 1)

        for face in dets:
            shape = predictor(input_face, face)
            temp = []
            for pt in shape.parts():
                temp.append([pt.x, pt.y])
            lip_mask = np.zeros([256, 256])
            eye_mask = np.zeros([256,256])
            face_mask = np.full((256, 256), 255).astype(np.uint8)
            cv2.fillPoly(lip_mask, [np.array(temp[48:60]).reshape((-1, 1, 2))], (255, 255, 255))
            cv2.fillPoly(lip_mask, [np.array(temp[60:68]).reshape((-1, 1, 2))], (0, 0, 0))

            left_left = min(x[0] for x in temp[36:42])
            left_right = max(x[0] for x in temp[36:42])
            left_bottom = min(x[1] for x in temp[36:42])
            left_top = max(x[1] for x in temp[36:42])
            left_rectangle = np.array(
                [[left_left - window, left_top + window], [left_right + window, left_top + window],
                 [left_right + window, left_bottom - window], [left_left - window, left_bottom - window]]
            ).reshape((-1, 1, 2))
            cv2.fillPoly(eye_mask, [left_rectangle], (255, 255, 255))
            cv2.fillPoly(eye_mask, [np.array(temp[36:42]).reshape((-1, 1, 2))], (0, 0, 0))

            right_left = min(x[0] for x in temp[42:48])
            right_right = max(x[0] for x in temp[42:48])
            right_bottom = min(x[1] for x in temp[42:48])
            right_top = max(x[1] for x in temp[42:48])
            right_rectangle = np.array(
                [[right_left - window, right_top + window], [right_right + window, right_top + window],
                 [right_right + window, right_bottom - window], [right_left - window, right_bottom - window]]
            ).reshape((-1, 1, 2))
            cv2.fillPoly(eye_mask, [right_rectangle], (255, 255, 255))
            cv2.fillPoly(eye_mask, [np.array(temp[42:47]).reshape((-1, 1, 2))], (0, 0, 0))

            cv2.polylines(face_mask, [np.array(temp[17:22]).reshape(-1, 1, 2)], False, (0, 0, 0), 7)
            cv2.polylines(face_mask, [np.array(temp[22:27]).reshape(-1, 1, 2)], False, (0, 0, 0), 7)
            cv2.fillPoly(face_mask, [np.array(temp[36:42]).reshape((-1, 1, 2))], (0, 0, 0))
            cv2.fillPoly(face_mask, [np.array(temp[42:48]).reshape((-1, 1, 2))], (0, 0, 0))
            cv2.fillPoly(face_mask, [np.array(temp[48:60]).reshape((-1, 1, 2))], (0, 0, 0))
            return lip_mask,eye_mask,face_mask

    # def get_mask2(self, input_mask, scope='get_mask2'): #input_mask is np array
    #     ## lip
    #     if lips_ft is True:
    #         if not use_np:
    #             # mask_lip = tf.cast(tf.equal(input_mask, 7), tf.float32) + tf.cast(tf.equal(input_mask, 9), tf.float32)  #   lip regions are 1 others are 0    size : [256 256 1]
    #             # mask_lip index_lip = self.mask_preprocess2(input_mask)
    #         else:
    #             mask_lip = (input_mask == 7).astype('float32') + (input_mask == 9).astype('float32')
    #     ## skin
    #     if skin_ft is True:
    #         if not use_np:
    #             mask_skin = tf.cast(tf.equal(input_mask, 1), tf.float32) + tf.cast(tf.equal(input_mask, 6), tf.float32) + tf.cast(tf.equal(input_mask, 13), tf.float32)
    #             mask_skin, index_skin = self.mask_preprocess2(mask_skin)
    #         else:
    #             mask_skin = (input_mask == 1).astype('float32') + (input_mask == 6).astype('float32') + (input_mask == 13).astype('float32')
    #     ## eyes
    #     if eyes_ft is True:
    #         if not use_np:
    #             mask_eye_left = tf.cast(tf.equal(input_mask, 4), tf.float32)
    #             mask_eye_right = tf.cast(tf.equal(input_mask, 5), tf.float32)
    #             mask_face = tf.cast(tf.equal(input_mask, 1), tf.float32) + tf.cast(tf.equal(input_mask, 6), tf.float32)
    #         else:
    #             mask_eye_left = (input_mask == 4).astype('float32')
    #             mask_eye_right = (input_mask == 5).astype('float32')
    #             mask_face = (input_mask == 1).astype('float32') + (input_mask == 6).astype('float32')
    #     # avoid the situation that images with closed eyes
    #     if not (mask_eye_left > 0).any() and (mask_eye_right).any()


    def rebound_box(self, mask_A, mask_B, mask_A_face, scope='rebound_box'):
        with tf.variable_scope(scope, reuse=True):
            index_tmp = tf.where(tf.equal(mask_A, 1.0))  #256*256*1  --> ? * 3
            print("index_tmp : ", index_tmp)
            x_A_index = index_tmp[:, 1]  #2-D ? * 1
            y_A_index = index_tmp[:, 0]
            index_tmp = tf.where(tf.equal(mask_B, 1.0))
            x_B_index = index_tmp[:, 1]  # 256 * 1
            y_B_index = index_tmp[:, 0]
            print("rebound x_A_index : ", x_A_index)
            print("rebound y_A_index : ", y_A_index)
            print("rebound x_B_index : ", x_B_index)
            print("rebound y_B_index : ", y_B_index)

            mask_A_tmp = tf.Variable(mask_A, validate_shape=False)
            mask_B_tmp = tf.Variable(mask_B, validate_shape=False)
            min_xA_idx = tf.argmin(x_A_index) - 10
            max_xA_idx = tf.argmax(x_A_index) + 11
            min_yA_idx = tf.argmin(y_A_index) - 10
            max_yA_idx = tf.argmax(y_A_index) + 11
            min_xB_idx = tf.argmin(x_B_index) - 10
            max_xB_idx = tf.argmax(x_B_index) + 11
            min_yB_idx = tf.argmin(y_B_index) - 10
            max_yB_idx = tf.argmax(y_B_index) + 11
            #mA_tmp[min(x_A_index.eval())-10:max(x_A_index.eval())+11, min(y_A_index.eval())-10:max(y_A_index.eval())+11, :] = \
            #       mA_face_tmp[min(x_A_index.eval())-10:max(x_A_index.eval())+11, min(y_A_index.eval())-10:max(y_A_index.eval())+11, :]
            mask_A_tmp[min_xA_idx:max_xA_idx, min_yA_idx:max_yA_idx, :].assign(mask_A_face[min_xA_idx:max_xA_idx, min_yA_idx:max_yA_idx, :])
            print("mA_tmp", mask_A_tmp)
            #mB_tmp[min(x_B_index.eval())-10:max(x_B_index.eval())+11, min(y_B_index.eval())-10:max(y_B_index.eval())+11, :] = \
                    #mA_face_tmp[min(x_B_index.eval())-10:max(x_B_index.eval())+11, min(y_B_index.eval())-10:max(y_B_index.eval())+11, :]
            mask_B_tmp[min_xB_idx:max_xB_idx, min_yB_idx:max_yB_idx, :].assign(mask_A_face[min_xB_idx:max_xB_idx, min_yB_idx:max_yB_idx, :])
            print("mB_tmp", mask_B_tmp)

            # convert tensor to var
            #mask_A_tmp = sess.run(mask_A_tmp)
            #mask_B_tmp = sess.run(mask_B_tmp)
            return mask_A_tmp, mask_B_tmp


    def mask_preprocess(self, mask_A, mask_B):
        index_tmp = tf.where(tf.equal(mask_A, 1.0))  #
        x_A_index = index_tmp[:, 1]
        y_A_index = index_tmp[:, 0]
        index_tmp = tf.where(tf.equal(mask_B, 1.0))
        x_B_index = index_tmp[:, 1]
        y_B_index = index_tmp[:, 0]
        # re-convert float to bool
        mask_A = tf.cast(mask_A, tf.bool)
        mask_B = tf.cast(mask_B, tf.bool)

        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        return mask_A, mask_B, index, index_2

    def mask_preprocess2(self, mask):
        index_tmp = tf.where(tf.equal(mask, 1.0))  #
        x_index = index_tmp[:, 1]
        y_index = index_tmp[:, 0]
        # re-convert float to bool
        mask = tf.cast(mask_A, tf.bool)
        index = [x_index, y_index]
        return mask, index

    def get_keypoints(self, img, detector, predictor):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(img_gray, 0)
        res = np.zeros([68, 2])

        for num in range(len(dets)):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img, dets[num]).parts()])
            for idx, point in enumerate(landmarks):
                # 68 points coordinates
                res[idx][0] = point[0, 0]
                res[idx][1] = point[0, 1]
        return res


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')
