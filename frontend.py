from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
import keras.backend as K
import numpy as np
import os
import cv2
from utils import decode_netout, compute_overlap, compute_ap
from keras.applications.mobilenet import MobileNet
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from keras.initializers import glorot_uniform
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from backend import ResNet50Feature, MobileNetFeature
from preprocessing import parse_annotation, normalize

class keypoints(object):
    def __init__(self, backend,
                       input_size, 
                       labels,
                       classes=14):
        
        self.X_predicted = [None] * 7
        self.Y_predicted = [None] * 7
        self.X_groundtruth = [None] * 7
        self.Y_groundtruth = [None] * 7
        
        self.input_size = input_size
        
        self.labels   = list(labels)
        self.nb_class = len(self.labels)

        self.class_wt = np.ones(self.nb_class, dtype='float32')

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        input_image     = Input(shape=(self.input_size, self.input_size, 3))
        # self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))  

        if backend == 'ResNet50':
            self.feature_extractor = ResNet50Feature(self.input_size)
        else:
            raise Exception('Architecture not supported! Only support ResNet50 at the moment!')

        print(self.feature_extractor.get_output_shape())    
        self.grid_h, self.grid_w = self.feature_extractor.get_output_shape()        
        features = self.feature_extractor.extract(input_image)            

        # make the keypoint detection layer

        output = Flatten()(features)
        output = Dense(14, activation='relu', name='fc', kernel_initializer = 'normal')(output)

        self.model = Model(input_image, output)

        
        # initialize the weights of the detection layer
        

        # print a summary of the whole model
        self.model.summary()


    def delta_(self, p1, p2, x, y):
        return K.sqrt((x[p1]-x[p1])**2 + (y[p2]-y[p2])**2)

    def cross_ratio(self, surf, x, y):
        if surf == "left":
            return (self.delta_(0, 2, x, y)/self.delta_(1, 3, x, y)) / (self.delta_(1, 2, x, y)/self.delta_(1, 3, x, y))
        elif surf == "right":
            return (self.delta_(0, 5, x, y)/self.delta_(0, 6, x, y)) / (self.delta_(4, 5, x, y)/self.delta_(4, 6, x, y))
    

    def custom_loss(self, groundtruth, pridection):      # ETH custom loss
        segma = 0.0001  # Cross ratio controlling factor
        Cr3D = 1.39408  # The 3D cross ratio of the cone
        loss = 0
        
        for i in range(0, 14, 2):
            self.X_predicted[int(i/2)] = pridection[i]
            self.X_groundtruth[int(i/2)] = groundtruth[i]

        for j in range(1, 14, 2):
            self.Y_predicted[int(j/2)] = pridection[j]
            self.Y_groundtruth[int(j/2)] = groundtruth[j]
        

        for k in range(0,7):
            loss += (self.X_predicted[k]-self.X_groundtruth[k])**2 + (self.Y_predicted[k]-self.Y_groundtruth[k])**2 + \
                segma*(self.cross_ratio("left", self.X_predicted, self.Y_predicted)-Cr3D)**2 + segma*(self.cross_ratio("right", self.X_predicted, self.Y_predicted)-Cr3D)**2

        return loss

    def train(self, train_imgs,     # the list of images to train the model
                    train_times,    # the number of time to repeat the training set, often used for small datasets
                    nb_epochs,      # number of epoches
                    learning_rate,  # the learning rate
                    batch_size,     # the size of the batch
                    saved_weights_name='best_weights.h5',
                    debug=False):     

        self.batch_size = batch_size

        self.debug = debug

        ############################################
        # Make train generators
        ############################################  

        X_train_orig ,Y_train = parse_annotation()

        X_train = normalize(X_train_orig)


        ############################################
        # Compile the model
        ############################################

        sgd = SGD(lr=0.0001, momentum=0.9)  
        self.model.compile(optimizer = sgd, loss = self.custom_loss, metrics = ['accuracy'])

        ############################################
        # Make a few callbacks
        ############################################

        early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)
        checkpoint = ModelCheckpoint(saved_weights_name, 
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='min', 
                                     period=1)
        tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/'), 
                                  histogram_freq=0, 
                                  #write_batch_performance=True,
                                  write_graph=True, 
                                  write_images=False)

        ############################################
        # Start the training process
        ############################################        

        self.model.fit(np.array(X_train), np.array(Y_train), 
                        epochs = nb_epochs, 
                        batch_size = self.batch_size, 
                        #callbacks = [early_stop, checkpoint, tensorboard], 
                        workers = 3,
                        max_queue_size = 8)      

        ############################################
        # Compute mAP on the validation set
        ############################################

        # average_precisions = self.evaluate(valid_generator)     

        # # print evaluation
        # for label, average_precision in average_precisions.items():
        #     print(self.labels[label], '{:.4f}'.format(average_precision))
        # print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))

    def evaluate(self, 
                 generator, 
                 iou_threshold=0.3,
                 score_threshold=0.3,
                 max_detections=100,
                 save_path=None):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
        """    
        # gather all detections and annotations
        all_detections     = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
        all_annotations    = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes  = self.predict(raw_image)

            
            score = np.array([box.score for box in pred_boxes])
            pred_labels = np.array([box.label for box in pred_boxes])        
            
            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])  
            
            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes  = pred_boxes[score_sort]
            
            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]
                
            annotations = generator.load_annotation(i)
            
            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
                
        # compute mAP by comparing all detections and all annotations
        average_precisions = {}
        
        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]
                num_annotations     += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = compute_ap(recall, precision)  
            average_precisions[label] = average_precision

        return average_precisions

    def predict(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = self.feature_extractor.normalize(image)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))

        netout = self.model.predict([input_image, dummy_array])[0]
        

        return boxes