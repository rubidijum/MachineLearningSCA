from datetime import datetime
from tabnanny import verbose
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm.auto import tqdm
from utils.AES import AES
from utils.data_preparation import SCAML_Dataset
class SCA_Trainer():

    def __init__(self, log_root_path='./logs/', save_root_path='./models/') -> None:
        self.log_root_path = log_root_path
        self.save_root_path = save_root_path

        self.y_predicted_list = []
        self.y_true_list = []

    def train_model(self, model, X_train, y_train, batch_size, epochs, validation_split, callbacks_list=None, nn_type='MLP', tag='', save_dir=''):
        """! Train model and log training process to tensorboard

        @param model Model to train
        @param X_train Model inputs
        @param y_train Correct labels
        @param batch_size Size of the input batch
        @param epochs Number of epochs to train
        @param validation_split Percentage of validation data from training data
        @param callbacks_list List of tf callbacks to use during training, default are EarlyStopping and Tensorboard
        @param nn_type Type of the neural net used for logging
        @param tag Additional tag regarding training details, default is none
        @save_dir Path to save model to, default is './models
        """

        _time = datetime.now().strftime("%Y-%d-%m_%H-%M")

        _log_dir = self.log_root_path + nn_type + _time + tag

        checkpoint_filepath = './checkpoints/' + nn_type + '_' + _time

        if callbacks_list is None:
            callbacks_list = [
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3),
                tf.keras.callbacks.TensorBoard(log_dir=_log_dir, histogram_freq=1),
                tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                save_weights_only=True,
                                                monitor='val_accuracy',
                                                mode='max',
                                                save_best_only=True)
            ]

        history = model.fit(X_train, 
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,verbose=1,
                            validation_split=validation_split,
                            callbacks=callbacks_list)
        
        model.save(self.save_root_path)
        
        return history

    def evaluate_model(self, model, dataset, attack_byte, traces_per_chunk=256, keys_to_attack=256):
        """! Evaluate model quality based on SCA relevant metrics.
        This is equivalent to attacking single key byte and displaying the results.
        
        @keys Matrix of keys () s
        @plaintexts Matrix of plaintexts
        @key_byte Key byte index to attack (0, 255)
        @param keys_per_chunk Number of traces in each attack chunk. All the data in one attack chunk is regarding the same key.
        """

        for key_index in tqdm(range(keys_to_attack)):
            attack_dataset = dataset.get_dataset(key_index, attack_byte, num_traces=traces_per_chunk, training=False)
            X_attack, y_attack, keys, plaintexts = attack_dataset.X, attack_dataset.y, attack_dataset.keys, attack_dataset.plaintexts

            print(X_attack.shape)

            true_key_byte = keys[0][0]
            print(f"Attacking key byte: {true_key_byte}")

            predictions = model.predict(X_attack, verbose=1)

            # Intermediate value classes
            # predicted_y_categorical = np.argmax(predictions, axis=1)

            # true_y = y_attack
            # true_y_categorical = np.argmax(true_y, axis=1)

            # self.y_predicted_list.append(predicted_y_categorical)
            # self.y_true_list.append(true_y_categorical)

            
            key_predicted_probabilities = self.get_key_probabilities(predictions, plaintexts, attack_byte)

            # Accumulate key predictions and classify
            probs = np.zeros(256)
            for i, p in enumerate(key_predicted_probabilities):
                probs += p
                rankings = np.argsort(probs)[::-1]
                print(f"Key guess {rankings[0]} using {i} traces")


        # for trace_num in tqdm(range(attack_traces)):
        #     # Predict only on the portion of data tied to one key
        #     min_ = trace_num*traces_per_chunk
        #     max_ = min_ + attack_traces # + traces_per_attack -> sa koliko trace-va napadamo ovaj konkretan kljuc

        #     plaintexts_ = plaintexts[min_:max_]
        #     true_key_byte = keys[min_:max_][trace_num]

        #     predictions = model.predict(X_attack[min_:max_,:,:], verbose=1)
        #     # print(predictions.shape)

        #     key_predicted_probabilities = get_key_probabilities

        #     print(f"Predicted key {key_predicted} with probability {key_predicted_probabilities[key_predicted]}")

        #     print(f"Predicted probs {key_predicted_probabilities}")
        #     # Intermediate value classes
        #     predicted_y_categorical = np.argmax(predictions, axis=1)

        #     true_y = y_attack[min_:max_,:]
        #     true_y_categorical = np.argmax(true_y, axis=1)

        #     self.y_predicted_list.append(predicted_y_categorical)
        #     self.y_true_list.append(true_y_categorical)

        #     # Voting
        #     values = np.zeros((256))
        #     for i in range(num_traces):
        #         values = values + key_predicted_probabilities
        #         print(f"values {values}, shape = {values.shape}")
        #         print(f"Argmaxed: {np.argmax(values)[::-1]}, correct_key: {true_key_byte}")

        # conf_pred = np.asarray(self.y_predicted_list).flatten()
        # conf_true = np.asarray(self.y_true_list).flatten()

        # conf_mat = tf.math.confusion_matrix(conf_pred, conf_true)
        # plt.imshow(conf_mat, interpolation='none', cmap='magma')
        # plt.colorbar()
        # plt.title('Confusion matrix')
        # plt.xlabel('Predicted attack points')
        # plt.ylabel('True attack points')
        # plt.grid(True)
        # plt.show()


    def get_key_probabilities(self, predictions, plaintexts, attack_byte):
        """! Calculate probabilities for each key guess based on the NN outputs

        @predictions NN predicted values for intermediate values
        @plaintexts Matrix of plaintexts
        @attack_byte Index of the byte to attack
        """
        aes = AES()

        xs = predictions.shape[0]
        #                                   num_classes = 256 for key bytes
        key_probabilities = np.zeros((xs, 256))
        for i, predictions_test in enumerate(predictions):
            pt_test = plaintexts[attack_byte][i]
            for j, prediction_value in enumerate(predictions_test):
                sbox_inverse = aes.reverse_SBOX(j)
                predicted_key = sbox_inverse ^ pt_test
                key_probabilities[i][predicted_key] = prediction_value
        
        return np.array(key_probabilities)

    def plot_confusion_matrix(self):
        """! Plot confusion matrix of the evaluated model
        """
        conf_pred = np.asarray(self.y_predicted_list).flatten()
        conf_true = np.asarray(self.y_true_list).flatten()
        conf_mat = tf.math.confusion_matrix(conf_pred, conf_true)
        plt.imshow(conf_mat, interpolation='none', cmap='magma')
        plt.colorbar()
        plt.title('Confusion matrix')
        plt.xlabel('Predicted attack points')
        plt.ylabel('True attack points')
        plt.grid(True)
        plt.show()

    def evaluation_summary(self):
        """! Display summary of model quality evaluation
        """
        pass

        