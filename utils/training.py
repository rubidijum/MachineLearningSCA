from datetime import datetime
from tabnanny import verbose
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm.auto import tqdm
from utils.AES import AES
from utils.data_preparation import SCAML_Dataset

import time


class SCA_Trainer():
    # TODO: tie trainer to a single model => add as constructor parameter
    # TODO: indicate if already trained
    def __init__(self, log_root_path='./logs/', save_root_path='./models/') -> None:
        self.log_root_path = log_root_path
        self.save_root_path = save_root_path

        self.y_predicted_list = []
        self.y_true_list = []

        self.key_ranks = []

        self.correct_key_predictions = []

        self.stats_per_trace = {}

        self.training_history = None

        self.area_under_curve = 0.0

        self.ranks_history = dict.fromkeys(list(range(0, 16)), [])

    def train_model(self, model, dataset, attack_byte, batch_size, epochs, validation_split=0.0, validation_data=None, callbacks_list=None, tag='', save_dir='', verbose=1):
        """! Model training wrapper function.

        Train the model, plot training data on Tensorboard and save trained model to specified path.

        @param model Model to train
        @param X_train Model inputs
        @param y_train Correct labels
        @param batch_size Size of the input batch
        @param epochs Number of epochs to train
        @param validation_split Percentage of data from training set to use for validation
        @param callbacks_list List of tf callbacks to use during training, default are EarlyStopping and Tensorboard
        @param tag Additional info regarding training details, default is none
        @save_dir Path to save model to, default is './models
        """
        assert (validation_split == 0.0 or validation_data is None)
        assert (not (validation_data is None and validation_split == 0.0))

        _time = datetime.now().strftime("%Y-%d-%m_%H-%M")

        _log_dir = self.log_root_path + model.name + _time + tag

        checkpoint_filepath = './checkpoints/' + model.name + '_' + _time

        if callbacks_list is None:
            callbacks_list = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=4, verbose=1),
                tf.keras.callbacks.TensorBoard(
                    log_dir=_log_dir, histogram_freq=1),
                tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                   save_weights_only=True,
                                                   monitor='val_accuracy',
                                                   mode='max',
                                                   save_best_only=True)
            ]

        train_dataset = dataset.get_profiling_dataset(attack_byte)
        X_train = train_dataset.X
        y_train = train_dataset.y

        history = model.fit(X_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            validation_split=validation_split,
                            validation_data=validation_data,
                            callbacks=callbacks_list)

        save_path = (self.save_root_path + model.name +
                     tag) if save_dir == '' else save_dir
        model.save(save_path)

        self.training_history = history

    def get_model_history(self):
        """! Return model training history
        """
        return self.training_history

    def plot_model_history(self):
        """! Plot training and validation accuracy and loss
        """
        plt.figure(figsize=(20, 10))

        plt.subplot(121)

        random_accuracy = [1/256] * \
            len(self.training_history.history['val_accuracy'])

        plt.title(f"Accuracy")
        plt.ylabel("Number of epochs")
        plt.plot(self.training_history.history['val_accuracy'])
        plt.plot(self.training_history.history['accuracy'])
        plt.plot(random_accuracy)
        plt.legend(['val_acc', 'train_acc', 'random_acc'], loc='best')

        plt.subplot(122)

        plt.title(f"Loss function")
        plt.ylabel("Number of epochs")
        plt.plot(self.training_history.history['val_loss'])
        plt.plot(self.training_history.history['loss'])
        plt.legend(['val_loss', 'training_loss'], loc='best')

        plt.show()

    def evaluate_model(self, model, dataset, attack_byte, traces_per_chunk=256, keys_to_attack=256, verbose=1):
        """! Evaluate model quality based on SCA relevant metrics.
        This is equivalent to attacking single key byte (of possibly many different keys).
        Accuracy and key rank metrics are calculated during evaluation.

        @param model Trained Keras model object to evaluate
        @param dataset Dataset on which to perform evaluation
        @param attack_byte Key byte index to attack (0 - 15)
        @param traces_per_chunk Number of traces in each attack chunk. All the data in one attack chunk is regarding the same key.
        @param keys_to_attack Number of different keys to attack. Equivalent to number of shards to attack.
        @param verbose Print extra information during evaluation if 1. 1 by default
        """

        # reset calculated metrics for each new evaluation run
        self.key_ranks = []
        self.y_predicted_list = []
        self.y_true_list = []
        self.stats_per_trace = {}

        with tqdm(total=keys_to_attack) as pbar:
            for key_index in range(keys_to_attack):
                attack_dataset = dataset.get_attack_dataset(
                    key_index, attack_byte, num_traces=traces_per_chunk)
                X_attack, y_attack, keys, plaintexts = attack_dataset.X, attack_dataset.y, attack_dataset.keys, attack_dataset.plaintexts

                true_key_byte = keys[attack_byte][0]

                predictions = model.predict(X_attack, verbose=verbose)

                # Intermediate value classes accuracy
                predicted_y_categorical = np.argmax(predictions, axis=1)
                true_y_categorical = np.argmax(y_attack, axis=1)

                self.y_predicted_list.append(predicted_y_categorical)
                self.y_true_list.append(true_y_categorical)

                key_predicted_probabilities = self.get_key_probabilities(
                    predictions, plaintexts, attack_byte)

                # Accumulate key predictions and classify
                probs = np.zeros(256)
                ranks = []
                for i, p in enumerate(key_predicted_probabilities):
                    probs += p
                    rankings = np.argsort(probs)[::-1]
                    true_key_byte_rank = np.where(
                        rankings == true_key_byte)[0][0]

                    ranks.append(true_key_byte_rank)

                self.key_ranks.append(ranks)

                pbar.set_description(
                    f"True key byte: {true_key_byte}, predicted: {rankings[0]}")
                pbar.update()

        # num_traces -> (correctPreds_no, % of keys guessed)
        for num_traces in range(traces_per_chunk):
            # Correct key predictions for num_traces
            _ranks_col = np.asarray(self.key_ranks)[:, num_traces]
            _num_correct = np.sum(_ranks_col == 0, axis=0)
            self.stats_per_trace[num_traces] = (
                _num_correct, (_num_correct/keys_to_attack)*100)

        # Scale to 0-1 and integrate to get area under curve
        scaled_y = np.array([x[1] for x in self.stats_per_trace.values()])/100
        step = 1/len(self.stats_per_trace.values())
        scaled_x = np.arange(0, 1, step)
        self.area_under_curve = np.trapz(scaled_y, scaled_x)

        self.plot_key_ranks(self.key_ranks)
        self.plot_confusion_matrix()

    def get_key_probabilities(self, predictions, plaintexts, attack_byte):
        """! Calculate probabilities for each key guess based on the NN outputs

        @predictions NN predicted values for intermediate values
        @plaintexts Matrix of plaintexts
        @attack_byte Index of the byte to attack
        """
        aes = AES()

        xs = predictions.shape[0]
        #                                 num_classes = 256 for key bytes
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

    def plot_key_ranks(self, ranks, num_traces=None):
        """! Plot key ranks of the evaluated model
        """

        plt.figure(figsize=(20, 20))

        for r in ranks:
            plt.plot(r)

        x_max = len(ranks[0]) if num_traces is None else num_traces
        x_labels = [str(x) for x in range(x_max)]
        plt.xticks(np.arange(x_max), labels=x_labels, rotation=90)

        y_max = np.max(np.asarray(ranks))
        y_labels = [str(y) for y in range(y_max)]
        plt.yticks(np.arange(y_max), labels=y_labels, rotation='horizontal')

        plt.ylabel('Key rank')
        plt.xlabel('Number of traces')
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def evaluation_summary(self):
        """! Display summary of model quality evaluation
        """

        # Average model accuracy for predicting intermediate values
        num_keys = len(self.y_predicted_list)
        num_traces = len(self.y_predicted_list[0])
        avg_accuracy = np.sum(np.asarray(self.y_true_list)
                              == np.asarray(self.y_predicted_list)) / num_keys

        print(
            f"Model achieved average accuracy of {avg_accuracy}% on {num_keys} different keys ({num_traces} traces each)")

        # Print maximum recovery accuracy with x traces
        max_accuracy = max(self.stats_per_trace.values())[1]
        min_traces = max(self.stats_per_trace, key=self.stats_per_trace.get)
        print(
            f"Maximum key recovery success of {max_accuracy} achieved with {min_traces} traces")

        print(f"Area under curve score: {self.area_under_curve}")

        plt.xlabel('Number of traces')
        plt.ylabel('%% of the successful key byte guesses')

        plt.plot([x[1] for x in self.stats_per_trace.values()])
        plt.show()

    def attack_full_key(self, models, trainers, dataset, key_index, traces_per_chunk=256, verbose=1):
        """! Perform full key recovery.

        @param models List of pretrained models where models[0] was trained on the first key byte,
        models[1] was trained on the second key byte and so on.
        @param trainers List of SCA_Trainer objects used to train models or None if  trainers should be created here
        @param dataset Dataset from which attack shards are drawn
        @param key_index Index of the shard to attack
        @param traces_per_chunk Number of traces to be used in an attack
        @param verbose Toggle verbose output
        """

        true_key = dataset.get_correct_key(key_index)
        assert (len(models) == len(true_key))

        # Reset ranks history
        self.ranks_history = dict.fromkeys(list(range(0, 16)), [])

        recovered_key = []

        with tqdm(total=16, position=1, leave=False) as pbar:
            for attack_byte, model in enumerate(models):
                attack_dataset = dataset.get_attack_dataset(
                    key_index, attack_byte, num_traces=traces_per_chunk)
                X_attack, _, _, plaintexts = attack_dataset.X, attack_dataset.y, attack_dataset.keys, attack_dataset.plaintexts

                predictions = model.predict(X_attack, verbose=0)

                key_predicted_probabilities = self.get_key_probabilities(
                    predictions, plaintexts, attack_byte)

                # Accumulate key predictions and classify
                probs = np.zeros(256)
                ranks = []
                for i, p in enumerate(key_predicted_probabilities):
                    probs += p
                    rankings = np.argsort(probs)[::-1]
                    ranks.append((i, rankings))

                self.ranks_history[attack_byte] = ranks

                recovered_key.append(rankings[0])

                pbar.set_description(
                    f"True key: {true_key}\npredicted: {recovered_key}\ntraces used: {traces_per_chunk}")
                pbar.update()

        if verbose == 1:
            print(
                f"Recovered {np.sum(np.asarray(true_key)==np.asarray(recovered_key))/len(true_key)*100}% of the key using {traces_per_chunk} traces")
        return np.array(recovered_key)

    def evaluate_attack(self, models, dataset, keys_to_atack, max_traces=15):
        """ Perform multiple attacks using trained models and calculate
        global metrics.
        """
        trials = 0
        guesses = []
        correct_guesses = dict.fromkeys(list(range(1, max_traces+1)), 0)

        with tqdm(total=keys_to_atack, position=0, leave=True) as pbar:
            for key_index in range(keys_to_atack):
                pbar.set_description(f"Attacking key in shard {key_index}")
                for trace_no in range(1, max_traces+1):
                    key_guess = self.attack_full_key(
                        models, None, dataset, key_index, traces_per_chunk=trace_no)
                    guesses.append((trace_no, key_guess))
                    correct_key = dataset.get_correct_key(key_index)
                    if (key_guess == correct_key).all():
                        if trace_no in correct_guesses.keys():
                            correct_guesses[trace_no] += 1
                        else:
                            correct_guesses[trace_no] = 1
                        # Stop attacking when correct key is guessed
                        for t in range(trace_no+1, max_traces+1):
                            correct_guesses[t] += 1
                        break
                pbar.update()

        _m = {key: value for (key, value)
              in correct_guesses.items() if value > 0}

        if (len(_m) > 0):
            # Minimum number of traces that broke at least one key
            min_traces = min(_m.keys())
            print(f"Min traces {min_traces}")
            print(f"Keys recovered: {(_m[min_traces]/keys_to_atack)*100} %")

            # Maximum number of traces that broke most keys
            for key, val in _m.items():
                if val == max(_m.values()):
                    print(f"Max traces {key}")
                    print(f"Keys recovered: {(_m[key]/keys_to_atack)*100} %")
                    break
        else:
            print(
                f"Model failed to recover any keys using [1-{max_traces}] traces.")

        return guesses
