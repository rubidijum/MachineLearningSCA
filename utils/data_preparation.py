from dataclasses import dataclass
import tqdm
import keras
import tensorflow as tf
import os
import numpy as np
import h5py


class SCAML_Dataset():
    """ SCAAML dataset wrapper class
    """

    DEFAULT_TRACE_LENGTH = 80000

    @dataclass
    class ProfilingDataset:
        X: tf.Tensor
        y: tf.Tensor

    @dataclass
    class AttackDataset:
        X:          tf.Tensor
        y:          tf.Tensor
        keys:       tf.Tensor
        plaintexts: tf.Tensor

    def __init__(self) -> None:
        self.profiling_dataset = None
        self.attack_dataset = None

    def load_shards(self, path, num_shards=256):
        """! Load all of the shards found at path.
        """
        shard_array = []
        for shard_idx, shard_name in tqdm.tqdm(enumerate(os.listdir(path))):
            if shard_idx == num_shards:
                break
            shard_array.append(np.load(path + '/' + shard_name))

        return shard_array

    def create_dataset(self, data_path, attack_point, num_shards=256, trace_length=DEFAULT_TRACE_LENGTH, attack=False):
        """! Create dataset from the raw data and store it internaly

        @param data_path Path to raw data
        @param attack_point 'sub_bytes_in' or 'sub_bytes_out' ('keys' should not be used as models are behaving poorly with it)
        @param num_shards Number of shards to load into dataset
        @param trace_length Number of trace data points to use
        @param should_squeeze Indicates if the last dimension of the data should be discarded
        @param attack If attack data is created, return keys and plaintexts
        """

        shards = self.load_shards(data_path, num_shards)

        X = []
        y = []

        keys_list = []
        plaintexts_list = []

        for shard in tqdm.tqdm(shards, desc='Loading shards', position=0, leave=True):
            if attack:
                keys_list.append(tf.convert_to_tensor(shard['keys']))
                plaintexts_list.append(tf.convert_to_tensor(shard['pts']))

            X.append(tf.convert_to_tensor(
                shard['traces'][:, :trace_length, :], dtype='float32'))
            y.append(tf.convert_to_tensor(shard[attack_point]))

        X = tf.concat(X, axis=0)
        y = tf.concat(y, axis=1)

        if attack:
            keys_list = tf.concat(keys_list, axis=1)
            plaintexts_list = tf.concat(plaintexts_list, axis=1)

            self.attack_dataset = self.AttackDataset(
                X, y, keys_list, plaintexts_list)
        else:
            self.profiling_dataset = self.ProfilingDataset(X, y)

    def get_attack_dataset(self, shard_index, attack_byte, num_traces=256, trace_length=DEFAULT_TRACE_LENGTH):
        """! Get subset of the attack dataset based on shard index. Note that all keys in one shard are the same.


                attack_byte
                      |
                ______|_________________
                |     |
                |     |
                |     |
                .     |
                .     V
                .     _
   shard_index->|    | |            }
                |    | |            } num_traces
                |    |_|            }
                .
                .
                .

        @param shard_index Which shard within dataset is being attacked
        @param attack_byte Which byte within the key is being attacked.
        Note that this should match on which key byte the model was trained.
        @param num_traces Number of traces to attack in each shard
        @param trace_length Number of datapoints in each trace (resolution)
        """
        NUM_TRACES_PER_KEY = 256
        start_idx = shard_index * NUM_TRACES_PER_KEY
        end_idx = start_idx + num_traces

        y = self.attack_dataset.y[attack_byte]
        y = keras.utils.np_utils.to_categorical(
            y, num_classes=256, dtype='uint8')
        y = y[start_idx:end_idx, :]

        X = self.attack_dataset.X[start_idx:end_idx, :trace_length, :]
        return self.AttackDataset(X, y, self.attack_dataset.keys[:, start_idx:end_idx], self.attack_dataset.plaintexts[:, start_idx:end_idx])

    def get_profiling_dataset(self, attack_byte, trace_length=DEFAULT_TRACE_LENGTH):
        """! Get dataset of network inputs and outputs for specific attack point

        @param attack_byte Key byte on which profiling is performed
        @param trace_length Number of datapoints in each trace (resolution)
        """

        # Shuffle during training helps model convergence
        X = self.profiling_dataset.X
        y = keras.utils.np_utils.to_categorical(
            self.profiling_dataset.y[attack_byte], num_classes=256, dtype='uint8')
        indices = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        X = tf.gather(X, shuffled_indices)
        y = tf.gather(y, shuffled_indices)

        return self.ProfilingDataset(X[:, :trace_length, :], y)

    def get_correct_key(self, key_index):
        NUM_TRACES_PER_KEY = 256
        start_idx = key_index * NUM_TRACES_PER_KEY

        return np.array(self.attack_dataset.keys[:, start_idx])


class ASCAD_Dataset():

    DEFAULT_TRACE_LENGTH = 1400
    ATTACK_CHUNK_SIZE = 2000

    @dataclass
    class ProfilingDataset:
        X: tf.Tensor
        y: tf.Tensor

    @dataclass
    class AttackDataset:
        X:          tf.Tensor
        y:          tf.Tensor
        keys:       tf.Tensor
        plaintexts: tf.Tensor

    def __init__(self) -> None:

        self.ascad_path = "./data/SCA_datasets/datasets/ASCAD/ASCAD.h5"

        self.h5File = None
        self.profiling_data = None
        self.attack_data = None

    def load_data(self):

        self.h5File = h5py.File(self.ascad_path, "r")

        self.profiling_data = self.h5File.get('Profiling_traces')
        self.attack_data = self.h5File.get('Attack_traces')

        print(self.profiling_data.get('traces'))
        print(self.profiling_data.get('metadata'))

    def create_dataset(self, data_path, attack_point, num_shards=256, trace_length=DEFAULT_TRACE_LENGTH, attack=False):
        with h5py.File(data_path, 'r') as h5f:
            profiling_group = h5f.get('Profiling_traces')
            attack_group = h5f.get('Attack_traces')

    def get_attack_dataset(self, shard_index, attack_byte=3, num_traces=ATTACK_CHUNK_SIZE, trace_length=DEFAULT_TRACE_LENGTH):

        start_idx = shard_index * ATTACK_CHUNK_SIZE
        end_idx = start_idx + num_traces

        X = tf.expand_dims(self.attack_data['traces'][:, :trace_length], 1)
        y = keras.utils.np_utils.to_categorical(
            self.attack_data['labels'][:], num_classes=256, dtype='uint8')

        keys = self.attack_data['']

    def get_profiling_dataset(self, attack_byte=3, trace_length=DEFAULT_TRACE_LENGTH):

        X = self.profiling_data['traces'][:, :trace_length]
        X = tf.convert_to_tensor(X)
        X = tf.expand_dims(X, -1)

        y = self.profiling_data['labels'][:]
        y = keras.utils.np_utils.to_categorical(
            y, num_classes=256, dtype='uint8')
        y = tf.convert_to_tensor(y)

        return self.ProfilingDataset(X[:, :trace_length, :], y)
