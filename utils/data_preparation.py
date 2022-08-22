import tqdm
import keras
import tensorflow as tf
import os
import numpy as np
import h5py

class SCAML_Dataset():

    DEFAULT_TRACE_LENGTH = 80000

    def load_shards(self, path, num_shards=256):
        shard_array = []
        for shard_idx, shard_name in tqdm.tqdm(enumerate(os.listdir(path))):
            if shard_idx == num_shards:
                break
            shard_array.append(np.load(path + '/' + shard_name))

        return shard_array

    def create_data(self, shard_array, attack_point, attack_byte, trace_length=DEFAULT_TRACE_LENGTH, attack=False):
        """! Extract data from the shards based on attack point and attack byte
            
        @param shard_array Array of loaded .npz shards
        @param attack_point 'sub_bytes_in' or 'sub_bytes_out' ('keys' should not be used as models are behaving poorly with it)
        @param attack_byte Index of the key byte to attack
        @param trace_length Number of trace data points to use
        @param should_squeeze Indicates if the last dimension of the data should be discarded
        @param attack If attack data is created, return keys and plaintexts
        @return Tuple of network inputs and outputs for specific attack point and key byte 
        """
    
        X = []
        y = []
        
        keys = []
        plaintexts = []
        
        for shard in tqdm.tqdm(shard_array, desc='Loading shards', position=0, leave=True):
            if attack:
                keys.append(tf.convert_to_tensor(shard['keys'][attack_byte,:]))
                plaintexts.append(tf.convert_to_tensor(shard['pts'][attack_byte,:]))
                
            X.append(tf.convert_to_tensor(shard['traces'][:,:trace_length,:], dtype='float32'))
            
            y_ = shard[attack_point][attack_byte]
            y_ = keras.utils.np_utils.to_categorical(y_, num_classes=256, dtype='uint8')
            
            y.append(tf.convert_to_tensor(y_))
            
        X = tf.concat(X, axis=0)
        y = tf.concat(y, axis=0)

        indices = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        X = tf.gather(X, shuffled_indices)
        y = tf.gather(y, shuffled_indices)
            
        if attack:
            keys = tf.concat(keys, axis=0)
            plaintexts = tf.concat(plaintexts, axis=0)
            
            keys = tf.gather(keys, shuffled_indices)
            plaintexts = tf.gather(plaintexts, shuffled_indices)
            
            return (X, y, keys, plaintexts)
        else:
            return (X, y)

class ASCAD_Dataset():

    DEFAULT_TRACE_LEN = 700

    def __init__(self) -> None:

        self.ascad_path = "drive/MyDrive/ASCAD_data/ASCAD_databases/ASCAD.h5"

        self.h5File = None
        self.profiling_data = None
        self.attack_data = None

    def load_data(self):

        self.h5File = h5py.File(self.ascad_path, "r")

        self.profiling_data = self.h5File.get('Profiling_traces')
        self.attack_data = self.h5File.get('Attack_traces')

    def create_data(self, training, num_traces=None, trace_length=DEFAULT_TRACE_LEN, return_metadata=True ):
        
        traces = self.profiling_data.get('traces') if training else self.attack_data.get('traces')
        labels = self.profiling_data.get('labels') if training else self.attack_data.get('labels')
        metadata = self.profiling_data.get('metadata') if training else self.attack_data.get('metadata')

        if num_traces is None:
                num_traces = traces.shape[0]

        X = traces[:num_traces, :trace_length]
        y = labels[:num_traces]
        
        if return_metadata != True:
            metadata = None

        return ((X,y), metadata)