from datetime import datetime
import tensorflow as tf

def train_model(model, X_train, y_train, batch_size, epochs, validation_data, callbacks_list=None, nn_type='MLP', tag='', save_dir=''):
    """! Train model and log training process to tensorboard

    @param model Model to train
    @param X_train Model inputs
    @param y_train Correct labels
    @param batch_size Size of the input batch
    @param epochs Number of epochs to train
    @param validation_data Validation data tuple in the format (X_val, y_val)
    @param callbacks_list List of tf callbacks to use during training, default are EarlyStopping and Tensorboard
    @param nn_type Type of the neural net used for logging
    @param tag Additional tag regarding training details, default is none
    @save_dir Path to save model to, default is './models
    """

    _date = datetime.now().strftime("%Y-%d-%m_%H-%M")
    _save_dir = './models/' + save_dir + '_' + _date + '[' + tag + ']' 
    _log_dir =  './logs/' + nn_type + '_' + _date + '[' + tag + ']'
    
    if callbacks_list is None:
        callbacks_list = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3),
        tf.keras.callbacks.TensorBoard(log_dir=_log_dir, histogram_freq=1)
    ]

    history = model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=validation_data, callbacks=callbacks_list)
    model.save(_save_dir)
    
    return history