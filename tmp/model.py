from src.config import hyprparameter_ranges, callbacks

from keras.backend import clear_session
from keras.models import Sequential

from keras.layers import Activation, ReLU, LeakyReLU, PReLU, ELU
from keras.layers import Dense, Dropout, BatchNormalization, InputLayer

from keras.optimizers import Adam
from keras.regularizers import l2
from keras.metrics import AUC


# Model builder
def create_ann_model(hyprparameters, fit_model, input_shape, X_train_selected, y_train):
    layer_keys = []

    for key in hyprparameter_ranges:
        if key.startswith('l') and key[1:].isdigit():
            layer_keys.append(key)

    hidden_layers = len(layer_keys)

    clear_session()
    layer_units = hyprparameters[:hidden_layers]
    lr, dr, l2_reg, alpha = hyprparameters[hidden_layers:]

    model = Sequential()
    model.add(InputLayer(shape=(input_shape,)))

    for neurons in layer_units:
        model.add(Dense(int(neurons), kernel_regularizer=l2(l2_reg)))
        model.add(ReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dr))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )

    if fit_model:
        model.fit(
            X_train_selected, y_train,
            validation_split=0.30,
            epochs=145, batch_size=35,
            callbacks=callbacks(),
            verbose=0
        )

    return model
