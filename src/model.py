import tensorflow as tf  # noqa: F401,E402

from keras.backend import clear_session  # noqa: E402
from keras.models import Sequential  # noqa: E402
from keras.layers import ReLU, LeakyReLU, PReLU, ELU, Activation  # noqa: F401,E402,E501
from keras.layers import Dense, InputLayer  # noqa: E402
from keras.layers import Dropout, BatchNormalization  # noqa: E402
from keras.callbacks import EarlyStopping, ReduceLROnPlateau  # noqa: E402
from keras.optimizers import Adam  # noqa: E402
from keras.regularizers import l2  # noqa: E402
from keras.metrics import AUC  # noqa: E402


# Callback Functions
def callbacks():
    # Early stopping
    early_stopping = EarlyStopping(
        start_from_epoch=25,
        monitor='val_loss',
        min_delta=0.001,
        patience=5
    )

    # Reduce Learning
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        min_lr=0.0001,
        factor=0.8,
        patience=2
    )

    return [early_stopping, reduce_lr]


def create_ann_model(hyprparameters, input_shape):
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

    # Hidden Layers
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

    model_history = model.fit(
        X_train_selected, y_train,
        validation_split=0.30,
        epochs=145, batch_size=35,
        callbacks=callbacks(),
        verbose=0
    )

    return model, model_history
