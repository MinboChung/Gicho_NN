import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout


# CNN classifier
class MyCNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # Convolutional Layers
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

        # Add dense layers
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
        # Convert target labels to one-hot encoding
        y_train_one_hot = keras.utils.to_categorical(y_train, self.num_classes)
        y_val_one_hot = keras.utils.to_categorical(y_val, self.num_classes)

        # Train the model
        self.model.fit(X_train, y_train_one_hot, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val_one_hot))

    def predict(self, X_test):
        # Get predictions
        predictions = self.model.predict(X_test)
        return np.argmax(predictions, axis=1)



if __name__=="__main__":
    ...