import tensorflow as tf
import numpy as np


def create_dataset(
    boards: np.ndarray[np.ndarray[np.int32]],
    moves: np.ndarray[np.int32],
) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices((boards, moves))


def test_model(dataset: tf.data.Dataset):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(16,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(4, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(dataset.batch(3), epochs=1)


test_boards = np.array(
    [
        np.zeros(16, dtype=np.int32),
        np.zeros(16, dtype=np.int32),
        np.zeros(16, dtype=np.int32),
    ]
)

test_moves = np.array([2, 3, 1], dtype=np.int32)

test_dataset = create_dataset(test_boards, test_moves)

test_model(test_dataset)

print("no boom")
