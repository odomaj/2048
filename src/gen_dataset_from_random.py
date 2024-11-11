from env.game_logic import Board, Move, MoveResult
from env.game_environment import GameEnvironment
import random
import numpy as np
import time
import copy
import typing

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import json
import pprint
import tensorflow as tf
import matplotlib.pyplot as plt

def tempHankelReward(board: np.array):
    hankel_matrix = np.array(
            [
                1,
                1 / 2,
                1 / 4,
                1 / 8,
                1 / 2,
                1 / 4,
                1 / 8,
                1 / 16,
                1 / 4,
                1 / 8,
                1 / 16,
                1 / 32,
                1 / 8,
                1 / 16,
                1 / 32,
                1 / 64,
            ]
        )
    # Calculate the reward as the dot product of the board and Hankel matrix
    return np.dot(board, hankel_matrix)

def getBestMove(currentEnvironment: GameEnvironment) -> tuple((int,float)):
    moves = []
    upMove = copy.deepcopy(currentEnvironment)
    downMove = copy.deepcopy(currentEnvironment)
    leftMove = copy.deepcopy(currentEnvironment)
    rightMove = copy.deepcopy(currentEnvironment)
    if upMove.board.move(Move.UP):
        moves.append((Move.UP,upMove.hankel_reward()))
    if downMove.board.move(Move.DOWN):
        moves.append((Move.DOWN,downMove.hankel_reward()))
    if leftMove.board.move(Move.LEFT):
        moves.append((Move.LEFT,leftMove.hankel_reward()))
    if rightMove.board.move(Move.RIGHT):
        moves.append((Move.RIGHT,rightMove.hankel_reward()))
    bestMove = (-1,-1)
    for move in moves:
        if move[1] > bestMove[1]:
            bestMove = move
    return bestMove[0], bestMove[1], currentEnvironment #, currentEnvironment.hankel_reward()


random.seed(time.time_ns())

movesStored = []

iterations = 5000
for i in range(1,iterations + 1):
    n_environment = GameEnvironment()
    for j, space in enumerate(n_environment.board.board):
        num = random.randint(-10,10)
        n_environment.board.board[j] = num if num > -1 else 0
    print(n_environment.board.board)
    movesStored.append(getBestMove(n_environment))

print(movesStored)

# # Create a dictionary with features that may be relevant.
# def image_example(image_string, label):
#   image_shape = tf.io.decode_jpeg(image_string).shape

#   feature = {
#       'height': _int64_feature(image_shape[0]),
#       'width': _int64_feature(image_shape[1]),
#       'depth': _int64_feature(image_shape[2]),
#       'label': _int64_feature(label),
#       'image_raw': _bytes_feature(image_string),
#   }

#   return tf.train.Example(features=tf.train.Features(feature=feature))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def example_2048(move: int, score, environment):
    print("****************************")
    print("THIS IS IT")
    print(move)
    feature = {
        'move': _int64_feature(move.value),
        'score': _float_feature(score),
        'board': _float_list_feature(environment.board.board.flatten().tolist())
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

record_file = 'presentationData.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
  for move, score, environment in movesStored:
    # image_string = open(filename, 'rb').read()
    tf_example = example_2048(move, score, environment)
    writer.write(tf_example.SerializeToString())

# for tfrec_num in range(len(movesStored)):
#     #samples = annotations[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)]

#     with tf.io.TFRecordWriter(
#         "/dataForPresentation.tfrec"
#         #tfrecords_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
#     ) as writer:
#         for sample in movesStored:
#             # image_path = f"{images_dir}/{sample['image_id']:012d}.jpg"
#             # image = tf.io.decode_jpeg(tf.io.read_file(image_path))
#             # example = create_example(image, image_path, sample)
#             example = create_example()
#             writer.write(example.SerializeToString())
    