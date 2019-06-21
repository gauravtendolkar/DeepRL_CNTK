import cntk as C
from cntk.train.trainer import Trainer
from cntk.learners import sgd, adam


class SimpleCNNPolicy:
    def __init__(self, name, observation_space_shape, num_actions, *args, **kwargs):
        self.name = name
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        self._build_network()
        self.trainer = Trainer(self.q, self.loss, [sgd(self.q.parameters, lr=0.000001)])

    def _build_network(self):
        self.image_frame = C.input_variable((1,) + self.observation_space_shape, name='image frame')
        h = C.layers.Convolution2D(filter_shape=(9,9), num_filters=32, strides=(4,4), pad=(True,True), name='conv_1', activation=C.relu)(self.image_frame)
        h = C.layers.Convolution2D(filter_shape=(5,5), num_filters=64, strides=(2,2), pad=(True,True), name='conv_2', activation=C.relu)(h)
        h = C.layers.Convolution2D(filter_shape=(3,3), num_filters=128, strides=(1,1), pad=(True,True), name='conv_3', activation=C.relu)(h)
        h = C.layers.Dense(64, activation=C.relu, name='dense_1')(h)
        self.q = C.layers.Dense(self.num_actions, name='dense_1')(h)
        self.q_target = C.input_variable(self.num_actions, name='q_target')
        self.loss = C.mean(C.losses.squared_error(self.q_target, self.q))

    def optimise(self, image_frame, q_target):
        self.trainer.train_minibatch({self.image_frame : image_frame, self.q_target : q_target})

    def predict(self, image_frame):
        return self.q.eval({self.image_frame : image_frame})


class StackedFrameCNNPolicy:
    def __init__(self, name, num_frames_to_stack, observation_space_shape, num_actions, *args, **kwargs):
        self.name = name
        self.num_frames_to_stack = num_frames_to_stack
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        self._build_network()
        self.trainer = Trainer(self.q, self.loss, [sgd(self.q.parameters, lr=0.01)])

    def _build_network(self):
        self.image_frame = C.input_variable((self.num_frames_to_stack,) + self.observation_space_shape)
        h = C.layers.Convolution2D(filter_shape=(8,8), num_filters=32, strides=(4,4), pad=True, name='conv_1', activation=C.relu)(self.image_frame)
        h = C.layers.Convolution2D(filter_shape=(4,4), num_filters=64, strides=(2,2), pad=True, name='conv_2', activation=C.relu)(h)
        h = C.layers.Convolution2D(filter_shape=(3,3), num_filters=128, strides=(1,1), pad=True, name='conv_3', activation=C.relu)(h)
        h = C.layers.Dense(64, activation=C.relu, name='dense_1')(h)
        self.q = C.layers.Dense(self.num_actions, name='dense_1')(h)
        self.q_target = C.input_variable(self.num_actions)
        self.loss = C.mean(C.losses.squared_error(self.q_target, self.q))

    def optimise(self, image_frame, q_target):
        self.trainer.train_minibatch({self.image_frame : image_frame, self.q_target : q_target})

    def predict(self, image_frame):
        return self.q.eval({self.image_frame : image_frame})


