import cntk as C
from cntk.train.trainer import Trainer
from cntk.learners import sgd
from utils.buffers import FrameStacker


class SimpleCNNPolicy:
    def __init__(self, name, observation_space_shape, num_actions, pretrained_policy=None, *args, **kwargs):
        self.name = name
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        self._build_network(pretrained_policy)
        self.trainer = Trainer(self.q, self.loss, [sgd(self.q.parameters, lr=0.000001)])

    def _build_network(self, pretrained_policy):
        self.image_frame = C.input_variable((1,) + self.observation_space_shape, name='image frame')
        if pretrained_policy is None:
            h = C.layers.Convolution2D(filter_shape=(9,9), num_filters=32, strides=(4,4), pad=(True,True), name='conv_1', activation=C.relu)(self.image_frame)
            h = C.layers.Convolution2D(filter_shape=(5,5), num_filters=64, strides=(2,2), pad=(True,True), name='conv_2', activation=C.relu)(h)
            h = C.layers.Convolution2D(filter_shape=(3,3), num_filters=128, strides=(1,1), pad=(True,True), name='conv_3', activation=C.relu)(h)
            h = C.layers.Dense(64, activation=C.relu, name='dense_1')(h)
            self.q = C.layers.Dense(self.num_actions, name='dense_1')(h)
        else:
            self.q = C.Function.load(pretrained_policy)(self.image_frame)
        self.q_target = C.input_variable(self.num_actions, name='q_target')
        self.loss = C.mean(C.losses.squared_error(self.q_target, self.q))

    def optimise(self, image_frame, q_target):
        self.trainer.train_minibatch({self.image_frame : image_frame, self.q_target : q_target})

    def predict(self, image_frame):
        return self.q.eval({self.image_frame : image_frame})


class StackedFrameCNNPolicy:
    def __init__(self, name, num_frames_to_stack, observation_space_shape, num_actions, pretrained_policy=None, *args, **kwargs):
        self.name = name
        self.num_frames_to_stack = num_frames_to_stack
        self.observation_space_shape = observation_space_shape
        self.frame_stacker = FrameStacker(stack_size=num_frames_to_stack, frame_shape=observation_space_shape)
        self.num_actions = num_actions
        self._build_network(pretrained_policy)
        self.trainer = Trainer(self.q, self.loss, [sgd(self.q.parameters, lr=0.000001)])

    def _build_network(self, pretrained_policy):
        self.image_frame = C.input_variable((self.num_frames_to_stack,) + self.observation_space_shape)
        if pretrained_policy is None:
            h = C.layers.Convolution2D(filter_shape=(7,7), num_filters=32, strides=(4,4), pad=True, name='conv_1', activation=C.relu)(self.image_frame)
            h = C.layers.Convolution2D(filter_shape=(5,5), num_filters=64, strides=(2,2), pad=True, name='conv_2', activation=C.relu)(h)
            h = C.layers.Convolution2D(filter_shape=(3,3), num_filters=128, strides=(1,1), pad=True, name='conv_3', activation=C.relu)(h)
            h = C.layers.Dense(64, activation=C.relu, name='dense_1')(h)
            self.q = C.layers.Dense(self.num_actions, name='dense_1')(h)
        else:
            self.q = C.Function.load(pretrained_policy)(self.image_frame)
        self.q_target = C.input_variable(self.num_actions)
        self.loss = C.mean(C.losses.squared_error(self.q_target, self.q))

    def optimise(self, image_frame, q_target):
        self.trainer.train_minibatch({self.image_frame : image_frame, self.q_target : q_target})

    def predict(self, image_frame):
        return self.q.eval({self.image_frame : image_frame})


class ActorStackedFrameCNNPolicy:
    def __init__(self, name, num_frames_to_stack, observation_space_shape, num_actions, pretrained_policy=None, *args, **kwargs):
        self.name = name
        self.num_frames_to_stack = num_frames_to_stack
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        self._build_network(pretrained_policy)
        self.trainer = Trainer(self.log_probability, self.loss, [sgd(self.probabilities.parameters, lr=0.000001)])

    def _build_network(self, pretrained_policy):
        self.image_frame = C.input_variable((self.num_frames_to_stack,) + self.observation_space_shape)
        self.td_error = C.input_variable((1,))
        self.action_index = C.input_variable((1,))
        one_hot_action = C.ops.squeeze(C.one_hot(self.action_index, self.num_actions))
        if pretrained_policy is None:
            h = C.layers.Convolution2D(filter_shape=(7,7), num_filters=32, strides=(4,4), pad=True, name='conv_1', activation=C.relu)(self.image_frame)
            h = C.layers.Convolution2D(filter_shape=(5,5), num_filters=64, strides=(2,2), pad=True, name='conv_2', activation=C.relu)(h)
            h = C.layers.Convolution2D(filter_shape=(3,3), num_filters=128, strides=(1,1), pad=True, name='conv_3', activation=C.relu)(h)
            h = C.layers.Dense(64, activation=C.relu, name='dense_1')(h)
            self.probabilities = C.layers.Dense(self.num_actions, name='dense_1', activation=C.softmax)(h)
        else:
            self.probabilities = C.Function.load(pretrained_policy)(self.image_frame)
        self.log_probability = C.ops.log(C.ops.times_transpose(self.probabilities, one_hot_action))
        self.loss = -self.td_error*self.log_probability

    def optimise(self, image_frame, td_error, action_index):
        self.trainer.train_minibatch({self.image_frame: image_frame, self.td_error: td_error, self.action_index: action_index})

    def predict(self, image_frame):
        return self.probabilities.eval({self.image_frame: image_frame})


class CriticStackedFrameCNNPolicy:
    def __init__(self, name, num_frames_to_stack, observation_space_shape, num_actions, pretrained_policy=None, *args, **kwargs):
        self.name = name
        self.num_frames_to_stack = num_frames_to_stack
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        self._build_network(pretrained_policy)
        self.trainer = Trainer(self.value, self.loss, [sgd(self.value.parameters, lr=0.000001)])

    def _build_network(self, pretrained_policy):
        self.image_frame = C.input_variable((self.num_frames_to_stack,) + self.observation_space_shape)
        self.target_current_state_value = C.input_variable((1,))
        if pretrained_policy is None:
            h = C.layers.Convolution2D(filter_shape=(7,7), num_filters=32, strides=(4,4), pad=True, name='conv_1', activation=C.relu)(self.image_frame)
            h = C.layers.Convolution2D(filter_shape=(5,5), num_filters=64, strides=(2,2), pad=True, name='conv_2', activation=C.relu)(h)
            h = C.layers.Convolution2D(filter_shape=(3,3), num_filters=128, strides=(1,1), pad=True, name='conv_3', activation=C.relu)(h)
            h = C.layers.Dense(64, activation=C.relu, name='dense_1')(h)
            self.value = C.layers.Dense(1, name='dense_1')(h)
        else:
            self.value = C.Function.load(pretrained_policy)(self.image_frame)
        self.td_error = self.target_current_state_value - self.value
        self.output = C.combine([self.value, self.td_error])
        self.loss = C.mean(C.squared_error(self.target_current_state_value, self.value))

    def optimise(self, image_frame, target_current_state_value):
        self.trainer.train_minibatch({self.image_frame: image_frame, self.target_current_state_value: target_current_state_value})

    def predict(self, image_frame):
        return self.value.eval({self.image_frame: image_frame}),

