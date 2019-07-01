import cntk as C
from cntk.train.trainer import Trainer
from cntk.learners import sgd, adam
from utils.buffers import FrameStacker


class SimpleNNPolicy:
    def __init__(self, name, observation_space_shape, num_actions, pretrained_policy=None, *args, **kwargs):
        self.name = name
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        self._build_network(pretrained_policy)
        self.trainer = Trainer(self.q, self.loss, [sgd(self.q.parameters, lr=0.000001)])

    def _build_network(self, pretrained_policy):
        self.input = C.input_variable(self.observation_space_shape, name='image frame')
        if pretrained_policy is None:
            h = C.layers.Dense(32, activation=C.relu, name='dense_1')(self.input)
            h = C.layers.Dense(8, activation=C.relu, name='dense_1')(h)
            self.q = C.layers.Dense(self.num_actions, name='dense_1')(h)
        else:
            self.q = C.Function.load(pretrained_policy)(self.input)
        self.q_target = C.input_variable(self.num_actions, name='q_target')
        self.loss = C.mean(C.losses.squared_error(self.q_target, self.q))

    def optimise(self, state, q_target):
        self.trainer.train_minibatch({self.input : state, self.q_target : q_target})

    def predict(self, state):
        return self.q.eval({self.input : state})


class ActorNNPolicy:
    def __init__(self, name, observation_space_shape, num_actions, pretrained_policy=None, *args, **kwargs):
        self.name = name
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        self._build_network(pretrained_policy)
        self.trainer = Trainer(self.log_probability, self.loss, [adam(self.probabilities.parameters, lr=0.00001, momentum=0.9)])

    def _build_network(self, pretrained_policy):
        self.input = C.input_variable(self.observation_space_shape)
        self.td_error = C.input_variable((1,))
        self.action_index = C.input_variable((1,))
        one_hot_action = C.ops.squeeze(C.one_hot(self.action_index, self.num_actions))
        if pretrained_policy is None:
            h = C.layers.Dense(64, activation=C.relu, name='dense_1')(self.input)
            h = C.layers.Dense(32, activation=C.tanh, name='dense_1')(h)
            self.probabilities = C.layers.Dense(self.num_actions, name='dense_1', activation=C.softmax)(h)
        else:
            self.probabilities = C.Function.load(pretrained_policy)(self.input)
        selected_action_probablity = C.ops.times_transpose(self.probabilities, one_hot_action)
        self.log_probability = C.ops.log(selected_action_probablity)
        self.loss = -self.td_error*self.log_probability

    def optimise(self, state, td_error, action_index):
        self.trainer.train_minibatch({self.input : state, self.td_error: td_error, self.action_index: action_index})

    def predict(self, state):
        return self.probabilities.eval({self.input : state})


class CriticNNPolicy:
    def __init__(self, name, observation_space_shape, num_actions, pretrained_policy=None, *args, **kwargs):
        self.name = name
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        self._build_network(pretrained_policy)
        self.trainer = Trainer(self.value, self.loss, [adam(self.value.parameters, lr=0.00001, momentum=0.9)])

    def _build_network(self, pretrained_policy):
        self.input = C.input_variable(self.observation_space_shape)
        self.target_current_state_value = C.input_variable((1,))
        if pretrained_policy is None:
            h = C.layers.Dense(64, activation=C.relu, name='dense_1')(self.input)
            h = C.layers.Dense(32, activation=C.relu, name='dense_1')(h)
            self.value = C.layers.Dense(1, name='dense_1')(h)
        else:
            self.value = C.Function.load(pretrained_policy)(self.input)
        self.loss = C.squared_error(self.target_current_state_value, self.value)

    def optimise(self, state, target_current_state_value):
        self.trainer.train_minibatch({self.input: state, self.target_current_state_value: target_current_state_value})

    def predict(self, state):
        return self.value.eval({self.input: state})


class REINFORCENNPolicy:
    def __init__(self, name, observation_space_shape, num_actions, pretrained_policy=None, *args, **kwargs):
        self.name = name
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        self._build_network(pretrained_policy)
        self.trainer = Trainer(self.action_probabilities, self.loss, [sgd(self.action_probabilities.parameters, lr=0.000001)])

    def _build_network(self, pretrained_policy):
        self.input = C.input_variable(self.observation_space_shape, name='image frame')
        self.target = C.input_variable((1,), name='q_target')
        self.action_index = C.input_variable((1,))
        one_hot_action = C.ops.squeeze(C.one_hot(self.action_index, self.num_actions))
        if pretrained_policy is None:
            h = C.layers.Dense(64, activation=C.relu, name='dense_1')(self.input)
            h = C.layers.Dense(32, activation=C.relu, name='dense_1')(h)
            self.action_probabilities = C.layers.Dense(self.num_actions, activation=C.softmax, name='dense_1')(h)
        else:
            self.action_probabilities = C.Function.load(pretrained_policy)(self.input)
        selected_action_probablity = C.ops.times_transpose(self.action_probabilities, one_hot_action)
        self.log_probability = C.ops.log(selected_action_probablity)
        self.loss = C.sum(self.log_probability*self.target)

    def optimise(self, state, action, target):
        self.trainer.train_minibatch({self.input: state, self.action_index : action, self.target : target})

    def predict(self, state):
        return self.action_probabilities.eval({self.input : state})

