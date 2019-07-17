import cntk as C
from cntk.train.trainer import Trainer
from cntk.learners import sgd, adam
from cntk.losses import cross_entropy_with_softmax
from utils.buffers import FrameStacker
from agents.REINFORCE.hyperparams import BATCH_SIZE
from agents.REINFORCE.hyperparams import DISCOUNT_FACTOR


class ActorCNNPolicy:
    def __init__(self, name, observation_space_shape, num_actions, pretrained_policy=None, *args, **kwargs):
        self.name = name
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        self._build_network(pretrained_policy)
        self.trainer = Trainer(self.probabilities, self.loss, [adam(self.probabilities.parameters, lr=0.0001, momentum=0.9)])

    def _build_network(self, pretrained_policy):
        self.image_frame = C.input_variable((1,)+self.observation_space_shape)
        self.td_error = C.input_variable((1,))
        self.action_index = C.input_variable((1,))
        one_hot_action = C.one_hot(self.action_index, self.num_actions)
        if pretrained_policy is None:
            h = C.layers.Convolution2D(filter_shape=(7,7), num_filters=32, strides=(4,4), pad=True, name='conv_1', activation=C.relu)(self.image_frame)
            h = C.layers.Convolution2D(filter_shape=(5,5), num_filters=64, strides=(2,2), pad=True, name='conv_2', activation=C.relu)(h)
            h = C.layers.Convolution2D(filter_shape=(3,3), num_filters=128, strides=(1,1), pad=True, name='conv_3', activation=C.relu)(h)
            h = C.layers.Dense(64, activation=C.relu, name='dense_1')(h)
            self.probabilities = C.layers.Dense(self.num_actions, name='dense_2', activation= C.softmax)(h)
        else:
            self.probabilities = C.Function.load(pretrained_policy)(self.image_frame)
        selected_action_probablity = C.ops.times_transpose(self.probabilities, one_hot_action)
        self.log_probability = C.ops.log(selected_action_probablity)
        self.loss = -self.td_error * self.log_probability

        # self.probabilities = C.softmax(self.logits)
        # log_probability_of_action_taken = cross_entropy_with_softmax(self.logits, one_hot_action)
        # self.loss = C.reduce_mean(self.td_error*log_probability_of_action_taken)

    def optimise(self, image_frame, td_error, action_index):
        self.trainer.train_minibatch({self.image_frame: image_frame, self.td_error: td_error, self.action_index: action_index})

    def predict(self, image_frame):
        return self.probabilities.eval({self.image_frame: image_frame})


class CriticCNNPolicy:
    def __init__(self, name, observation_space_shape, num_actions, pretrained_policy=None, *args, **kwargs):
        self.name = name
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        self._build_network(pretrained_policy)
        self.trainer = Trainer(self.value, self.loss, [adam(self.value.parameters, lr=0.0001, momentum=0.9)])

    def _build_network(self, pretrained_policy):
        self.image_frame = C.input_variable((1,) + self.observation_space_shape)
        self.target_current_state_value = C.input_variable((1,))
        if pretrained_policy is None:
            h = C.layers.Convolution2D(filter_shape=(7,7), num_filters=32, strides=(4,4), pad=True, name='conv_1', activation=C.relu)(self.image_frame)
            h = C.layers.Convolution2D(filter_shape=(5,5), num_filters=64, strides=(2,2), pad=True, name='conv_2', activation=C.relu)(h)
            h = C.layers.Convolution2D(filter_shape=(3,3), num_filters=128, strides=(1,1), pad=True, name='conv_3', activation=C.relu)(h)
            h = C.layers.Dense(64, activation=C.relu, name='dense_1')(h)
            self.value = C.layers.Dense(1, name='dense_2')(h)
        else:
            self.value = C.Function.load(pretrained_policy)(self.image_frame)
        self.loss = C.squared_error(self.target_current_state_value, self.value)

    def optimise(self, image_frame, target_current_state_value):
        self.trainer.train_minibatch({self.image_frame: image_frame, self.target_current_state_value: target_current_state_value})

    def predict(self, image_frame):
        return self.value.eval({self.image_frame: image_frame})


class ActorCriticCNNPolicy:
    def __init__(self, name, observation_space_shape, num_actions, pretrained_policy=None, *args, **kwargs):
        self.name = name
        self.observation_space_shape = observation_space_shape
        self.num_actions = num_actions
        self._build_network(pretrained_policy)
        self.trainer = Trainer(self.output, self.loss, [adam(self.output.parameters, lr=0.0001, momentum=0.9)])

    def _build_network(self, pretrained_policy):
        self.image_frame = C.input_variable((1,)+self.observation_space_shape)
        self.next_image_frame = C.input_variable((1,) + self.observation_space_shape)
        self.advantage = C.input_variable((1,))
        self.action_index = C.input_variable((1,))
        one_hot_action = C.one_hot(self.action_index, self.num_actions)
        if pretrained_policy is None:
            h = C.layers.Convolution2D(filter_shape=(7,7), num_filters=32, strides=(4,4), pad=True, name='conv_1', activation=C.relu)
            h = C.layers.Convolution2D(filter_shape=(5,5), num_filters=64, strides=(2,2), pad=True, name='conv_2', activation=C.relu)(h)
            h = C.layers.Convolution2D(filter_shape=(3,3), num_filters=128, strides=(1,1), pad=True, name='conv_3', activation=C.relu)(h)
            h = C.layers.Dense(64, activation=C.relu, name='dense_1')(h)
            self.probabilities = C.layers.Dense(self.num_actions, name='dense_2', activation= C.softmax)(h(self.image_frame))
            v = C.layers.Dense(1, name='dense_3')(h)
            self.value = v(self.image_frame)
            self.next_value = v(self.next_image_frame)
            self.output = C.combine([self.probabilities, self.value, self.next_value])
        else:
            [self.probabilities, self.value, self.next_value] = list(C.Function.load(pretrained_policy)(self.image_frame, self.next_image_frame))
        self.values_output = C.combine([self.value, self.next_value])
        selected_action_probablity = C.ops.times_transpose(self.probabilities, one_hot_action)
        self.log_probability = C.ops.log(selected_action_probablity)
        target_value = self.advantage + self.value
        self.actor_loss = -self.advantage * self.log_probability

        self.critic_loss = C.squared_error(target_value, self.value)

        self.loss = 0.2*self.actor_loss + 0.8*self.critic_loss

        # self.probabilities = C.softmax(self.logits)
        # log_probability_of_action_taken = cross_entropy_with_softmax(self.logits, one_hot_action)
        # self.loss = C.reduce_mean(self.td_error*log_probability_of_action_taken)

    def optimise(self, image_frame, action_index, advantage):
        self.trainer.train_minibatch({self.image_frame: image_frame, self.advantage: advantage,
                                      self.action_index: action_index})

    def predict(self, image_frame):
        return self.probabilities.eval({self.image_frame: image_frame})

    def values(self, image_frame, next_image_frame):
        output = self.values_output.eval({self.image_frame: image_frame, self.next_image_frame: next_image_frame})
        return output[self.value.output], output[self.next_value.output]


