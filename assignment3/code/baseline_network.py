import tensorflow as tf
import numpy as np
import pdb
from network_utils import build_mlp


class BaselineNetwork(object):
    """
  Class for implementing Baseline network
  """

    def __init__(self, env, config, observation_placeholder):
        self.config = config
        self.env = env
        self.observation_placeholder = observation_placeholder

        self.add_baseline_placeholder()

        self.baseline = None
        self.lr = self.config.learning_rate

    def add_baseline_placeholder(self):
        self.baseline_target_placeholder = tf.placeholder(tf.float32, shape=(None,))

    def set_session(self, session):
        self.sess = session

    def add_baseline_op(self, scope="baseline"):
        """
    Build the baseline network within the scope.

    In this function we will build the baseline network.
    Use build_mlp with the same parameters as the policy network to
    get the baseline estimate. You also have to setup a target
    placeholder and an update operation so the baseline can be trained.

    Args:
        scope: the scope of the baseline network

    TODO: Set the following fields
        self.baseline
            HINT: use build_mlp, the network is the same as policy network
            check self.config for n_layers and layer_size
            HINT: tf.squeeze might be helpful
        self.baseline_target_placeholder --> Not required anymore
        self.update_baseline_op
            HINT: first construct a loss using tf.losses.mean_squared_error.
            HINT: use AdamOptimizer with self.lr

    """
        ######################################################
        #########   YOUR CODE HERE - 4-8 lines.   ############

        self.baseline = tf.squeeze(
            build_mlp(
                mlp_input=self.observation_placeholder,
                output_size=1,
                scope=scope,
                n_layers=self.config.n_layers,
                size=self.config.layer_size,
                output_activation=None,
            ),
            axis=-1,
        )
        loss = tf.compat.v1.losses.mean_squared_error(
            labels=self.baseline_target_placeholder, predictions=self.baseline
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.update_baseline_op = optimizer.minimize(loss)

        #######################################################
        #########          END YOUR CODE.          ############

    def calculate_advantage(self, returns, observations):
        """
    Calculate the advantage

    Args:
            returns: all discounted future returns for each step
            observations: observations
    Returns:
            adv: Advantage

    Calculate the advantages, using baseline adjustment
    
    TODO:
    We need to evaluate the baseline and subtract it from the returns to get the advantage.
    HINT: evaluate the self.baseline with self.sess.run(...)

    """
        #######################################################
        #########   YOUR CODE HERE - 1-4 lines.   ############

        baseline = self.sess.run(
            self.baseline, feed_dict={self.observation_placeholder: observations}
        )
        adv = returns - baseline

        #######################################################
        #########          END YOUR CODE.          ############
        return adv

    def update_baseline(self, returns, observations):
        """
    Update the baseline from given returns and observation.

    Args:
            returns: Returns from get_returns
            observations: observations
    TODO:
      apply the baseline update op with the observations and the returns.
      HINT: Run self.update_baseline_op with self.sess.run(...)
    """
        #######################################################
        #########   YOUR CODE HERE - 1-5 lines.   ############

        baseline = self.sess.run(
            self.update_baseline_op,
            feed_dict={
                self.observation_placeholder: observations,
                self.baseline_target_placeholder: returns,
            },
        )

        #######################################################
        #########          END YOUR CODE.          ############
