import numpy as np

import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.python.training import training_util


class Normal_Policy:
    def __init__(self, inp,  scope='policy', **userconfig):
        self._scope = scope

        self._sess = tf.get_default_session() or tf.InteractiveSession()
        self._input = inp
        self._config = {
            "hidden_layers": [256, 256], 
            "hidden_act_fct": tf.nn.relu,
            "output_act_fct": None,
            "weights_init" : tf.contrib.layers.xavier_initializer(),
            "bias_init": tf.constant_initializer(0.),
            "dim": 1
            }
        self._config.update(userconfig)

    
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            self._build_graph()
            
    def _build_graph(self):
        x = self._input
        for l in self._config["hidden_layers"]:
            x = tf.layers.dense(x, l, activation=self._config["hidden_act_fct"], kernel_initializer=self._config["weights_init"], bias_initializer= self._config["bias_init"])
        
        self.mu = tf.layers.dense(x, self._config["dim"],activation=self._config["output_act_fct"], kernel_initializer=self._config["weights_init"], bias_initializer= self._config["bias_init"])
 
        self.log_std = tf.layers.dense(x,self._config["dim"],activation=self._config["output_act_fct"], kernel_initializer=self._config["weights_init"], bias_initializer= self._config["bias_init"])
        self.log_std = tf.clip_by_value(self.log_std, -20, 2)
        self.std = tf.exp(self.log_std)

        self.normal_dist = tfp.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=self.std)
        
        self.sample = self.normal_dist.sample()
        
        self.act = tf.tanh(self.sample)
        self.log_prob = self.normal_dist.log_prob(self.sample)
        
        