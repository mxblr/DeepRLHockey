import numpy as np
import tensorflow as tf
import gym
from gym import spaces
import itertools
import time
import pylab as plt
import sonnet as snt
import laser_hockey_env as lh

mode = 2
env_name = "LaserHockey"
env = lh.LaserHockeyEnv(mode = mode)
ac_space = env.action_space
o_space = env.observation_space
print(ac_space)
print(o_space)
print(env.observation_space.low, env.observation_space.high)
print(ac_space.low, ac_space.high)

# TODO: Parameter space noise for exploration Paper !!!

class Memory():
    # class to store x/u trajectory
    def __init__(self, buffer_shapes, buffer_size=int(1e5)):
        self._buffer_size = buffer_size
        self._buffer_shapes = buffer_shapes
        self._data = {key: np.empty((self._buffer_size, value) if value is not None else (self._buffer_size,))
                      for key, value in self._buffer_shapes.items()}
        self._current_size = 0
        self._t = 0

    def add_item(self, new_data):
        for key in self._data.keys():
            self._data[key][self._t%self._buffer_size] = new_data[key]
        self._t += 1
        self._current_size = min(self._t, self._buffer_size)

    def sample(self, batch_size=1):
        if batch_size > self._current_size:
            batch_size = self._current_size
        inds = np.random.choice(range(self._current_size), size=batch_size, replace=False)
        batch = {key: value[inds] for key, value in self._data.items()}
        return batch



class ActorFunction:
    def __init__(self, o_space, a_space, gamma=0.99, scope='Q'):
        self._scope = scope
        self._o_space = o_space
        self._a_space = a_space
        self._gamma = gamma
        self._sess = tf.get_default_session() or tf.InteractiveSession()

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, 1], name = "action_gradient")

        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            self._build_graph()

    def _build_graph(self):
        self.state = tf.placeholder(dtype=tf.float32, shape=(None, self._o_space.shape[0]), name="state")

        net_arch1 = snt.Linear(output_size=125,
                              name='h1',
                              initializers={'w': tf.initializers.glorot_normal(),
                                            'b': tf.initializers.constant(0.0)})

        net_arch2 = snt.Linear(output_size=125,
                               name='h2',
                               initializers={'w': tf.initializers.glorot_normal(),
                                             'b': tf.initializers.constant(0.0)})

        w_init = tf.initializers.random_uniform(minval=-0.03, maxval=0.03)
        net_arch3 = snt.Linear(output_size=3,
                              name='h3',
                              initializers={'w': w_init,
                                            'b': tf.initializers.constant(0.0)})

        self.h = tf.nn.leaky_relu(tf.layers.batch_normalization(net_arch1(self.state)))
        self.h = tf.nn.leaky_relu(tf.layers.batch_normalization(net_arch2(self.h)))
        self.h = net_arch3(self.h)
        self.output = tf.squeeze(self.h)
        self.output = tf.multiply(tf.math.tanh(self.output), 1)



    def predict(self, state):
        _state = np.asarray(state).reshape(-1, self._o_space.shape[0])
        inp = {self.state: _state}
        return self._sess.run(self.output, feed_dict=inp).reshape(-1, 1)

class CriticFunction:
    def __init__(self, o_space, a_space, gamma=0.99, scope='Q'):
        self._scope = scope
        self._o_space = o_space
        self._a_space = a_space
        self._gamma = gamma
        self._sess = tf.get_default_session() or tf.InteractiveSession()

        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            self._build_graph()

    def _build_graph(self):
        self.state = tf.placeholder(dtype=tf.float32, shape=(None, self._o_space.shape[0]), name="state")
        self.action = tf.placeholder(dtype=tf.float32, shape=(None, 3), name="action")

        net_arch1a = snt.Linear(output_size=125,
                              name='h1',
                              initializers={'w': tf.initializers.glorot_normal(),
                                            'b': tf.initializers.constant(0.0)})
        net_arch1b = snt.Linear(output_size=125,
                              name='h1',
                              initializers={'w': tf.initializers.glorot_normal(),
                                            'b': tf.initializers.constant(0.0)})

        net_arch2 = snt.Linear(output_size=125,
                               name='h2',
                               initializers={'w': tf.initializers.glorot_normal(),
                                             'b': tf.initializers.constant(0.0)})

        net_arch3 = snt.Linear(output_size=125,
                              name='h2',
                              initializers={'w': tf.initializers.glorot_normal(),
                                            'b': tf.initializers.constant(0.0)})


        net_arch4 = snt.Linear(output_size=1,
                              name='h3',
                              initializers={'w': tf.initializers.glorot_normal(),
                                            'b': tf.initializers.constant(0.0)})

        self.ha = tf.nn.relu(tf.layers.batch_normalization(net_arch1a(self.state)))
        self.hb = tf.nn.relu(tf.layers.batch_normalization(net_arch1b(self.ha)))
        self.aaa = tf.nn.relu(net_arch2(self.action))
        self.h = tf.nn.relu(tf.concat([self.aaa, self.hb], axis = 1))
        self.h = net_arch3(self.h)
        self.output = net_arch4(self.h)
        #self.output = tf.squeeze(self.h)

        self.action_grads = tf.gradients(self.output, self.action)


    def predict(self, state, action):
        _state = np.asarray(state).reshape(-1, self._o_space.shape[0])
        _action = np.asarray(action).reshape(-1, 3)
        inp = {self.state: _state, self.action: _action}
        res =  self._sess.run(self.output, feed_dict=inp)
        res = res.reshape(-1, 1)
        return res

    def action_gradients(self, inputs, actions, session):
        return session.run(self.action_grads, feed_dict={
            self.state: inputs,
            self.action: actions.reshape(-1, 3)
        })

class Agent:
    def __init__(self, o_space, a_space, scope='Agent', **userconfig):

        self._o_space = o_space
        self._a_space = a_space
        self._config = {
            "eps_begin": 0.05,            # Epsilon in epsilon greedy policies
            "eps_end": 0.05,
            "eps_decay": 0.99,
            "discount": 0.99,
            "buffer_size": int(1e5),
            "batch_size": 50,
            "learning_rate_actor": 0.0001,
            "learning_rate_critic":0.001,
            "learning_rate": 1e-4,
            "theta": 0.05,
            "use_target_net": True,}
        self._config.update(userconfig)
        self._scope = scope
        self._eps = self._config['eps_begin']
        self._buffer_shapes = {
            's': self._o_space.shape[0],
            'a': 3,
            'r': None,
            's_prime': self._o_space.shape[0],
            'd': None,
        }
        self._buffer = Memory(buffer_shapes=self._buffer_shapes, buffer_size=self._config['buffer_size'])
        self._sess = tf.get_default_session() or tf.InteractiveSession()
        self.batch_size = 64


        # Create  Networks
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            self._actor = ActorFunction(scope='actor', o_space=self._o_space, a_space=self._a_space,
                                         gamma=self._config['discount'])
            self._critic = CriticFunction(scope='critic', o_space=self._o_space, a_space=self._a_space,
                                         gamma=self._config['discount'])
            if self._config['use_target_net']:
                self._actor_target = ActorFunction(scope='Actor_target', o_space=self._o_space, a_space=self._a_space,
                                             gamma=self._config['discount'])
                self._critic_target = CriticFunction(scope='Critic_target', o_space=self._o_space, a_space=self._a_space,
                                             gamma=self._config['discount'])

            self._prep_train_critic()
            self._prep_train_actor()

        self._sess.run(tf.global_variables_initializer())


    def _vars(self, scope=''):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope + '/' + scope)

    def _global_vars(self, scope=''):
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + scope)

    def _prep_train_critic(self):
            self._Qval = self._critic.output
            self._target = tf.placeholder(dtype=tf.float32, shape=(None,1), name="target")
            self._loss = tf.reduce_mean(tf.square(self._Qval - self._target))
            self._optim = tf.contrib.opt.AdamWOptimizer(0.005,learning_rate=self._config['learning_rate_critic'])
            self._train_op = self._optim.minimize(self._loss)

            if self._config['use_target_net']:
                self._update_target_op_critic = [tf.assign(
                target_var,
                (1-self._config['theta'])*target_var+self._config['theta']*Q_var)
                                     for target_var, Q_var in zip(self._vars('Critic_target'), self._vars('critic'))]

    def _prep_train_actor(self):
            self.network_params = self._vars(scope = 'actor')
            self.action_gradient = tf.placeholder(tf.float32, [None, 3], name = "action_gradeient_in_prep_train_actor")
            self.unnormalized_actor_gradients = tf.gradients(self._actor.output, self.network_params, -self.action_gradient)
            self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
            # Optimization Op
            self.optimize = tf.contrib.opt.AdamWOptimizer(0.005,self._config['learning_rate_actor']).\
                             apply_gradients(zip(self.actor_gradients, self.network_params))


            if self._config['use_target_net']:
                self._update_target_op_actor = [tf.assign(
                target_var,
                (1-self._config['theta'])*target_var+self._config['theta']*Q_var)
                                     for target_var, Q_var in zip(self._vars('Actor_target'), self._vars('actor'))]


    def _update_target_net(self):
            self._sess.run(self._update_target_op_critic)
            self._sess.run(self._update_target_op_actor)


    def act(self, observation, eps=None):
            if eps is None:
                eps = self._eps
            # epsilon greedy.
            if np.random.random() > eps:
                observation = observation.reshape(-1, 16)
                fddct = {self._actor.state : observation}
                action = self._sess.run([self._actor.output], feed_dict = fddct)
                action = action[0]
            else:
                action = self._a_space.sample()[:3]
            return action


    def store_transition(self, transition):
            self._buffer.add_item(transition)

    def _eps_scheduler(self, writer=None, ep=0):
            self._eps = max(self._config['eps_end'], self._eps*self._config['eps_decay'])
            if writer: writer.add_scalar('policy/eps', self._eps, ep)


    def train(self, iter_fit=10, writer=None, ep=0):
            losses = []
            for i in range(iter_fit):

                # sample from the replay buffer
                data = self._buffer.sample(batch_size=self._config['batch_size'])

                s = data['s'] # s_t
                a = data['a'] # a_t
                r = data['r'] # rew
                s_prime = data['s_prime'] # s_t+1
                d = data['d'] # done
                a_prime = self._actor_target.predict(s_prime)
                if self._config['use_target_net']:
                    v_prime = self._critic_target.predict(s_prime, a_prime)
                else:
                    v_prime = self._critic.predict(s_prime, a_prime)

                # target
                part1 = r.reshape(-1,1)
                part2 = (self._config['discount'] * v_prime).reshape(-1,1)
                part3 = (1-d).reshape(-1,1)
                td_target = np.add(part1, np.multiply(part3, part2))
                td_target = td_target.reshape(-1,1)

                # optimize the lsq objective
                a = a.reshape(-1, 3)
                inp = {self._critic.state: s, self._critic.action: a, self._target: td_target}
                fit_loss = self._sess.run([self._train_op, self._loss], feed_dict=inp)[1]
                losses.append(fit_loss)

                # optimize the actor objective
                a_outs = self._actor.predict(s)
                grads = self._critic.action_gradients(s, a_outs, self._sess)
                grads = np.asarray(grads).reshape(-1,3)
                self._sess.run(self.optimize, feed_dict={self._actor.state: s, self.action_gradient: grads})

                if self._config['use_target_net']:
                    self._update_target_net()

            return losses




fps = 100 # env.metadata.get('video.frames_per_second')
max_steps = 1000#env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']

agent = Agent(o_space, ac_space, discount=0.99, eps_begin=0.3)

basic_op = lh.BasicOpponent()

ob = env.reset()
agent._actor.predict(ob)

stats = []
losses = []

from tensorboardX import SummaryWriter
from datetime import datetime
writer = SummaryWriter('{}-{}'.format(env_name, datetime.now().strftime('%b%d_%H-%M-%S')))

max_episodes=10000
#mode="random"
show=True
mode="Q"

agent._update_target_net()
agent._update_target_net()

# run the train loop?
do_training = True

if do_training:
    for i in range(max_episodes):
       # print("Starting a new episode")
       total_reward = 0
       ob = env.reset()
       max_height = -np.inf
       for t in range(max_steps):
           done = False
           if mode == "random":
               a = ac_space.sample()
           elif mode == "Q":
               a = agent.act(ob)
           else:
               raise ValueError("no implemented")
           obs_agent2 = env.obs_agent_two()
           basic_op_a = basic_op.act(obs_agent2)
           concat_act = np.hstack([a, basic_op_a])
           (ob_new, reward, done, _info) = env.step(concat_act)
           reward = reward + _info['reward_closeness_to_puck'] + _info['reward_puck_direction'] + _info['reward_touch_puck']
           total_reward+= reward
           if mode == "Q":
               agent.store_transition({'s': ob, 'a': a, 'r': reward, 's_prime': ob_new, 'd': done})
           ob=ob_new
           if show:
               time.sleep(1.0/fps)
               env.render(mode='human')
           if mode == "Q":
              losses.extend(agent.train(1, writer=writer, ep=i))
              stats.append([i,total_reward,t+1])
              agent._eps_scheduler(writer=writer, ep=i)
              writer.add_scalar('rollout/tot_reward', total_reward, i)
              writer.add_scalar('policy/loss', losses[-1], i)
              win = _info['winner']
           if done:
              writer.add_scalar('win', win, i)
              break

       if ((i-1)%1==0):
           print("Done after {} steps. Reward: {}".format(t+1, total_reward), " Episode:", i)
