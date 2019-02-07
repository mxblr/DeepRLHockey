__author__  = "Maximilian Beller"

import gym
import numpy as np

import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.python.training import training_util

from  SAC.memory import Memory
from  SAC.memory2 import ReplayBuffer
import progressbar

from IPython.display import clear_output
import matplotlib.pyplot as plt

import os 

def plot(total_rewards_per_episode=[],total_loss_V= [],total_loss_Q1= [], total_loss_Q2= [], total_loss_PI= [], winning = [], plot_type = 0):
    clear_output(True)
    if plot_type== 0:
        plt.plot(range(len(total_rewards_per_episode)), total_rewards_per_episode)
    elif plot_type ==1:
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].plot(range(len(total_rewards_per_episode)), total_rewards_per_episode)
        axes[0, 0].set_title("Reward")
        axes[0, 1].plot(range(len(total_loss_V)), total_loss_V)
        axes[0, 1].set_title("V loss")
      
        axes[1, 0].plot(range(len(total_loss_Q1)), total_loss_Q1, c="r")
        axes[1, 0].plot(range(len(total_loss_Q2)), total_loss_Q2, c="b")
        axes[1, 0].set_title("Q losses")
        axes[1, 1].plot(range(len(total_loss_PI)), total_loss_PI)
        axes[1, 1].set_title("PI Loss") 
    elif plot_type ==2:
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(range(len(total_rewards_per_episode)), total_rewards_per_episode)
        axes[0].set_title("Reward")
        axes[1].plot(range(len(winning)), winning)
        axes[1].set_title("Win fraction")
    elif plot_type ==3:
        plt.plot(range(len(winning)), winning)
        
    plt.show()




class SoftActorCritic:
    def __init__(self, o_space, a_space, value_fct, policy_fct, env, q_fct_config = {}, v_fct_config = {}, pi_fct_config = {}, scope='SACAgent', save_path ="/weights/model.ckpt" , **userconfig):
        self._o_space = o_space
        self._a_space = a_space
        self._config = {
            "tau": 0.005, 
            "lambda_V": 3e-4,
            "lambda_Q": 3e-4,
            "lambda_Pi": 3e-4,
            "lambda_Alpha": 3e-4,
            "discount": 0.99,
            "target_update":1,
            "buffer_size": int(1e6),
            "batch_size": 256,
            "initial_alpha":1.0,
            "dim_act":3,
            "dim_obs":16}
        self._config.update(userconfig)
        self._scope = scope
        self._sess = tf.get_default_session() or tf.InteractiveSession()
        self.global_step = training_util.get_or_create_global_step()
        #self.buffer = Memory(self._config["buffer_size"])
        self.buffer =  ReplayBuffer(obs_dim=self._config["dim_obs"], act_dim=self._config["dim_act"], size=self._config["buffer_size"])
        self._v_fct =  value_fct
        self._pi_fct = policy_fct
        self._q_fct_config = q_fct_config
        self._v_fct_config =v_fct_config 
        self._pi_fct_config = pi_fct_config
        self._save_path = save_path
        self.alpha = 'auto'
        self.target_entropy = 'auto'

        self.env = env

        self.global_step =  tf.train.get_or_create_global_step()
        
        self._prep_train()   
        
        
        if os.path.isfile(self._save_path+".meta") :
           

            self._saver = tf.train.Saver()
            self._saver.restore(self._sess, self._save_path)
            print("restored")           
        else: 
            
            
            self._sess.run(tf.global_variables_initializer())
            self._saver = tf.train.Saver()
            
        self._init_update_target_V()
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step+1)
        self._sess.run(self._update_target_V_ops_hard)
            
        
    def _init_update_target_V(self):
        source_params = self._vars("V_func")
        target_params = self._vars("V_target")    
        self._update_target_V_ops = [
            tf.assign(target, (1 - self._config["tau"]) * target + self._config["tau"] * source)
            for target, source in zip(target_params, source_params)
        ]  
        self._update_target_V_ops_hard = [
            tf.assign(target, source)
            for target, source in zip(target_params, source_params)
        ]    
   
    def _vars(self, scope=''):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope + '/' + scope)
    
    def _global_vars(self, scope=''):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scope + '/' + scope)
    
    def action(self, observation):
        fddct = {self.obs : observation}
        actions = self._sess.run([self._Policy.act], feed_dict=fddct)
        return actions[0]
    def act_greedy(self, observation):
        fddct = {self.obs : observation}
        actions = self._sess.run([self._Policy.mu_tanh], feed_dict=fddct)
        #actions = self._sess.run([self._Policy.mu], feed_dict=fddct)
        
        return actions[0]
    
    
    def store_transition(self, ob, a, reward, ob_new, done):
        self.buffer.add_item([ob,a,reward,ob_new, done])

    def reverse_action(self, action):
        low  = self._a_space.low
        high = self._a_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action
        

    #################################################################################################
    #################################################################################################
    def _prep_train(self):
        
        self.obs = tf.placeholder(dtype=tf.float32, shape=(None, self._config["dim_obs"]), name="obs")        
        self.act = tf.placeholder(dtype=tf.float32, shape=(None,self._config["dim_act"]), name="act")  
        self.rew = tf.placeholder(dtype=tf.float32, shape=(None,1), name="rew") 
        self.obs_new = tf.placeholder(dtype=tf.float32, shape=(None, self._config["dim_obs"]), name="obs_new") 
        self.done = tf.placeholder(dtype=tf.float32, shape=(None,1), name="done") 


        
        # if version is 12 or 11: uncomment
        """
        if self.target_entropy == 'auto':
            self.target_entropy = -self._config["dim_act"]
        else:
            self.target_entropy = self.target_entropy.astype(np.float32)

        if self.alpha == 'auto':
            inital_alpha = self._config["initial_alpha"]
            self.train_alpha = True
            self.log_alpha = tf.get_variable(name='log_alpha', dtype=tf.float32, initializer=np.log(inital_alpha).astype(np.float32))
            self.alpha = tf.exp(self.log_alpha)
        else:
            self.alpha = self.alpha.astype(np.float32)
        """
        


        
        with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
            
            if self.target_entropy == 'auto':
                self.target_entropy = -self._config["dim_act"]
            else:
                self.target_entropy = self.target_entropy.astype(np.float32)

            if self.alpha == 'auto':
                inital_alpha = self._config["initial_alpha"]
                self.train_alpha = True
                self.log_alpha = tf.get_variable(name='log_alpha', dtype=tf.float32, initializer=np.log(inital_alpha).astype(np.float32))
                self.alpha = tf.exp(self.log_alpha)
            else:
                self.alpha = self.alpha.astype(np.float32)
            

            # Q-function network
            self._Q1 = self._v_fct(inp = tf.concat([self.obs,self.act], axis=-1), scope = "Q_func1", **self._q_fct_config)
            self._Q2 = self._v_fct(inp = tf.concat([self.obs,self.act], axis=-1), scope = "Q_func2", **self._q_fct_config) 
            # Policy Network
            self._Policy = self._pi_fct(inp=self.obs,  scope = "Policy", **self._pi_fct_config)
            # Value-function network
            self._V        = self._v_fct(inp = self.obs, scope = "V_func", **self._v_fct_config)
            self._V_target = self._v_fct(inp = self.obs_new,  scope = "V_target", **self._v_fct_config)
        
        with tf.variable_scope(self._scope, reuse=True):     
            self._Q1_pi = self._v_fct( inp = tf.concat([self.obs,self._Policy.act], axis=-1), scope = "Q_func1", **self._q_fct_config)
            self._Q2_pi = self._v_fct( inp = tf.concat([self.obs,self._Policy.act], axis=-1), scope = "Q_func2", **self._q_fct_config) 
            
        self.log_prob_new_act = self._Policy.log_prob        
        self.log_prob_prior =0.0
        
        self.min_Q1_Q2 = tf.minimum(self._Q1_pi.output,self._Q2_pi.output)                                
        self.y_v = tf.stop_gradient(self.min_Q1_Q2 - self.alpha* (self.log_prob_new_act + self.log_prob_prior))
        
        # Q update
        self.Q1_loss = 0.5 *  tf.reduce_mean((tf.stop_gradient(self.rew + self._config["discount"] * (1 - self.done) * self._V_target.output)-self._Q1.output )**2)
        self.Q2_loss = 0.5 *  tf.reduce_mean((tf.stop_gradient(self.rew + self._config["discount"] * (1 - self.done) * self._V_target.output)-self._Q2.output )**2)
        
        self.Q1optim = tf.train.AdamOptimizer(learning_rate=self._config["lambda_Q"],name='AdamQ1')
        self._train_opQ1 = self.Q1optim.minimize(loss= self.Q1_loss, var_list= self._vars(self._Q1._scope), name='AdamQ1_min') #########################
        
        self.Q2optim = tf.train.AdamOptimizer(learning_rate=self._config["lambda_Q"],name='AdamQ2')
        with tf.control_dependencies([self._train_opQ1]):       
            self._train_opQ2 = self.Q2optim.minimize(loss=self.Q2_loss, var_list= self._vars(self._Q2._scope), name='AdamQ2_min') #########################
        
        # V update 
        self.V_loss = 0.5 * tf.reduce_mean((self._V.output - self.y_v)**2)
        self.Voptim = tf.train.AdamOptimizer(learning_rate=self._config["lambda_V"],name='AdamV')
        with tf.control_dependencies([self._train_opQ2 ]):      
            self._train_opV = self.Voptim.minimize(loss= self.V_loss, var_list= self._vars(self._V._scope),name='AdamV_min') #########################

        # PI update
        self.PI_loss_KL = tf.reduce_mean(self.alpha* self.log_prob_new_act - self._Q1_pi.output)
        self.policy_regularization_loss = 0.001 * 0.5 * (tf.reduce_mean(self._Policy.log_std**2)+tf.reduce_mean(self._Policy.mu ** 2))
        self.PI_loss =   self.PI_loss_KL + self.policy_regularization_loss
        
        self.PIoptim = tf.train.AdamOptimizer(learning_rate=self._config["lambda_Pi"],name='AdamPi')
        with tf.control_dependencies([self._train_opV]):      
            self._train_opPI = self.PIoptim.minimize(loss= self.PI_loss, var_list=self._vars(self._Policy._scope),name='AdamPi_min')  #########################

        if self.train_alpha:
            with tf.control_dependencies([self._train_opPI]):
                self.alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(self.log_prob_new_act + self.target_entropy))
                self.alpha_optimizer = tf.train.AdamOptimizer(learning_rate=self._config["lambda_Alpha"], name='AdamAlpha')       
                self._train_opAlpha = self.alpha_optimizer.minimize(self.alpha_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._scope+"/log_alpha"),name='AdamAlpha_min')
                #self._train_opAlpha = self.alpha_optimizer.minimize(self.alpha_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="log_alpha"),name='AdamAlpha_min')
    
    def _train(self,  update_value_target):
        # Sample from replay buffer
        # Extract states, actions, ...


        batch = self.buffer.sample_batch(self._config["batch_size"])
        fddct = {self.obs: batch['obs1'],
                    self.act: batch['acts'],
                    self.rew: batch['rews'].reshape(-1, 1),
                    self.obs_new: batch['obs2'],
                    self.done: batch['done'].reshape(-1, 1),
                    }
        

         
        train_ops = [self._train_opPI, self.PI_loss,
                     self._train_opV, self.V_loss,
                     self._train_opQ1, self.Q1_loss,
                     self._train_opQ2, self.Q2_loss,
                     self._update_target_V_ops,
                     self.alpha_loss, self._train_opAlpha]
        _, loss_PI_fct,_, loss_V_fct,_, loss_Q1_fct,_, loss_Q2_fct,_ ,_,_= self._sess.run(train_ops, feed_dict=fddct)
        
        
        return loss_V_fct, loss_Q1_fct,loss_Q2_fct,loss_PI_fct
        
    
    
    def train(self, iter_fit=1000, max_steps= 500, env_steps = 1, grad_steps = 1, burn_in =1000):
        bar = progressbar.ProgressBar(max_value=iter_fit)
        # Initilalize target V network
        self._sess.run(self._update_target_V_ops_hard)
        # Init Statistics      
        total_rewards_per_episode = [] 
        total_loss_V = []
        total_loss_Q1 = []
        total_loss_Q2 = []
        total_loss_PI = []
        
        
        j = 0
        for i in range(iter_fit):
            
            ob = self.env.reset() 
            total_reward = 0
            for _ in range(max_steps):
                for e_i in range(env_steps):#
                    if j < burn_in:
                        a = self.env.action_space.sample()
                        a = self.reverse_action(a)    
                                        
                    else:
                        a = self.action(np.asarray(ob).reshape(1, self._o_space.shape[0]))
                        a = a[0]
                        
                    (ob_new, reward, done, _info) = self.env.step(a)
                    total_reward += reward
                    self.buffer.store(ob, a, reward, ob_new, done)
                    ob=ob_new
  
                if j  >= self._config["batch_size"]:
                    for g_i in range(grad_steps):
                        update_value_target = False
                        if i % self._config["target_update"] == 0:
                            update_value_target = True
                        loss_V_fct, loss_Q1_fct,loss_Q2_fct,loss_PI_fct= self._train(update_value_target)
                        total_loss_V.append(loss_V_fct)
                        total_loss_Q1.append(loss_Q1_fct)
                        total_loss_Q2.append(loss_Q2_fct)
                        total_loss_PI.append(loss_PI_fct)        
                j += 1
                if done:
                    break   
            total_rewards_per_episode.append(total_reward)
            if i % 1 == 0:
                plot(total_rewards_per_episode,total_loss_V,total_loss_Q1, total_loss_Q2, total_loss_PI) 
            bar.update(i)   
        self._saver.save(self._sess, self._save_path, self.global_step)

        return total_rewards_per_episode

