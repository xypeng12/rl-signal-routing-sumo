import os
from agents.utils import *
from agents.policies import *
import logging
import multiprocessing as mp
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class A2C:
    def __init__(self, n_s, n_a, total_step, model_config, seed=0, n_f=None):
        # load parameters
        self.name = 'a2c'
        self.n_agent = 1
        self.problem =None

        self.n_s = n_s
        self.n_a = n_a
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy = self._init_policy(n_s, n_a, n_f, model_config)
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_policy(self, n_s, n_a, n_w, n_f, model_config,n_o=0,agent_type=None, agent_name=None):
        n_lstm = model_config.getint('num_lstm')
        if self.problem=='signal':
            n_fw_s = model_config.getint('s_num_fw')
            n_ft_s = model_config.getint('s_num_ft')
            n_fp_s = model_config.getint('s_num_fp')
            entropy=model_config.getfloat('entropy_coef_s')
            policy = FPLstmACPolicy(n_s, n_a, n_w, n_f, self.n_step, n_fc_wave=n_fw_s,
                                    n_fc_wait=n_ft_s, n_fc_fp=n_fp_s,n_lstm=n_lstm, name=agent_name,entropy_coef=entropy)
        elif self.problem == 'route':
            n_fw_r = model_config.getint('r_num_fw')
            n_ft_r = model_config.getint('r_num_ft')
            n_fp_r = model_config.getint('r_num_fp')
            entropy = model_config.getfloat('entropy_coef_r')
            policy = FPLstmACPolicy(n_s, n_a, n_w, n_f, self.n_step, n_fc_wave=n_fw_r,
                                    n_fc_wait=n_ft_r, n_fc_fp=n_fp_r, n_lstm=n_lstm, name=agent_name,entropy_coef=entropy)
        if self.problem=='signal_route':
            n_fw_s = model_config.getint('num_fw_s')
            n_ft_s = model_config.getint('num_ft_s')
            n_fo_s = model_config.getint('num_fo_s')
            n_fp_s = model_config.getint('num_fp_s')

            n_fw_r = model_config.getint('num_fw_r')
            n_ft_r = model_config.getint('num_ft_r')
            n_fo_r = model_config.getint('num_fo_r')
            n_fp_r = model_config.getint('num_fp_r')
            entropy_s = model_config.getfloat('entropy_coef_s')
            entropy_r = model_config.getfloat('entropy_coef_r')
            if agent_type=='signal':
                #agent=signal
                policy = FPLstmACPolicySR(n_s, n_a, n_w,n_o, n_f, self.n_step, n_fc_wave=n_fw_s,
                                    n_fc_wait=n_ft_s, n_fc_other=n_fo_s,n_fc_fp=n_fp_s,n_lstm=n_lstm, name=agent_name,entropy_coef=entropy_s)
            else:
                #agent=route
                policy = FPLstmACPolicySR(n_s, n_a, n_w, n_o, n_f, self.n_step, n_fc_wave=n_fw_r,
                                         n_fc_wait=n_ft_r, n_fc_other=n_fo_r, n_fc_fp=n_fp_r,n_lstm=n_lstm, name=agent_name,entropy_coef=entropy_r)

        if self.name != 'ma2c':
            policy = LstmACPolicy(n_s, n_a, n_w, self.n_step, n_fc_wave=n_fw_s,
                                  n_fc_wait=n_ft_s,n_lstm=n_lstm, name=agent_name)
        return policy

    def _init_scheduler(self, model_config):
        lr_init = model_config.getfloat('lr_init')
        lr_decay = model_config.get('lr_decay')

        if lr_decay == 'constant':
            self.lr_scheduler = Scheduler(lr_init, decay=lr_decay)
        else:
            lr_min = model_config.getfloat('LR_MIN')
            self.lr_scheduler = Scheduler(lr_init, lr_min, self.total_step, decay=lr_decay)


    def _init_train(self, model_config):
        # init loss
        entropy_coef=model_config.getfloat('entropy_coef')
        max_grad_norm = model_config.getfloat('max_grad_norm')
        adam_beta1=model_config.getfloat('adam_beta1')
        adam_beta2=model_config.getfloat('adam_beta2')
        adam_epsilon = model_config.getfloat('adam_epsilon')
        self.policy.prepare_loss(max_grad_norm, adam_beta1,adam_beta2, adam_epsilon)

        # init replay buffer
        gamma = model_config.getfloat('gamma')
        self.trans_buffer = OnPolicyBuffer(gamma)

    def save(self, model_dir, global_step):
        self.saver.save(self.sess, model_dir + 'checkpoint', global_step=global_step)

    def load(self, model_dir, checkpoint=None):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if checkpoint is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        prefix = file.split('.')[0]
                        tokens = prefix.split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = prefix
                            save_step = cur_step
            else:
                save_file = 'checkpoint-' + str(int(checkpoint))
        if save_file is not None:
            self.saver.restore(self.sess, model_dir + save_file)
            logging.info('Checkpoint loaded: %s' % save_file)
            return True
        logging.error('Can not find old checkpoint for %s' % model_dir)
        return False

    def reset(self):
        self.policy._reset()

    def backward(self, R, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        obs, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(R)
        self.policy.backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr,
                             summary_writer=summary_writer, global_step=global_step)

    def forward(self, ob, done, out_type='pv'):
        return self.policy.forward(self.sess, ob, done, out_type)

    def add_transition(self, ob, action, reward, value, done):
        # Hard code the reward norm for negative reward only
        self.trans_buffer.add_transition(ob, action, reward, value, done)

class IA2C(A2C):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, total_step,
                 model_config, seed=0):
        self.name = 'ia2c'
        self.agents = []
        self.n_agent = len(n_s_ls)

        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        for i, (n_s, n_w, n_a) in enumerate(zip(self.n_s_ls, self.n_w_ls, self.n_a_ls)):
            # agent_name is needed to differentiate multi-agents
            self.policy_ls.append(self._init_policy(n_s - n_w, n_a, n_w, 0, model_config,
                                  agent_name='{:d}a'.format(i)))
        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())

    def _init_train(self, model_config):
        # init loss
        max_grad_norm = model_config.getfloat('max_grad_norm')
        entropy_coef=model_config.getfloat('entropy_coef')
        adam_beta1 = model_config.getfloat('adam_beta1')
        adam_beta2 = model_config.getfloat('adam_beta2')
        adam_epsilon = model_config.getfloat('adam_epsilon')

        gamma = model_config.getfloat('gamma')
        self.trans_buffer_ls = []
        for i in range(self.n_agent):
            self.policy_ls[i].prepare_loss(max_grad_norm, adam_beta1, adam_beta2, adam_epsilon)
            self.trans_buffer_ls.append(OnPolicyBuffer(gamma))

    def backward(self, R_ls, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)
        for i in range(self.n_agent):
            obs, acts, dones, Rs, Advs = self.trans_buffer_ls[i].sample_transition(R_ls[i])
            if i == 0:
                self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr,
                                           summary_writer=summary_writer, global_step=global_step)
            else:
                self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr)

    def forward(self, obs, done, out_type='pv'):
        if len(out_type) == 1:
            out = []
        elif len(out_type) == 2:
            out1, out2 = [], []
        for i in range(self.n_agent):
            # policy, value
            cur_out = self.policy_ls[i].forward(self.sess, obs[i], done, out_type)
            if len(out_type) == 1:
                out.append(cur_out)
            else:
                out1.append(cur_out[0])
                out2.append(cur_out[1])
        if len(out_type) == 1:
            return out
        else:
            #policy, value
            return out1, out2

    def backward_mp(self, R_ls, summary_writer=None, global_step=None):
        cur_lr = self.lr_scheduler.get(self.n_step)

        def worker(i):
            obs, acts, dones, Rs, Advs = self.trans_buffer_ls[i].sample_transition(R_ls[i])
            self.policy_ls[i].backward(self.sess, obs, acts, dones, Rs, Advs, cur_lr,
                                       summary_writer=summary_writer, global_step=global_step)
        mps = []
        for i in range(self.n_agent):
            p = mp.Process(target=worker, args=(i))
            p.start()
            mps.append(p)
        for p in mps:
            p.join()

    def reset(self):
        for policy in self.policy_ls:
            policy._reset()

    def add_transition(self, obs, actions, rewards, values, done):
        for i in range(self.n_agent):
            self.trans_buffer_ls[i].add_transition(obs[i], actions[i],
                                                   rewards[i], values[i], done)

class MA2C(IA2C):
    def __init__(self, n_s_ls, n_a_ls, n_w_ls, n_f_ls,total_step,
                 model_config,problem,agent_type=None ,n_o_ls=None,seed=0):
        self.name = 'ma2c'
        self.agents = []
        self.n_agent = len(n_s_ls)
        self.problem = problem
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.n_f_ls = n_f_ls
        self.n_w_ls = n_w_ls
        self.n_step = model_config.getint('batch_size')
        # init tf
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)
        self.policy_ls = []
        if self.problem=='signal_route':
            self.n_o_ls=n_o_ls
            self.agent_type = agent_type
            for i, (n_s, n_a, n_w, n_o,n_f,a_t) in enumerate(zip(self.n_s_ls, self.n_a_ls, self.n_w_ls, self.n_o_ls,self.n_f_ls,self.agent_type)):
                # agent_name is needed to differentiate multi-agents
                self.policy_ls.append(self._init_policy(n_s, n_a, n_w,n_f, model_config,n_o=n_o,agent_type=a_t,
                                                        agent_name='{:d}a'.format(i)))
        else:
            for i, (n_s, n_a, n_w, n_f) in enumerate(zip(self.n_s_ls, self.n_a_ls, self.n_w_ls, self.n_f_ls)):
                # agent_name is needed to differentiate multi-agents
                self.policy_ls.append(self._init_policy(n_s, n_a, n_w, n_f, model_config,
                                                        agent_name='{:d}a'.format(i)))

        self.saver = tf.train.Saver(max_to_keep=5)
        if total_step:
            # training
            self.total_step = total_step
            self._init_scheduler(model_config)
            self._init_train(model_config)
        self.sess.run(tf.global_variables_initializer())