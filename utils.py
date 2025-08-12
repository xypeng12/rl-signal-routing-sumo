import itertools
import logging
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import pandas as pd


def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True

import os
import shutil

def copy_file(src_dir, tar_dir, suffix='.ini'):

    os.makedirs(tar_dir, exist_ok=True)

    src = src_dir
    if os.path.isdir(src_dir):
        src = find_file(src_dir, suffix)
        if src is None:
            raise FileNotFoundError(f'Cannot find {suffix} under {src_dir}')

    shutil.copy(src, tar_dir)

def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' % (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


def plot_train(data_dirs, labels):
    pass

def plot_evaluation(data_dirs, labels):
    pass


class Counter:
    def __init__(self, total_step,test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


class Trainer():
    def __init__(self, env, model, problem,global_counter, summary_writer, run_test, output_path=None):
        self.cur_step=0
        self.global_counter = global_counter
        self.problem=problem
        self.env = env
        self.agent = self.env.agent
        self.model = model
        self.sess = self.model.sess
        self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        self.run_test = run_test
        assert self.env.T % self.n_step == 0
        self.data = []
        self.agent_reward=[]
        self.output_path = output_path
        if run_test:
            self.test_num = self.env.test_num
            logging.info('Testing: total test num: %d' % self.test_num)
        self._init_summary()

    def _init_summary(self):
        self.train_reward = tf.placeholder(tf.float32, [])
        self.train_summary = tf.summary.scalar('train_reward', self.train_reward)
        self.test_reward = tf.placeholder(tf.float32, [])
        self.test_summary = tf.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            summ = self.sess.run(self.train_summary, {self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def explore(self, prev_ob,prev_done):
        ob = prev_ob
        done = prev_done
        rewards = 0
        total_arrived_number = 0
        total_departed_number = 0
        total_waiting_time = 0
        total_travel_time = 0
        agent_reward=[]
        for i in range(self.n_step):

            policy, value = self.model.forward(ob, done)
            # need to update fingerprint before calling step
            self.env.update_fingerprint(policy)
            action = []
            for pi in policy:
                action.append(np.random.choice(np.arange(len(pi)), p=pi))
            next_ob, reward, done, global_reward,arrived_number, departed_number,waiting_time,travel_time,agent_reward0 = self.env.step(action)
            total_arrived_number += arrived_number
            total_departed_number += departed_number
            total_waiting_time += waiting_time
            total_travel_time += travel_time

            rewards += global_reward
            #if agent_reward==[]:
            if agent_reward is None or np.size(agent_reward) == 0:
                agent_reward=np.array(agent_reward0)
            else:
                agent_reward += np.array(agent_reward0)

            global_step = self.global_counter.next()
            self.cur_step +=1


            self.model.add_transition(ob, action, reward, value, done)
            if self.global_counter.should_log():
                logging.info('''Training: global step %d, current step %d, r: %.2f, train r: %.2f, done: %r''' %
                         (global_step, self.cur_step, global_reward, np.mean(reward), done))

            if done:
                break
            ob = next_ob

        if done:
            R = [0] * self.model.n_agent
        else:
            R = self.model.forward(ob, False, 'v')

        return ob, done, R, rewards,agent_reward,total_arrived_number, total_departed_number,total_waiting_time,total_travel_time

    def perform(self, test_ind, demo=True, policy_type='default',load=False):
        ob = self.env.reset(gui=demo, test_ind=test_ind,load=load)
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        rewards = 0
        total_arrived_number = 0
        total_departed_number = 0
        total_waiting_time = 0
        total_travel_time = 0

        total_agent_reward=[]
        while True:
            policy = self.model.forward(ob, done, 'p')
            self.env.update_fingerprint(policy)
            action = []
            for pi in policy:
                if policy_type != 'deterministic':
                    action.append(np.random.choice(np.arange(len(pi)), p=pi))
                else:
                    action.append(np.argmax(np.array(pi)))

            next_ob, reward, done, global_reward, arrived_number, departed_number,waiting_time,travel_time,agent_reward0 = self.env.step(action)

            total_arrived_number += arrived_number
            total_departed_number += departed_number
            total_waiting_time += waiting_time
            total_travel_time += travel_time

            rewards += global_reward
            if total_agent_reward==[]:
                total_agent_reward=np.array(agent_reward0)
            else:
                total_agent_reward += np.array(agent_reward0)

            if done:
                break
            ob = next_ob

        return rewards,total_agent_reward,total_arrived_number, total_departed_number,total_waiting_time,total_travel_time

    def run(self):
        while not self.global_counter.should_stop():
            if self.run_test and self.global_counter.should_test():
                rewards = 0
                total_arrived_number = 0
                total_departed_number = 0
                total_waiting_time = 0
                total_travel_time = 0

                total_agent_reward=[]
                global_step = self.global_counter.cur_step
                self.env.train_mode = False
                for test_ind in range(self.test_num):
                    cur_rewards, _,arrived_number, departed_number,waiting_time,travel_time= self.perform(test_ind)
                    self.env.terminate()
                    rewards+=cur_rewards

                    total_arrived_number += arrived_number
                    total_departed_number += departed_number
                    total_waiting_time += waiting_time
                    total_travel_time += travel_time

                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': rewards,
                            'total arrived vehicle number': total_arrived_number,
                           'total departured vehicle number': total_departed_number,
                           'total waiting time': total_waiting_time,
                           'total travel time': total_travel_time,
                           'average waiting time': total_waiting_time / total_departed_number,
                           'average travel time': total_travel_time / total_departed_number}

                    self.data.append(log)
                self._add_summary(rewards, global_step, is_train=False)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, rewards))

            # train
            self.env.train_mode = True
            ob = self.env.reset()

            done=True
            self.model.reset()

            rewards = 0
            total_arrived_number=0
            total_departed_number=0
            total_waiting_time =0
            total_travel_time = 0
            total_agent_reward=[]
            self.cur_step = 0
            while True:
                ob, done, R, cur_rewards,agent_reward,arrived_number, departed_number,waiting_time,travel_time = self.explore(ob,done)
                global_step = self.global_counter.cur_step
                rewards += cur_rewards
                total_arrived_number += arrived_number
                total_departed_number += departed_number
                total_waiting_time+=waiting_time
                total_travel_time+=travel_time

                if total_agent_reward is None or np.size(total_agent_reward) == 0:
                    total_agent_reward=np.array(agent_reward)
                else:
                    total_agent_reward+=np.array(agent_reward)

                self.model.backward(R, self.summary_writer, global_step)
                # termination
                if done:
                    #到达3600s
                    #结束 关闭SUMO文件
                    self.env.terminate()
                    break

            log = {'agent': self.agent,
                   'step': global_step,
                   'test_id': -1,
                   'reward': rewards,
                   'total arrived vehicle number':total_arrived_number,
                   'total departured vehicle number': total_departed_number,
                   'total waiting time':total_waiting_time,
                   'total travel time': total_travel_time,
                   'average waiting time':total_waiting_time/total_departed_number,
                   'average travel time': total_travel_time / total_departed_number
                   }

            self.data.append(log)
            self.agent_reward.append(total_agent_reward)

            self._add_summary(rewards, global_step)
            self.summary_writer.flush()
        df = pd.DataFrame(self.data)
        path = self.output_path.replace('data', self.problem)
        df.to_csv(path + 'train_reward.csv')
        np.savetxt(path + 'agent_reward.csv', self.agent_reward, delimiter=',')


class Tester(Trainer):
    def __init__(self, env, model,problem,global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.problem=problem
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def _init_summary(self):
        self.reward = tf.placeholder(tf.float32, [])
        self.summary = tf.summary.scalar('test_reward', self.reward)

class Evaluator(Tester):
    def __init__(self, env, model, output_path, demo=False, policy_type='default'):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.demo = demo
        self.policy_type = policy_type

    def run(self):
        is_record = True
        self.env.cur_episode = 0
        self.env.init_data(is_record, self.output_path)
        time.sleep(1)

        for test_ind in range(self.test_num):
            reward, _,_,_,_,_ = self.perform(test_ind, demo=True, policy_type=self.policy_type)
            self.env.terminate()
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()

    def run_diff_com(self):
        numbers = np.arange(0, 1.1, 0.1)
        for i in numbers:
            self.env.CAV_rate=i

            folder_path="Sioux/%s/eva_data/%.1f/" % (self.env.problem, i)
            os.mkdir(folder_path)

            is_record = True
            self.env.cur_episode = 0
            self.env.init_data(is_record, self.output_path,diff=i)
            time.sleep(1)
            for test_ind in range(self.test_num):
                reward, _, _, _, _, _ = self.perform(test_ind, demo=False, policy_type=self.policy_type)
                self.env.terminate()
                logging.info('test %i, avg reward %.2f' % (test_ind, reward))
                time.sleep(2)
                self.env.collect_tripinfo()
            self.env.output_data()

    def run_diff_load(self):
        numbers = np.arange(0.2, 2.2, 0.2)
        for i in numbers:

            folder_path="Sioux/%s/eva_data/%.1f/" % (self.env.problem, i)
            os.mkdir(folder_path)

            is_record = True
            self.env.cur_episode = 0
            self.env.init_data(is_record, self.output_path,diff=i)
            time.sleep(1)
            for test_ind in range(self.test_num):
                reward, _, _, _, _, _ = self.perform(test_ind, demo=False, policy_type=self.policy_type,load=i)
                self.env.terminate()
                logging.info('test %i, avg reward %.2f' % (test_ind, reward))
                time.sleep(2)
                self.env.collect_tripinfo()
            self.env.output_data()

class Evaluator_separate(Evaluator):
    def __init__(self, env, model_signal,model_route, output_path, demo=False, policy_type='default'):
        self.env = env
        self.model_signal = model_signal
        self.model_route=model_route

        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.demo = demo
        self.policy_type = policy_type

    def run(self):
        is_record = True
        self.env.cur_episode = 0
        self.env.init_data(is_record, self.output_path)
        time.sleep(1)

        for test_ind in range(self.test_num):
            self.perform(test_ind, demo=True, policy_type=self.policy_type)
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        self.env.output_data()

    def perform(self, test_ind, demo=True, policy_type='default',load=False):
        ob = self.env.reset(gui=demo, test_ind=test_ind,load=load,separate=True)
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model_signal.reset()
        self.model_route.reset()
        while True:
            ob_signal=ob[:17]
            ob_route=ob[17:]

            policy_signal = self.model_signal.forward(ob_signal, done, 'p')
            policy_route=self.model_route.forward(ob_route, done, 'p')
            policy=policy_signal+policy_route

            self.env.update_fingerprint(policy)
            action_signal = []
            for pi in policy_signal:
                if policy_type != 'deterministic':
                    action_signal.append(np.random.choice(np.arange(len(pi)), p=pi))
                else:
                    action_signal.append(np.argmax(np.array(pi)))

            action_route = []
            for pi in policy_route:
                if policy_type != 'deterministic':
                    action_route.append(np.random.choice(np.arange(len(pi)), p=pi))
                else:
                    action_route.append(np.argmax(np.array(pi)))

            next_ob, done= self.env.step_separate(action_signal,action_route)

            if done:
                break
            ob=next_ob
