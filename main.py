import configparser
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import threading

from envs.Sioux_env import SiouxEnv, SiouxController
from agents.models import MA2C
from utils import *

def init_env(config, port=0, naive_policy=False):
    if not naive_policy:
        return SiouxEnv(config, problem, port=port)
    else:
        env = SiouxEnv(config, problem, port=port)
        policy = SiouxController(env.node_names, env.rc_names)
        return env, policy


def get_parameter(scenario):
    config_dir = default_config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)
    env = SiouxEnv(config['ENV_CONFIG'],problem, port=0)
    env.get_parameter(scenario)
    env.terminate()

def train():
    base_dir = default_base_dir
    dirs = init_dir(base_dir)
    store_dir=default_store_dir
    dirs2=init_dir(store_dir)
    init_log(dirs2['log'])
    config_dir = default_config_dir
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)
    test_mode='no_test'
    in_test, post_test = init_test_flag(test_mode)

    # init env
    env = init_env(config['ENV_CONFIG'])
    logging.info('Training: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r' %
                 (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls))

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init centralized or multi agent
    seed = config.getint('ENV_CONFIG', 'seed')

    if problem == 'signal_route':
        model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, total_step,
                     config['MODEL_CONFIG'], problem, agent_type=env.agent_type, n_o_ls=env.n_o_ls, seed=seed)
    else:
        model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, total_step,
                     config['MODEL_CONFIG'], problem, seed=seed)


    summary_writer =tf.summary.FileWriter(dirs2['log'])
    trainer = Trainer(env, model,problem, global_counter, summary_writer, in_test, output_path=dirs['data'])
    trainer.run()

    # post-training test
    if post_test:
        tester = Tester(env, model, problem,global_counter, summary_writer, dirs['data'])
        tester.run_offline(dirs['data'])

    # save model
    final_step = global_counter.cur_step
    logging.info('Training: save final model at step %d ...' % final_step)
    model.save(dirs2['model'], final_step)


def evaluate_fn(agent, output_dir, seeds, port, demo, policy_type):
    # load config file for env
    config_dir = default_config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env, greedy_policy = init_env(config['ENV_CONFIG'], port=port, naive_policy=True)
    logging.info('Evaluation: s dim: %d, a dim %d, s dim ls: %r, a dim ls: %r' %
                 (env.n_s, env.n_a, env.n_s_ls, env.n_a_ls))
    env.init_test_seeds(seeds)
    if separate_train==True:
        n_s_ls=[46, 36, 14, 22, 38, 38, 30, 22, 28, 34, 28, 30, 12, 14, 14, 28, 24]
        n_a_ls=[5, 4, 2, 2, 4, 4, 2, 2, 2, 4, 2, 4, 2, 2, 2, 4, 2]
        n_w_ls=[10, 8, 6, 6, 8, 8, 6, 6, 6, 8, 6, 8, 6, 6, 6, 8, 6]
        n_f_ls=[11, 7, 3, 6, 9, 9, 8, 6, 7, 6, 7, 7, 1, 3, 3, 5, 7]
        model_signal = MA2C(n_s_ls, n_a_ls, n_w_ls, n_f_ls, 0,
                             config['MODEL_CONFIG'], 'signal', agent_type=env.agent_type, n_o_ls=env.n_o_ls)
        n_s_ls=[1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1]
        n_a_ls=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        n_w_ls=[6, 7, 10, 7, 6, 10, 7, 9, 7, 10, 6, 9]
        n_f_ls=[0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]
        model_route=MA2C(n_s_ls, n_a_ls, n_w_ls, n_f_ls, 0,
                             config['MODEL_CONFIG'], 'route', agent_type=env.agent_type, n_o_ls=env.n_o_ls)
        if not model_signal.load(default_base_dir+ '/signal/model/'):
            return
        if not model_route.load(default_base_dir+'/route/model/'):
            return
        env.agent=agent
        evaluator = Evaluator_separate(env, model_signal,model_route, output_dir, demo=demo, policy_type=policy_type)
        evaluator.run()
        return

    if problem == 'signal_route':
        model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, 0,
                             config['MODEL_CONFIG'], problem, agent_type=env.agent_type, n_o_ls=env.n_o_ls)
    else:
        model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, 0,
                             config['MODEL_CONFIG'], problem)

    if not model.load(default_store_dir + '/model/'):
        return

    env.agent = agent
    evaluator = Evaluator(env, model, output_dir, demo=demo, policy_type=policy_type)
    if diff==False:
        evaluator.run()
    else:
        if compliance==True:
            evaluator.run_diff_com()
        elif load==True:
            evaluator.run_diff_load()


def evaluate():
    store_dir = default_store_dir
    dirs = init_dir(store_dir, pathes=['eva_data', 'eva_log'])
    init_log(dirs['eva_log'])
    agents={'ma2c'}
    seeds = ','.join([str(i) for i in range(10000, 100001, 10000)])
    policy_type = 'default'
    logging.info('Evaluation: policy type: %s, random seeds: %s' % (policy_type, seeds))
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    threads = []
    for i, agent in enumerate(agents):
        demo=True
        thread = threading.Thread(target=evaluate_fn,
                                  args=(agent, dirs['eva_data'], seeds, i, demo, policy_type))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


default_config_dir = './config/config_ma2c_Sioux.ini'
scenario='Sioux'

option='train' #'evaluate','get_parameter'*3

separate_train=False

diff=False
compliance=False
load=False

problem='signal_route'
default_base_dir = scenario
if separate_train==True:
    default_store_dir = default_base_dir + '/signal_route(separate_training)'
else:
    default_store_dir=default_base_dir+ '/' + problem


if __name__ == '__main__':
    if option == 'train':
        train()
    elif option=='evaluate':
        evaluate()
    elif option=='get_parameter':
        get_parameter(scenario)