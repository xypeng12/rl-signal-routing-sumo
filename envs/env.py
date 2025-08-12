import logging
import numpy as np
import pandas as pd
import subprocess
from sumolib import checkBinary
import time
import traci
import random
import xml.etree.cElementTree as ET
import os

DEFAULT_PORT = 8000
SEC_IN_MS = 1000

REALNET_REWARD_NORM = 20


class RouteSet:
    def __init__(self, routes):
        self.num_route = len(routes)
        self.routes = routes


class PhaseSet:
    def __init__(self, phases):
        self.num_phase = len(phases)
        self.num_lane = len(phases[0])
        self.phases = phases

    @staticmethod
    def _get_phase_lanes(phase, signal='r'):
        phase_lanes = []
        for i, l in enumerate(phase):
            if l == signal:
                phase_lanes.append(i)
        return phase_lanes

    def _init_phase_set(self):
        self.red_lanes = []
        # self.green_lanes = []
        for phase in self.phases:
            self.red_lanes.append(self._get_phase_lanes(phase))
            # self.green_lanes.append(self._get_phase_lanes(phase, signal='G'))


class RouteMap:
    def __init__(self):
        self.routes = {}

    def get_route(self, rc_id, action):
        # route choice rate
        return self.routes[rc_id].routes

    def get_route_num(self, route_id):
        return self.routes[route_id].num_route


class PhaseMap:
    def __init__(self):
        self.phases = {}

    def get_phase(self, phase_id, action):
        # phase_type is either green or yellow
        return self.phases[phase_id].phases[int(action)]

    def get_phase_num(self, phase_id):
        return self.phases[phase_id].num_phase

    def get_lane_num(self, phase_id):
        # the lane number is link number
        return self.phases[phase_id].num_lane

    def get_red_lanes(self, phase_id, action):
        # the lane number is link number
        return self.phases[phase_id].red_lanes[int(action)]


class Node:
    def __init__(self, name, relevant_ss=[], relevant_sr={}, control=False):
        self.control = control  # disabled
        self.lanes_in = []
        self.ilds_in = []  # for state
        self.fingerprint = []  # local policy
        self.name = name
        self.relevant_ss = relevant_ss
        self.relevant_sr = relevant_sr
        self.num_state = 0  # wave and wait should have the same dim
        self.num_fingerprint = 0
        self.wave_state = []  # local state
        self.wait_state = []  # local state
        self.reward = 0
        self.wait = 0
        self.vehwait = 0

        # self.waits = []
        self.phase_id = -1
        self.n_a = 0
        self.prev_action = -1


class Rc:
    def __init__(self, name, f_id=None, relevant_rr=[], relevant_rs={}, up_edge=None, down_edge={}, forward_route=[],
                 det=[], control=False):
        self.control = control  # disabled
        self.fingerprint = []  # local policy
        self.up_edge = 0
        self.name = name
        self.f_id = f_id
        self.det = det
        self.relevant_rr = relevant_rr
        self.relevant_rs = relevant_rs
        self.down_edge = down_edge
        self.up_edge = up_edge
        self.forward_route = forward_route
        self.num_state_arrival = 0
        self.num_state_occupancy = 0
        self.num_fingerprint = 0
        self.arrived_vehicles = 0
        self.reward = 0
        self.wait = 0
        self.arrived = 0
        self.arrival_state = []  # 需要做路径选择的交通量 local state
        self.occupancy_state = []  # 路径沿线的交通量 local state

        self.route_id = -1
        self.n_a = 0
        self.prev_action = -1


class TrafficSimulator:
    def __init__(self, config, problem, output_path, is_record, port=0):
        self.name = config.get('scenario')
        self.seed = config.getint('seed')
        self.control_interval_sec = config.getint('control_interval_sec')
        self.episode_length_sec = config.getint('episode_length_sec')
        self.T = np.ceil(self.episode_length_sec / self.control_interval_sec)
        self.CAV_rate = config.getfloat('CAV_rate')
        self.port = DEFAULT_PORT + port
        self.sim_thread = port
        self.data_path = config.get('data_path')
        self.agent = config.get('agent')
        self.problem = problem
        self.coop_gamma_ss = config.getfloat('coop_gamma_ss')
        self.coop_gamma_sr = config.getfloat('coop_gamma_sr')
        self.coop_gamma_rr = config.getfloat('coop_gamma_rr')
        self.coop_gamma_rs = config.getfloat('coop_gamma_rs')
        self.cur_episode = 0
        # state
        self.norms = {'wave': config.getfloat('norm_wave'),
                      'wait': config.getfloat('norm_wait'),
                      'arrival': config.getfloat('norm_arrival'),
                      'occupancy': config.getfloat('norm_occupancy'),
                      }
        # reward
        if self.problem == 'signal':
            self.norms['reward_signal'] = config.getfloat('norm_reward_signal')
        elif self.problem == 'route':
            self.norms['reward_route'] = config.getfloat('norm_reward_route')
        elif self.problem == 'signal_route':
            self.norms['reward_signal'] = config.getfloat('norm_reward_sr_signal')
            self.norms['reward_route'] = config.getfloat('norm_reward_sr_route')

        self.clips = {'wave': config.getfloat('clip_wave'),
                      'wait': config.getfloat('clip_wait'),
                      'arrival': config.getfloat('clip_arrival'),
                      'occupancy': config.getfloat('clip_occupancy'),
                      'reward': config.getfloat('clip_reward')
                      }

        self.coef_rreward = config.getfloat('coef_rreward')
        self.coef_vehwait = config.getfloat('coef_vehwait')

        self.train_mode = True
        test_seeds = config.get('test_seeds').split(',')
        test_seeds = [int(s) for s in test_seeds]

        self._init_map()

        self.init_data(is_record, output_path)
        self.init_test_seeds(test_seeds)
        self._init_sim(self.seed)
        self._init_agents()
        self.terminate()

    def _debug_traffic_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase = self.sim.trafficlight.getRedYellowGreenState(self.node_names[0])
            cur_traffic = {'episode': self.cur_episode,
                           'time_sec': self.cur_sec,
                           'node': node_name,
                           'action': node.prev_action,
                           'phase': phase}
            for i, ild in enumerate(node.ilds_in):
                cur_name = 'lane%d_' % i
                cur_traffic[cur_name + 'queue'] = self.sim.lane.getLastStepHaltingNumber(ild)
                cur_traffic[cur_name + 'flow'] = self.sim.lane.getLastStepVehicleNumber(ild)
            self.traffic_data.append(cur_traffic)

    def _get_node_phase(self, action, node_name):
        node = self.nodes[node_name]
        cur_phase = self.phase_map.get_phase(node.phase_id, action)
        prev_action = node.prev_action
        node.prev_action = action
        if (prev_action < 0) or (action == prev_action):
            return cur_phase
        prev_phase = self.phase_map.get_phase(node.phase_id, prev_action)
        switch_reds = []
        switch_greens = []
        for i, (p0, p1) in enumerate(zip(prev_phase, cur_phase)):
            if (p0 in 'Gg') and (p1 == 'r'):
                switch_reds.append(i)
            elif (p0 in 'r') and (p1 in 'Gg'):
                switch_greens.append(i)
        if not len(switch_reds):
            return cur_phase
        yellow_phase = list(cur_phase)
        for i in switch_reds:
            yellow_phase[i] = 'y'
        for i in switch_greens:
            yellow_phase[i] = 'r'
        return ''.join(yellow_phase)

    def _get_node_phase_id(self, node_name):
        # needs to be overwriteen
        raise NotImplementedError()

    def _get_rc_state_num(self):
        raise NotImplementedError()

    def _get_rc_route_num(self, rc_name):
        # needs to be overwriteen
        raise NotImplementedError()

    def _get_rc_names(self):
        # needs to be overwriteen
        raise NotImplementedError()

    def _get_node_state_num(self, node):
        assert len(node.lanes_in) == self.phase_map.get_lane_num(node.phase_id)
        return len(node.ilds_in)

    def _get_reward(self, get_parameter=False):
        if get_parameter:
            norm_sreward = []
            norm_rreward = []
            norm_rarrived = []
            norm_swait = []
            norm_svehwait = []

        rewards_signal = []
        rewards_route = []

        for node_name in self.node_names:
            node = self.nodes[node_name]
            reward = node.reward
            if get_parameter:
                norm_sreward.append(reward)
                norm_swait.append(node.wait)
                norm_svehwait.append(node.vehwait)
            rewards_signal.append(reward)

        for rc_name in self.rc_names:
            rc = self.rcs[rc_name]
            reward = rc.reward
            if get_parameter:
                norm_rreward.append(reward)
                norm_rarrived.append(rc.arrived)
            rewards_route.append(self.coef_rreward * reward)

        if get_parameter:
            sreward_ave = np.mean(norm_sreward)
            rreward_ave = np.mean(norm_rreward)
            rarrived_ave = np.mean(norm_rarrived)
            swait_ave = np.mean(norm_swait)
            svehwait_ave = np.mean(norm_svehwait)
            reward_norm_ave = {'sreward': sreward_ave, 'rreward': rreward_ave, 'rarrived_ave': rarrived_ave,
                               'swait_ave': swait_ave, 'svehwait_ave': svehwait_ave}
            self.problem = 'signal'
            signal_rewards_ave = np.mean(self._local_reward_step(np.array(rewards_signal)))
            self.problem = 'route'
            route_rewards_ave = np.mean(self._local_reward_step(np.array(rewards_route)))
            self.problem = 'signal_route'
            rewards = self._local_reward_step(np.array(rewards_signal + rewards_route))
            sr_signal_rewards_ave = np.mean(rewards[:len(self.node_names)])
            sr_route_rewards_ave = np.mean(rewards[len(self.node_names):])
            total_rewards_ave = {'signal': signal_rewards_ave, 'route': route_rewards_ave,
                                 'sr_signal': sr_signal_rewards_ave,
                                 'sr_route': sr_route_rewards_ave}
            return reward_norm_ave, total_rewards_ave

        rewards = []
        if self.problem == 'signal_route':
            rewards = rewards_signal + rewards_route
        elif self.problem == 'signal':
            rewards = rewards_signal
        elif self.problem == 'route':
            rewards = rewards_route
        return np.array(rewards)

    def _get_state(self, reset=False, problem=None):
        if reset:
            self._measure_state_step()

        if problem is not None:
            self.problem = problem

        self._clip_state()
        state_signal = []
        state_route = []
        if self.problem == 'signal' or self.problem == 'signal_route':
            # hard code the state ordering as wave, wait, fp,(arrival)
            # get the appropriate state vectors
            for node_name in self.node_names:
                node = self.nodes[node_name]
                cur_state = [node.wave_state]
                # include wave states of relevant_ss
                for nnode_name in node.relevant_ss:
                    cur_state.append(self.nodes[nnode_name].wave_state)
                cur_state.append(node.wait_state)

                if self.problem == 'signal_route':
                    for rrc_name in node.relevant_sr:
                        cur_state.append(self.rcs[rrc_name].arrival_state)

                for nnode_name in node.relevant_ss:
                    fset = []
                    for f in self.nodes[nnode_name].fingerprint:
                        fset.append(f)
                    cur_state.append(fset)
                if self.problem == 'signal_route':
                    for rrc_name in node.relevant_sr:
                        fset = []
                        for f in self.rcs[rrc_name].fingerprint:
                            fset.append(f)
                        cur_state.append(fset)
                state_signal.append(np.concatenate(cur_state))
        if self.problem == 'route' or self.problem == 'signal_route':
            # hard code the state ordering as arrival, traffic, fp,(wave)
            for rc_name in self.rc_names:
                rc = self.rcs[rc_name]
                cur_state = [rc.arrival_state]
                for rrc_name in rc.relevant_rr:
                    cur_state.append(self.rcs[rrc_name].arrival_state)
                temp = 0
                for edge in rc.down_edge:
                    rc.occupancy_state[temp] = rc.occupancy_state[temp] * rc.down_edge[edge]
                    temp += 1
                cur_state.append(rc.occupancy_state)

                if self.problem == 'signal_route':
                    for nnode_name in rc.relevant_rs:
                        cur_state.append(self.nodes[nnode_name].wave_state)

                for rrc_name in rc.relevant_rr:
                    fset = []
                    for f in self.rcs[rrc_name].fingerprint:
                        fset.append(f)
                    cur_state.append(fset)
                if self.problem == 'signal_route':
                    for nnode_name in rc.relevant_rs:
                        fset = []
                        for f in self.nodes[nnode_name].fingerprint:
                            fset.append(f)
                        cur_state.append(fset)
                state_route.append(np.concatenate(cur_state))

        state = []
        if self.problem == 'signal_route':
            state = state_signal + state_route
        elif self.problem == 'signal':
            state = state_signal
        elif self.problem == 'route':
            state = state_route
        return state

    def _init_agents(self):
        nodes = {}
        for node_name in self.sim.trafficlight.getIDList():
            if node_name in self.relevant_ss_map:
                relevant_ss = self.relevant_ss_map[node_name]
            else:
                logging.info('node %s can not be found!' % node_name)
                relevant_ss = []

            if node_name in self.relevant_sr_map:
                relevant_sr = self.relevant_sr_map[node_name]
            else:

                logging.info('node %s can not be found!' % node_name)
                relevant_sr = []
            nodes[node_name] = Node(node_name,
                                    relevant_ss=relevant_ss,
                                    relevant_sr=relevant_sr,
                                    control=True)

            lanes_in = self.sim.trafficlight.getControlledLanes(node_name)
            nodes[node_name].lanes_in = lanes_in

            # controlled edges: e:j,i
            # lane ilds: ild:j,i_k for road ji, lane k.
            ilds_in = []
            for lane_name in lanes_in:
                ild_name = lane_name
                if ild_name not in ilds_in:
                    ilds_in.append(ild_name)
            nodes[node_name].ilds_in = ilds_in

        self.nodes = nodes
        self.node_names = sorted(list(nodes.keys()))

        self.rc_names = self._get_rc_names()
        rcs = {}
        for rc_name in self.rc_names:
            if rc_name in self.relevant_rr_map:
                relevant_rr = self.relevant_rr_map[rc_name]
            else:
                logging.info('rc %s can not be found!' % rc_name)
                relevant_rr = []

            if rc_name in self.relevant_rs_map:
                relevant_rs = self.relevant_rs_map[rc_name]
            else:
                logging.info('rc %s can not be found!' % rc_name)
                relevant_rs = []

            if rc_name in self.up_edge_map:
                up_edge = self.up_edge_map[rc_name]
            else:
                logging.info('rc %s can not be found!' % rc_name)
                up_edge = None

            if rc_name in self.down_edge_map:
                down_edge = self.down_edge_map[rc_name]
            else:
                logging.info('rc %s can not be found!' % rc_name)
                down_edge = {}

            if rc_name in self.forward_route_map:
                forward_route = self.forward_route_map[rc_name]
            else:
                logging.info('rc %s can not be found!' % rc_name)
                forward_route = []

            if rc_name in self.det_map:
                det = self.det_map[rc_name]
            else:
                logging.info('rc %s can not be found!' % rc_name)
                det = []

            if rc_name in self.f_id_map:
                f_id = self.f_id_map[rc_name]
            else:
                logging.info('rc %s can not be found!' % rc_name)
                f_id = []

            rcs[rc_name] = Rc(rc_name, f_id=f_id, relevant_rr=relevant_rr, relevant_rs=relevant_rs, up_edge=up_edge,
                              down_edge=down_edge, forward_route=forward_route, det=det, control=True)

        # 获取rc名
        self.rcs = rcs

        # 交叉口信息输出
        '''
        s = 'Env: init %d node information:\n' % len(self.node_names)
        for node in self.nodes.values():
            s += node.name + ':\n'
            s += '\trelevant_ss: %r\n' % node.relevant_ss
            s += '\tilds_in: %r\n' % node.ilds_in
        logging.info(s)
        '''
        self.agent_type = []
        if self.problem == 'signal_route':
            for i in range(len(self.node_names)):
                self.agent_type.append('signal')
            for i in range(len(self.rc_names)):
                self.agent_type.append('route')
        elif self.problem == 'signal':
            for i in range(len(self.node_names)):
                self.agent_type.append('signal')
        elif self.problem == 'route':
            for i in range(len(self.rc_names)):
                self.agent_type.append('route')

        self._init_action_space()
        self._init_state_space()

    def _init_action_space(self):
        # for local and relevant_ss coop level
        n_a_ls_signal = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            phase_id = self._get_node_phase_id(node_name)
            node.phase_id = phase_id
            node.n_a = self.phase_map.get_phase_num(phase_id)
            n_a_ls_signal.append(node.n_a)

        n_a_ls_route = []
        for rc_name in self.rc_names:
            rc = self.rcs[rc_name]
            route_num = self._get_rc_route_num(rc_name)
            rc.route_num = route_num
            rc.n_a = route_num
            n_a_ls_route.append(rc.n_a)

        self.n_a_ls = []
        if self.problem == 'signal_route':
            self.n_a_ls = n_a_ls_signal + n_a_ls_route
        elif self.problem == 'signal':
            self.n_a_ls = n_a_ls_signal
        elif self.problem == 'route':
            self.n_a_ls = n_a_ls_route

        # for global coop level
        self.n_a = np.prod(np.array(self.n_a_ls))

    def _init_map(self):
        # needs to be overwriteen
        self.phase_map = None
        self.route_map = None
        self.relevant_ss_map = None
        self.relevant_rs_map = None
        self.relevant_sr_map = None
        self.relevant_rr_map = None
        self.up_edge_map = None
        self.down_edge_map = None
        self.forward_route_map = None
        self.f_id_map = None
        self.det_map = None
        self.state_names_signal = None
        self.state_names_route = None

        raise NotImplementedError()

    def _init_policy(self):
        policy_signal = []
        policy_route = []
        for node_name in self.node_names:
            phase_num = self.nodes[node_name].n_a
            p = 1. / phase_num
            policy_signal.append(np.array([p] * phase_num))
        for rc_name in self.rc_names:
            route_num = self.rcs[rc_name].n_a
            p = 1. / route_num
            policy_route.append(np.array([p] * route_num))
        if self.problem == 'signal_route':
            policy = policy_signal + policy_route
        elif self.problem == 'signal':
            policy = policy_signal
        elif self.problem == 'route':
            policy = policy_route
        return policy

    def _init_sim(self, seed, gui=False, load=False):
        sumocfg_file = self._init_sim_config(seed, load)
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), '-c', sumocfg_file]
        command += ['--seed', str(seed)]
        command += ['--remote-port', str(self.port)]
        command += ['--no-step-log', 'True']
        if self.name != 'real_net':
            command += ['--time-to-teleport', '600']  # long teleport for safety
        else:
            command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        # collect trip info if necessary
        command += ['--tripinfo-output',
                    self.output_path + ('%s_%s_trip.xml' % (self.name, self.problem))]
        # command+=['--fcd-output', self.output_path+('%s_%s_sumoTrace.xml' % (self.name, self.problem))]
        subprocess.Popen(command)
        # wait 2s to establish the traci server
        time.sleep(2)
        self.sim = traci.connect(port=self.port)

    def _init_sim_config(self):
        # needs to be overwriteen
        raise NotImplementedError()

    def _init_sim_traffic(self):
        return

    def _init_state_space(self):
        # print(self.nodes['nt1'].num_state)
        self._reset_state()
        self.n_s_ls = []
        self.n_w_ls = []
        self.n_f_ls = []

        n_s_ls_signal = []
        n_w_ls_signal = []
        n_f_ls_signal = []
        n_o_ls_signal = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # wave wait
            num_wave = node.num_state
            num_wait = node.num_state
            num_fingerprint = 0
            num_arrival = 0
            for nnode_name in node.relevant_ss:
                num_wave += self.nodes[nnode_name].num_state
                num_fingerprint += self.nodes[nnode_name].num_fingerprint
            if self.problem == 'signal_route':
                for rrc_name in node.relevant_sr:
                    num_arrival += self.rcs[rrc_name].num_state_arrival
                    num_fingerprint += self.rcs[rrc_name].num_fingerprint

            n_s_ls_signal.append(num_wave)
            n_w_ls_signal.append(num_wait)
            n_o_ls_signal.append(num_arrival)
            n_f_ls_signal.append(num_fingerprint)

        n_s_ls_route = []
        n_w_ls_route = []
        n_o_ls_route = []
        n_f_ls_route = []

        for rc_name in self.rc_names:
            rc = self.rcs[rc_name]
            num_arrival = rc.num_state_arrival
            num_occupancy = rc.num_state_occupancy
            num_fingerprint = 0
            num_wave = 0
            for rrc_name in rc.relevant_rr:
                num_arrival += self.rcs[rrc_name].num_state_arrival
                num_fingerprint += self.rcs[rrc_name].num_fingerprint
            if self.problem == 'signal_route':
                for nnode_name in rc.relevant_rs:
                    num_wave += self.nodes[nnode_name].num_state
                    num_fingerprint += self.nodes[nnode_name].num_fingerprint
            n_s_ls_route.append(num_arrival)
            n_w_ls_route.append(num_occupancy)
            n_o_ls_route.append(num_wave)
            n_f_ls_route.append(num_fingerprint)

        if self.problem == 'signal':
            self.n_s_ls = n_s_ls_signal
            self.n_w_ls = n_w_ls_signal
            self.n_f_ls = n_f_ls_signal
            self.n_s = np.sum(np.array(self.n_s_ls + self.n_w_ls + self.n_f_ls))
        elif self.problem == 'route':
            self.n_s_ls = n_s_ls_route
            self.n_w_ls = n_w_ls_route
            self.n_f_ls = n_f_ls_route
            self.n_s = np.sum(np.array(self.n_s_ls + self.n_w_ls + self.n_f_ls))
        elif self.problem == 'signal_route':
            # 拼接
            self.n_s_ls_signal = n_s_ls_signal
            self.n_s_ls_route = n_s_ls_route
            self.n_w_ls_signal = n_w_ls_signal
            self.n_w_ls_route = n_w_ls_route
            self.n_o_ls_signal = n_o_ls_signal
            self.n_o_ls_route = n_o_ls_route
            self.n_f_ls_signal = n_f_ls_signal
            self.n_f_ls_route = n_f_ls_route

            self.n_s_ls = n_s_ls_signal + n_s_ls_route
            self.n_w_ls = n_w_ls_signal + n_w_ls_route
            self.n_o_ls = n_o_ls_signal + n_o_ls_route
            self.n_f_ls = n_f_ls_signal + n_f_ls_route
            self.n_s = np.sum(np.array(self.n_s_ls + self.n_w_ls + self.n_o_ls + self.n_f_ls))

    def _measure_reward_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            waits = []
            vehwaits = []
            for ild in self.nodes[node_name].ilds_in:
                '''
                if self.name == 'Sioux':
                    cur_wait = min(10, self.sim.lane.getLastStepHaltingNumber(ild))
                else:
                    cur_wait = self.sim.lanearea.getLastStepHaltingNumber(ild)
                '''
                waits.append(min(40, self.sim.lane.getLastStepHaltingNumber(ild)) * self.control_interval_sec)
                max_pos = 0
                veh_wait = 0
                if self.sim.lane.getLastStepHaltingNumber(ild) == 0:
                    veh_wait = 0
                else:
                    vehs = self.sim.lane.getLastStepVehicleIDs(ild)
                    for vid in vehs:
                        pos = self.sim.vehicle.getLanePosition(vid)
                        if pos > max_pos:
                            max_pos = pos
                            veh_wait = self.sim.vehicle.getWaitingTime(vid)
                vehwaits.append(veh_wait)

            wait = np.sum(np.array(waits)) if len(waits) else 0
            vehwait = np.sum(np.array(vehwaits)) if len(vehwaits) else 0
            node.wait = wait
            node.vehwait = vehwait
            node.reward = -wait - self.coef_vehwait * vehwait

        for rc_name in self.rc_names:
            rc = self.rcs[rc_name]

            vehicles = []
            for det in rc.det:
                vehicles += self.sim.inductionloop.getLastStepVehicleIDs(det)
            arrived = 0
            for vehicle in vehicles:
                id = vehicle.split('_')
                if id[1] == rc.f_id:
                    arrived += 1

            rc.arrived = arrived
            rc.reward = arrived

    def get_parameter_state(self):
        wave = []
        wait = []
        occupancy = []
        arrival = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            wave.append(node.wave_state)
            wait.append(node.wait_state)
        for rc_name in self.rc_names:
            rc = self.rcs[rc_name]
            arrival.append(rc.arrival_state)
            occupancy.append(rc.occupancy_state)

        wave_temp = []
        for i in range(len(wave)):
            wave_temp.append(np.mean(wave[i]))
        wave_ave = np.mean(wave_temp)

        wait_temp = []
        for i in range(len(wait)):
            wait_temp.append(np.mean(wait[i]))
        wait_ave = np.mean(wait_temp)

        arrival_temp = []
        for i in range(len(arrival)):
            arrival_temp.append(np.mean(arrival[i]))
        arrival_ave = np.mean(arrival_temp)

        occupancy_temp = []
        for i in range(len(occupancy)):
            occupancy_temp.append(np.mean(occupancy[i]))
        occupancy_ave = np.mean(occupancy_temp)

        state_norm_ave = {'wave': wave_ave, 'wait': wait_ave, 'arrival': arrival_ave, 'occupancy': occupancy_ave}
        return state_norm_ave

    def _clip_state(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            node.wave_state = self._norm_clip_state(np.array(node.wave_state),
                                                    self.norms['wave'],
                                                    self.clips['wave'])
            node.wait_state = self._norm_clip_state(np.array(node.wait_state),
                                                    self.norms['wait'],
                                                    self.clips['wait'])
        for rc_name in self.rc_names:
            rc = self.rcs[rc_name]
            rc.arrival_state = self._norm_clip_state(np.array(rc.arrival_state),
                                                     self.norms['arrival'],
                                                     self.clips['arrival'])
            rc.occupancy_state = self._norm_clip_state(np.array(rc.occupancy_state),
                                                       self.norms['occupancy'],
                                                       self.clips['occupancy'])

    def _measure_state_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            for state_name in self.state_names_signal:
                if state_name == 'wave':
                    cur_state = []
                    for ild in node.ilds_in:
                        if self.name == 'Sioux':
                            cur_wave = self.sim.lane.getLastStepVehicleNumber(ild)
                        else:
                            cur_wave = self.sim.lanearea.getLastStepVehicleNumber(ild)
                        cur_state.append(cur_wave)
                    cur_state = np.array(cur_state)
                else:
                    cur_state = []
                    # halting number*1s=wait time
                    for ild in node.ilds_in:
                        if self.name == 'Sioux':
                            cur_wait = self.sim.lane.getLastStepHaltingNumber(ild) * self.control_interval_sec
                        else:
                            cur_wait = self.sim.lanearea.getLastStepHaltingNumber(ild) * self.control_interval_sec
                        cur_state.append(cur_wait)
                    cur_state = np.array(cur_state)
                if state_name == 'wave':
                    node.wave_state = cur_state
                else:
                    node.wait_state = cur_state

        for rc_name in self.rc_names:
            rc = self.rcs[rc_name]
            for state_name in self.state_names_route:
                if state_name == 'arrival':
                    cur_state = []
                    # cur_arrival
                    cur_arrival = 0
                    Vehicle_Set = self.sim.edge.getLastStepVehicleIDs(rc.up_edge)
                    for vehicle in Vehicle_Set:
                        id = vehicle.split('_')
                        if id[1] == rc.f_id:
                            cur_arrival = cur_arrival + 1
                    cur_state.append(cur_arrival)
                    cur_state = np.array(cur_state)
                elif state_name == 'occupancy':
                    # occupancy
                    cur_state = []
                    # cur_occupancy
                    for edge in rc.down_edge:
                        cur_occupancy = self.sim.edge.getLastStepVehicleNumber(edge)
                        cur_state.append(cur_occupancy)
                    cur_state = np.array(cur_state)

                if state_name == 'arrival':
                    rc.arrival_state = cur_state
                elif state_name == 'occupancy':
                    rc.occupancy_state = cur_state

    def _measure_traffic_step(self):
        cars = self.sim.vehicle.getIDList()
        num_tot_car = len(cars)
        num_in_car = self.sim.simulation.getDepartedNumber()
        num_out_car = self.sim.simulation.getArrivedNumber()

        if num_tot_car > 0:
            avg_waiting_time = np.mean([self.sim.vehicle.getWaitingTime(car) for car in cars])
            avg_speed = np.mean([self.sim.vehicle.getSpeed(car) for car in cars])
        else:
            avg_speed = 0
            avg_waiting_time = 0
        # all trip-related measurements are not supported by traci,
        # need to read from outputfile afterwards
        queues = []
        for node_name in self.node_names:
            for ild in self.nodes[node_name].ilds_in:
                queues.append(self.sim.lane.getLastStepHaltingNumber(ild))
        avg_queue = np.mean(np.array(queues))
        std_queue = np.std(np.array(queues))
        cur_traffic = {'episode': self.cur_episode,
                       'time_sec': self.cur_sec,
                       'number_total_car': num_tot_car,
                       'number_departed_car': num_in_car,
                       'number_arrived_car': num_out_car,
                       'avg_wait_sec': avg_waiting_time,
                       'avg_speed_mps': avg_speed,
                       'std_queue': std_queue,
                       'avg_queue': avg_queue}
        self.traffic_data.append(cur_traffic)

    @staticmethod
    def _norm_clip_state(x, norm, clip=-1):
        x = x / norm
        return x if clip < 0 else np.clip(x, 0, clip)

    @staticmethod
    def _norm_clip_reward(x, norm, clip=-1):
        x = x / norm
        return x if clip < 0 else np.clip(x, -clip, clip)

    def _reset_state(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # prev action for yellow phase before each switch
            node.prev_action = 0
            node.num_fingerprint = node.n_a - 1
            node.num_state = self._get_node_state_num(node)

        for rc_name in self.rc_names:
            rc = self.rcs[rc_name]
            rc.prev_action = 0
            rc.num_fingerprint = rc.n_a - 1
            rc.num_state_arrival = self._get_rc_state_num()
            rc.num_state_occupancy = len(rc.down_edge)


    def _set_phase(self, action):
        for i in range(len(self.node_names)):
            node_name = self.node_names[i]
            a = list(action)[i]
            phase = self._get_node_phase(a, node_name)
            self.sim.trafficlight.setRedYellowGreenState(node_name, phase)
            self.sim.trafficlight.setPhaseDuration(node_name, self.control_interval_sec)

    def _set_route(self, action):
        for i in range(len(self.rc_names)):
            rc_name = self.rc_names[i]
            rc = self.rcs[rc_name]
            a = list(action)[i]
            Vehicle_Set = self.sim.edge.getLastStepVehicleIDs(rc.up_edge)
            vehicles = []
            for vehicle in Vehicle_Set:
                id = vehicle.split('_')
                if id[1] == rc.f_id:
                    vehicles.append(vehicle)

            route = rc.forward_route

            pi = [self.CAV_rate, 1 - self.CAV_rate]
            for i in range(len(vehicles)):
                p = np.random.choice(np.arange(len(pi)), p=pi)
                if p == 0:
                    # cav
                    if a == 0:
                        self.sim.vehicle.setRoute(vehicles[i], route[0])
                    else:
                        self.sim.vehicle.setRoute(vehicles[i], route[1])


    def _simulate(self):
        self.cur_sec += self.control_interval_sec
        self.sim.simulation.step(time=float(self.cur_sec))

        arrived_number = self.sim.simulation.getArrivedNumber()
        departed_number = self.sim.simulation.getDepartedNumber()
        waiting_time = 0
        travel_time = 0
        for id in self.sim.edge.getIDList():
            waiting_time += self.sim.edge.getLastStepHaltingNumber(id) * self.control_interval_sec
            travel_time += self.sim.edge.getLastStepVehicleNumber(id) * self.control_interval_sec

        self._measure_state_step()
        self._measure_reward_step()
        if self.is_record:
            self._measure_traffic_step()

        return arrived_number, departed_number, waiting_time, travel_time

    def collect_tripinfo(self):
        # read trip xml, has to be called externally to get complete file
        trip_file = self.output_path + ('%s_%s_trip.xml' % (self.name, self.problem))
        print(trip_file)
        tree = ET.ElementTree(file=trip_file)
        for child in tree.getroot():
            cur_trip = child.attrib
            cur_dict = {}
            cur_dict['episode'] = self.cur_episode
            cur_dict['id'] = cur_trip['id']
            cur_dict['depart_sec'] = cur_trip['depart']
            cur_dict['arrival_sec'] = cur_trip['arrival']
            cur_dict['duration_sec'] = cur_trip['duration']
            cur_dict['wait_step'] = cur_trip['waitingCount']
            cur_dict['wait_sec'] = cur_trip['waitingTime']
            self.trip_data.append(cur_dict)
        # delete the current xml
        os.remove(trip_file)

    def init_data(self, is_record, output_path, diff=None):
        self.is_record = is_record
        if diff == None:
            self.output_path = output_path
        else:
            self.output_path = "Sioux\\%s\\eva_data\\%.1f\\" % (self.problem, diff)
        if self.is_record:
            self.traffic_data = []
            self.control_data = []
            self.trip_data = []

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def output_data(self):
        if not self.is_record:
            logging.error('Env: no record to output!')
        control_data = pd.DataFrame(self.control_data)
        control_data.to_csv(self.output_path + ('%s_%s_control.csv' % (self.name, self.problem)))
        traffic_data = pd.DataFrame(self.traffic_data)
        traffic_data.to_csv(self.output_path + ('%s_%s_traffic.csv' % (self.name, self.problem)))
        trip_data = pd.DataFrame(self.trip_data)
        trip_data.to_csv(self.output_path + ('%s_%s_trip.csv' % (self.name, self.problem)))

    def reset(self, gui=False, test_ind=0, load=False, separate=False):
        # have to terminate previous sim before calling reset
        self._reset_state()
        if self.train_mode:
            seed = self.seed
        else:
            seed = self.test_seeds[test_ind]
        if load == False:
            self._init_sim(seed, gui=gui)
        else:
            self._init_sim(seed, gui=gui, load=load)
        self.cur_sec = 0
        self.cur_episode += 1
        if self.agent == 'ma2c':
            self.update_fingerprint(self._init_policy())
        self._init_sim_traffic()
        self.seed += 1
        if separate == True:
            state=self._get_state(reset=True, problem='signal') + self._get_state(reset=True, problem='route')
            self.problem='signal_route'
            return state
        else:
            return self._get_state(reset=True)

    def terminate(self):
        self.sim.close()

    def get_parameter(self, scenario):
        state_data = []
        reward_data = []
        total_reward_data = []
        self._init_sim(self.seed)
        self.cur_sec = 0
        for i in range(int(self.episode_length_sec / self.control_interval_sec)):
            self._simulate()
            get_parameter = True
            state_ave = self.get_parameter_state()
            state_data.append(state_ave)

            reward_ave, total_rewards_ave = self._get_reward(get_parameter=get_parameter)
            reward_data.append(reward_ave)
            total_reward_data.append(total_rewards_ave)

        path = scenario + '\\'
        state = pd.DataFrame(state_data)
        state.to_csv(path + ('state.csv'))
        reward = pd.DataFrame(reward_data)
        reward.to_csv(path + ('reward.csv'))
        total_reward = pd.DataFrame(total_reward_data)
        total_reward.to_csv(path + ('total_reward.csv'))

    def step_separate(self, action_signal, action_route):
        self._set_route(action_route)
        self._set_phase(action_signal)

        _, _, _, _= self._simulate()

        signal_state = self._get_state(problem='signal')

        route_state = self._get_state(problem='route')
        self.problem = 'signal_route'
        state = signal_state + route_state

        done = False
        if self.cur_sec >= self.episode_length_sec:
            done = True

        return state, done

    def step(self, action):

        if self.problem == 'signal_route':
            self._set_route(action[len(self.node_names):])
            self._set_phase(action[:len(self.node_names)])
        elif self.problem == 'signal':
            self._set_phase(action)
        elif self.problem == 'route':
            self._set_route(action)

        arrived_number, departed_number, waiting_time, travel_time = self._simulate()
        state = self._get_state()
        reward = self._get_reward()

        done = False
        if self.cur_sec >= self.episode_length_sec:
            done = True

        global_reward = np.sum(reward)

        if self.is_record:
            action_r = ','.join(['%d' % a for a in action])
            cur_control = {'episode': self.cur_episode,
                           'time_sec': self.cur_sec,
                           'step': self.cur_sec / self.control_interval_sec,
                           'action': action_r,
                           'reward': global_reward}
            self.control_data.append(cur_control)

        agent_reward = reward
        reward = self._local_reward_step(reward)
        reward = self._clip_reward(reward)

        # use local rewards in test
        if not self.train_mode:
            return state, reward, done, global_reward, arrived_number, departed_number, waiting_time, travel_time, agent_reward

        return state, reward, done, global_reward, arrived_number, departed_number, waiting_time, travel_time, agent_reward

    def _clip_reward(self, reward):
        if self.problem != 'route':
            rewards_signal = reward[:len(self.node_names)]
            rewards_signal = self._norm_clip_reward(np.array(rewards_signal), self.norms['reward_signal'],
                                                    self.clips['reward'])
            if self.problem == 'signal_route':
                rewards_route = reward[len(self.node_names):]
                rewards_route = self._norm_clip_reward(np.array(rewards_route), self.norms['reward_route'],
                                                       self.clips['reward'])
        elif self.problem == 'route':
            rewards_route = reward[:len(self.rc_names)]
            rewards_route = self._norm_clip_reward(np.array(rewards_route), self.norms['reward_route'],
                                                   self.clips['reward'])
        rewards = []
        if self.problem == 'signal_route':
            rewards = list(rewards_signal) + list(rewards_route)
        elif self.problem == 'signal':
            rewards = list(rewards_signal)
        elif self.problem == 'route':
            rewards = list(rewards_route)
        return np.array(rewards)

    def _local_reward_step(self, reward):
        new_reward = []
        if self.problem != 'route':
            # signal_route  signal
            for i in range(len(self.node_names)):
                node_name = self.node_names[i]
                cur_reward = reward[i]
                for nnode_name in self.nodes[node_name].relevant_ss:
                    j = self.node_names.index(nnode_name)
                    cur_reward += self.coop_gamma_ss * reward[j]
                if self.problem == 'signal_route':
                    for rrc_name in self.nodes[node_name].relevant_sr:
                        j = self.rc_names.index(rrc_name)
                        cur_reward += self.coop_gamma_sr * self.nodes[node_name].relevant_sr[rrc_name] * reward[
                            len(self.node_names) + j]
                new_reward.append(cur_reward)
        if self.problem != 'signal':
            # signal_route  route
            for i in range(len(self.rc_names)):
                rc_name = self.rc_names[i]
                if self.problem == 'signal_route':
                    cur_reward = reward[len(self.node_names) + i]
                    for rrc_name in self.rcs[rc_name].relevant_rr:
                        j = self.rc_names.index(rrc_name)
                        cur_reward += self.coop_gamma_rr * reward[len(self.node_names) + j]
                    for nnode_name in self.rcs[rc_name].relevant_rs:
                        j = self.node_names.index(nnode_name)
                        cur_reward += self.coop_gamma_rs * self.rcs[rc_name].relevant_rs[nnode_name] * reward[j]
                else:
                    cur_reward = reward[i]
                    for rrc_name in self.rcs[rc_name].relevant_rr:
                        j = self.rc_names.index(rrc_name)
                        cur_reward += self.coop_gamma_rr * reward[j]
                new_reward.append(cur_reward)
        reward = np.array(new_reward)
        return reward

    def update_fingerprint(self, policy):
        if self.problem != 'route':
            # signal_route  signal
            for i in range(len(self.node_names)):
                node_name = self.node_names[i]
                pi = policy[i]
                self.nodes[node_name].fingerprint = np.array(pi)[:-1]
            if self.problem == 'signal_route':
                for i in range(len(self.rc_names)):
                    rc_name = self.rc_names[i]
                    pi = policy[len(self.node_names) + i]
                    self.rcs[rc_name].fingerprint = np.array(pi)[:-1]
        else:
            # route
            for i in range(len(self.rc_names)):
                rc_name = self.rc_names[i]
                pi = policy[i]
                self.rcs[rc_name].fingerprint = np.array(pi)[:-1]
