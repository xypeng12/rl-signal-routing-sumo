"""
Particular class of Sioux
"""

import configparser
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time

from envs.env import PhaseMap, PhaseSet,RouteMap,RouteSet, TrafficSimulator
from Sioux.data.build_file import gen_rou_file,get_routes,relevant_ss,get_nodes,get_phases,get_flows

sns.set_color_codes()
STATE_NAMES_SIGNAL = ['wave', 'wait']
STATE_NAMES_ROUTE = ['arrival','occupancy']
STATE_ARRIVAL=1
STATE_OCCUPANCY=2
ROUTE_NUM=2
NODE,NOTLNODE=get_nodes()
routes0,rc_num=get_routes()
RC_NAMES = []
rs_rate=0.5
thres=0.05
for i in range(1, rc_num + 1):
    RC_NAMES.append('rc%d' % i)
PHASES,PHASE_NODE_MAP=get_phases()


class SiouxPhase(PhaseMap):
    def __init__(self):
        self.phases = {}
        for key, val in PHASES.items():
            self.phases[key] = PhaseSet(val)

class SiouxRoute(RouteMap):
    def __init__(self):
        routes = []
        index = 1
        self.relevant_rr_map = {}
        self.relevant_rs_map = {}
        self.up_edge_map={}
        self.down_edge_map={}
        self.forward_route_map={}
        self.f_id_map = {}
        self.det_map={}



        for i in range(len(routes0)):
            if len(routes0[i]) == 1:
                continue
            if len(routes0[i]) == 2:
                routes.append(routes0[i])
                self.relevant_rr_map['rc%d' % (index)] = []
                for k in range(len(routes0[i][0])):
                    if routes0[i][0][k] != routes0[i][1][k]:
                        self.relevant_rs_map['rc%d' % (index)] = self.merge_two_dic(self.get_rs_dic(routes0[i][0][k-1:]), self.get_rs_dic(routes0[i][1][k-1:]))
                        if k-3>=0:
                            self.up_edge_map['rc%d' % (index)] = '%s_%s' % (routes0[i][0][k - 3], routes0[i][0][k - 2])
                            self.forward_route_map['rc%d' % (index)] = [routes0[i][0][(k - 3):],
                                                                        routes0[i][1][(k - 3):]]
                        else:
                            self.up_edge_map['rc%d' % (index)]='%s_%s'%(routes0[i][0][k-2],routes0[i][0][k-1])
                            self.forward_route_map['rc%d' % (index)] = [routes0[i][0][(k - 2):],
                                                                        routes0[i][1][(k - 2):]]

                        self.down_edge_map['rc%d' % (index)]=self.merge_two_dic(self.get_edge_dic(routes0[i][0][k-1:]), self.get_edge_dic(routes0[i][1][k-1:]))
                        self.det_map['rc%d' % (index)] = [routes0[i][0][-2]+'_'+routes0[i][0][-1]+'_0',routes0[i][0][-2]+'_'+routes0[i][0][-1]+'_1']
                        self.f_id_map['rc%d' % (index)] = str(i+1)
                        break
                index = index + 1
            elif len(routes0[i]) == 3:
                # route0 route1route2| route1 route2
                route0, route1, route2 = self.sort_route(routes0[i][0], routes0[i][1], routes0[i][2])

                for k in range(len(route0)):
                    if route0[k] == route1[k]:
                        continue
                    routes.append([route0,route1])

                    self.relevant_rs_map['rc%d' % (index)] =self.merge_two_dic(self.merge_two_dic(self.get_rs_dic(route0[k-1:]), self.get_rs_dic(route1[k-1:])),self.get_rs_dic(route2[k-1:]))
                    self.down_edge_map['rc%d' % (index)] = self.merge_two_dic(self.get_edge_dic(route0[k-1:]), self.get_edge_dic(route1[k-1:]))
                    if k-3>=0:
                        self.up_edge_map['rc%d' % (index)] = '%s_%s' % (
                            route0[k - 3], route0[k - 2])
                        self.forward_route_map['rc%d' % (index)] = [route0[(k - 3):], route1[(k - 3):]]
                    else:
                        self.up_edge_map['rc%d' % (index)] = '%s_%s' % (
                            route0[k - 2], route0[k - 1])
                        self.forward_route_map['rc%d' % (index)] = [route0[(k - 2):], route1[(k - 2):]]

                    self.det_map['rc%d' % (index)] = [route0[-2]+'_'+route0[-1]+'_0',route0[-2]+'_'+route0[-1]+'_1']
                    self.f_id_map['rc%d' % (index)] = str(i + 1)

                    routes.append([route1, route2])
                    for k2 in range(k, len(route1)):
                        if route1[k2] != route2[k2]:
                            self.relevant_rs_map['rc%d' % (index + 1)] =self.merge_two_dic(self.get_rs_dic(route1[k2-1:]), self.get_rs_dic(route2[k2-1:]))
                            self.down_edge_map['rc%d' % (index + 1)] =self.merge_two_dic(self.get_edge_dic(route1[k2-1:]), self.get_edge_dic(route2[k2-1:]))
                            if k2-3>=0:
                                self.up_edge_map['rc%d' % (index + 1)] = '%s_%s' % (
                                    route1[k2 - 3], route1[k2 - 2])
                                self.forward_route_map['rc%d' % (index + 1)] = [route1[(k2 - 3):],
                                                                            route2[(k2 - 3):]]
                            else:
                                self.up_edge_map['rc%d' % (index + 1)] = '%s_%s' % (
                                    route1[k2 - 2], route1[k2 - 1])
                                self.forward_route_map['rc%d' % (index + 1)] = [route1[(k2 - 2):],
                                                                                route2[(k2 - 2):]]

                            self.det_map['rc%d' % (index+1)] = [route1[-2] + '_' + route1[-1] + '_0',
                                                              route1[-2] + '_' + route1[-1] + '_1']
                            self.f_id_map['rc%d' % (index + 1)] = str(i + 1)
                            break
                    self.relevant_rr_map['rc%d' % (index)] = ['rc%d' % (index + 1)]
                    self.relevant_rr_map['rc%d' % (index + 1)] = ['rc%d' % (index)]
                    index = index + 2
                    break

            else:
                print("more than 3 routes")

        self.relevant_sr_map = {}
        for r in self.relevant_rs_map:
            for n in self.relevant_rs_map[r]:
                if n not in NODE:
                   continue
                if n not in self.relevant_sr_map:
                   self.relevant_sr_map[n]={}
                self.relevant_sr_map[n][r]=self.relevant_rs_map[r][n]
        for n in NODE:
            if n not in self.relevant_sr_map:
                self.relevant_sr_map[n]={}

        for r in self.relevant_rs_map:
            for n in NOTLNODE:
                if n in self.relevant_rs_map[r]:
                    self.relevant_rs_map[r].pop(n)


        temp_map=self.forward_route_map
        self.forward_route_map={}
        for rc in temp_map:
            self.forward_route_map[rc]=[]
            for i in range(len(temp_map[rc])):
                self.forward_route_map[rc].append([])
                for j in range(len(temp_map[rc][i])-1):
                    self.forward_route_map[rc][i].append('%s_%s'%(temp_map[rc][i][j],temp_map[rc][i][j+1]))

        self.routes = {}
        for i in range(len(routes)):
            self.routes['rc%d' % (i + 1)] = RouteSet(routes[i])

    def merge_two_dic(self,s1, s2):
        dic = {}
        for k, v in s2.items():
            if k not in s1:
                dic[k] = v
            else:
                dic[k] = max(v, s1[k])
        for k, v in s1.items():
            if k not in s2:
                dic[k] = v
        return dic

    def get_edge_dic(self,route):
        dic = {}
        for i in range(len(route) - 1):
            if rs_rate ** i < thres:
                break
            edge = '%s_%s' % (route[i], route[i + 1])
            dic[edge] = rs_rate ** i
        return dic

    def get_rs_dic(self,route):
        dic = {}
        for i in range(len(route)):
            if rs_rate ** i < thres:
                break
            s = route[i]
            dic[s] = rs_rate ** i
        return dic

    def sort_route(self,r1, r2, r3):
        route1 = []
        route2 = []
        route3 = []
        for i in range(len(r1)):
            if r1[i] == r2[i] and r1[i] == r3[i]:
                continue
            if r1[i] == r2[i] and r1[i] != r3[i]:
                route1 = r3
                route2 = r1
                route3 = r2
                break
            elif r1[i] != r2[i] and r1[i] == r3[i]:
                route1 = r2
                route2 = r1
                route3 = r3
                break
            elif r2[i] == r3[i] and r1[i] != r2[i]:
                route1 = r1
                route2 = r2
                route3 = r3
                break
        return route1, route2, route3





class SiouxController:
    def __init__(self, node_names,rc_names):
        self.name = 'greedy'
        self.node_names = node_names
        self.rc_names = rc_names

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        # hard code the mapping from state to number of cars
        flows = [ob[0] + ob[3], ob[2] + ob[5], ob[1] + ob[4],
                 ob[1] + ob[2], ob[4] + ob[5]]
        return np.argmax(np.array(flows))


class SiouxEnv(TrafficSimulator):
    def __init__(self, config,problem, port=0, output_path='', is_record=False):
        self.demand_flows = get_flows()
        super().__init__(config, problem,output_path, is_record, port=port)

    def _get_node_phase_id(self, node_name):
        return self.phase_node_map[node_name]

    def _get_rc_route_num(self,rc_name):
        return ROUTE_NUM
    def _get_rc_names(self):

        return RC_NAMES

    def _get_rc_state_num(self):
        return STATE_ARRIVAL

    def _init_relevant_ss_map(self):

        return relevant_ss()


    def _init_map(self):

        self.phase_node_map=PHASE_NODE_MAP
        self.phase_map = SiouxPhase()
        self.route_map = SiouxRoute()
        self.relevant_ss_map = self._init_relevant_ss_map()
        self.relevant_rs_map=self.route_map.relevant_rs_map
        self.relevant_sr_map=self.route_map.relevant_sr_map
        self.relevant_rr_map=self.route_map.relevant_rr_map
        self.up_edge_map = self.route_map.up_edge_map
        self.down_edge_map = self.route_map.down_edge_map
        self.forward_route_map = self.route_map.forward_route_map
        self.f_id_map = self.route_map.f_id_map
        self.det_map=self.route_map.det_map
        self.state_names_signal = STATE_NAMES_SIGNAL
        self.state_names_route = STATE_NAMES_ROUTE


    def _init_sim_config(self, seed,load):
        return gen_rou_file(self.data_path,
                            self.demand_flows,
                            seed=seed,
                            thread=self.sim_thread,
                            load=load)

    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards
        for name, data in self.state_stat.items():
            fig = plt.figure(figsize=(8, 6))
            plot_cdf(data)
            plt.ylabel(name)
            fig.savefig(self.output_path + self.name + '_' + name + '.png')


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, color=c, label=label)
