# -*- coding: utf-8 -*-

import numpy as np
import os
import csv
import math

MAX_CAR_NUM = 30
SPEED_LIMIT = 20
build=False
time_length=3600
def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def output_nodes(node,build=False):
    if build==True:
        csv_reader = csv.reader(open("SiouxFalls_node.csv"))
    else:
        csv_reader = csv.reader(open("Sioux/data/SiouxFalls_node.csv"))
    str_nodes = '<nodes>\n'
    for row in csv_reader:
        if row[0] == 'Node':
            continue
        if row[0].startswith('n'):
            x = int(float(row[1]) / 1000)
            y = int(float(row[2]) / 1000)
        elif row[0] in ['e1','e3','e13','e23']:
            x = int(float(row[1]) / 1000) - 1000
            y = int(float(row[2]) / 1000)
        elif row[0] in ['e5','e7']:
            x = int(float(row[1]) / 1000)
            y = int(float(row[2]) / 1000)+1000
        elif row[0] in ['e2']:
            x = int(float(row[1]) / 1000)+1000
            y = int(float(row[2]) / 1000)

        str_nodes += node % (str(row[0]), x, y, str(row[3]))
    str_nodes += '</nodes>\n'
    return str_nodes

def output_road_types():
    str_types = '<types>\n'
    str_types += '  <type id="a" priority="1" numLanes="2" speed="%.2f"/>\n' % SPEED_LIMIT
    str_types += '</types>\n'
    return str_types

def get_edge_str(edge, from_node, to_node, edge_type):
    edge_id = '%s_%s' % (from_node, to_node)
    return edge % (edge_id, from_node, to_node, edge_type)

def output_edges(edge,build=False):
    str_edges = '<edges>\n'

    if build==True:
        csv_reader = csv.reader(open("SiouxFalls_net.csv"))
    else:
        csv_reader = csv.reader(open("Sioux/data/SiouxFalls_net.csv"))

    for row in csv_reader:
        if row[0] == 'LINK':
            continue
        if row[0]=='e':
            # external roads
            str_edges += get_edge_str(edge, row[1],row[2],'a')
        else:
            # internal roads
            str_edges += get_edge_str(edge, 'n'+row[1], 'n'+row[2], 'a')

    str_edges += '</edges>\n'
    return str_edges

def get_con_str(con, from_node, cur_node, to_node, from_lane, to_lane):
    from_edge = '%s_%s' % (from_node, cur_node)
    to_edge = '%s_%s' % (cur_node, to_node)
    return con % (from_edge, to_edge, from_lane, to_lane)


def get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node,s_node_2=None):
    str_cons = ''
    # go-through
    str_cons += get_con_str(con, s_node, cur_node, n_node, 0, 0)
    str_cons += get_con_str(con, n_node, cur_node, s_node, 0, 0)
    str_cons += get_con_str(con, w_node, cur_node, e_node, 0, 0)
    str_cons += get_con_str(con, e_node, cur_node, w_node, 0, 0)
    # left-turn
    str_cons += get_con_str(con, s_node, cur_node, w_node, 1, 1)
    str_cons += get_con_str(con, n_node, cur_node, e_node, 1, 1)
    str_cons += get_con_str(con, w_node, cur_node, n_node, 1, 1)
    str_cons += get_con_str(con, e_node, cur_node, s_node, 1, 1)
    # right-turn
    str_cons += get_con_str(con, s_node, cur_node, e_node, 0, 0)
    str_cons += get_con_str(con, n_node, cur_node, w_node, 0, 0)
    str_cons += get_con_str(con, w_node, cur_node, s_node, 0, 0)
    str_cons += get_con_str(con, e_node, cur_node, n_node, 0, 0)
    if s_node_2 is not None:
        str_cons += get_con_str(con, s_node_2, cur_node, n_node, 0, 0)
        str_cons += get_con_str(con, s_node_2, cur_node, w_node, 1, 1)
        str_cons += get_con_str(con, s_node_2, cur_node, e_node, 0, 0)
        str_cons += get_con_str(con, s_node_2, cur_node, s_node, 1, 1)
        str_cons += get_con_str(con, n_node, cur_node, s_node_2, 1, 1)
        str_cons += get_con_str(con, e_node, cur_node, s_node_2, 1, 1)
        str_cons += get_con_str(con, w_node, cur_node, s_node_2, 0, 0)
        str_cons += get_con_str(con, s_node, cur_node, s_node_2, 0, 0)

    return str_cons

def output_ild(ild,build):
    str_ild='<additional>\n'
    if build==True:
        csv_reader=csv.reader(open("SiouxFalls_net.csv"))
    else:
        csv_reader = csv.reader(open("Sioux/data/SiouxFalls_net.csv"))
    for row in csv_reader:
        if row[0] == 'LINK':
            continue
        if row[0]=='e' and row[1][0]=='n':
            edge=row[1]+'_'+row[2]
            for i in range(2):
                lane=edge+'_'+str(i)
                str_ild += ild%(lane,lane)
    str_ild+='</additional>\n'
    return str_ild


def output_connections(con):
    str_cons = '<connections>\n'
    # n3 n5 n8 n11 n15 n16 n20 n22 n23
    cur_node = 'n3'
    n_node = 'n1'
    s_node = 'n12'
    w_node = 'e3'
    e_node = 'n4'
    str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    cur_node = 'n5'
    n_node = 'e5'
    s_node = 'n9'
    w_node = 'n4'
    e_node = 'n6'
    str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    cur_node = 'n8'
    n_node = 'n6'
    s_node = 'n16'
    w_node = 'n9'
    e_node = 'n7'
    str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    cur_node = 'n11'
    n_node = 'n4'
    s_node = 'n14'
    w_node = 'n12'
    e_node = 'n10'
    str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    cur_node = 'n15'
    n_node = 'n10'
    s_node = 'n22'
    w_node = 'n14'
    e_node = 'n19'
    str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    cur_node = 'n16'
    n_node = 'n8'
    s_node = 'n17'
    w_node = 'n10'
    e_node = 'n18'
    str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    cur_node = 'n20'
    n_node = 'n19'
    s_node = 'n21'
    w_node = 'n22'
    e_node = 'n18'
    str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    cur_node = 'n22'
    n_node = 'n15'
    s_node = 'n21'
    w_node = 'n23'
    e_node = 'n20'
    str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    cur_node = 'n23'
    n_node = 'n14'
    s_node = 'n24'
    w_node = 'e23'
    e_node = 'n22'
    str_cons += get_con_str_set(con, cur_node, n_node, s_node, w_node, e_node)

    str_cons +=get_con_str_set(con, 'n10', 'n9', 'n15', 'n11','n16', s_node_2='n17')

    str_cons+=get_con_str_set_n18(con,'n18','n7','n20','n16')

    str_cons += '</connections>\n'
    return str_cons

def get_con_str_set_n18(con, cur_node, n_node, s_node, w_node):
    str_cons = ''
    # go-through
    str_cons += get_con_str(con, s_node, cur_node, n_node, 0, 0)
    str_cons += get_con_str(con, n_node, cur_node, s_node, 0, 0)
    str_cons += get_con_str(con, s_node, cur_node, n_node, 1, 1)
    str_cons += get_con_str(con, n_node, cur_node, s_node, 1, 1)

    # left-turn
    str_cons += get_con_str(con, w_node, cur_node, n_node, 1, 1)
    str_cons += get_con_str(con, s_node, cur_node, w_node, 1, 1)
    str_cons += get_con_str(con, w_node, cur_node, w_node, 1, 1)
    str_cons += get_con_str(con, n_node, cur_node, n_node, 1, 1)
    str_cons += get_con_str(con, s_node, cur_node, s_node, 1, 1)

    # right-turn
    str_cons += get_con_str(con, n_node, cur_node, w_node, 0, 0)
    str_cons += get_con_str(con, w_node, cur_node, s_node, 0, 0)
    str_cons += get_con_str(con, w_node, cur_node, s_node, 1, 1)
    return str_cons

def output_netconfig():
    str_config = '<configuration>\n  <input>\n'
    str_config += '    <edge-files value="exp.edg.xml"/>\n'
    str_config += '    <node-files value="exp.nod.xml"/>\n'
    str_config += '    <type-files value="exp.typ.xml"/>\n'
    str_config += '    <tllogic-files value="exp.tll.xml"/>\n'
    str_config += '    <connection-files value="exp.con.xml"/>\n'
    str_config += '  </input>\n  <output>\n'
    str_config += '    <output-file value="exp.net.xml"/>\n'
    str_config += '  </output>\n</configuration>\n'
    return str_config

def get_flows(build=False):
    if build==True:
        csv_reader = csv.reader(open("SiouxFalls_od.csv"))
    else:
        csv_reader = csv.reader(open("Sioux/data/SiouxFalls_od.csv"))

    flows=[]
    for row in csv_reader:
        if row[0] == 'OD':
            continue
        flows.append(int(row[3]))
    return flows

def get_phases():
    phases = {'a': ['GGggrrrrGGGg', 'rrrrGGGgGrrr'],
              'b': ['rrrrGGGgGGgg', 'GGGgGrrrrrrr'],
              'c': ['GGGgGGggrrrr', 'GrrrrrrrGGGg'],
              'd': ['GGrrrrGGrrrr', 'GrGGrrGrGGrr', 'rrrGGrrrrGGr', 'GrrGrGGrrGrG'],
              'e': ['GGrrrrrrrrrrGGGrrrrr', 'rrGGrrrrrrrrrrrGrrrr', 'rrrrGGrrrrrrrrrrGGGr', 'rrrrrrGGrrrrrrrrrrrG',
                    'rrrrrrrrGGGGGrrrrrrr']
              }
    phase_node_map = {'n1': 'a', 'n2': 'a', 'n4': 'a', 'n9': 'a', 'n12': 'a', 'n14': 'a',
                      'n13': 'b', 'n21': 'b', 'n24': 'b',
                      'n6': 'c', 'n7': 'c', 'n17': 'c', 'n18': 'c', 'n19': 'c',
                      'n3': 'd', 'n5': 'd', 'n8': 'd', 'n11': 'd', 'n15': 'd', 'n16': 'd', 'n20': 'd', 'n22': 'd',
                      'n23': 'd',
                      'n10': 'e'
                      }
    return phases,phase_node_map


def get_routes():
    csv_reader = csv.reader(open("Sioux/data/SiouxFalls_od.csv"))
    route = []
    o = []
    d = []
    for row in csv_reader:
        if row[0] == 'OD':
            continue
        r = []
        for i in range(int(row[4])):
            r.append(row[int(5 + i)].split(' '))
        route.append(r)
        o.append(row[1])
        d.append(row[2])
    routes=[]
    for i in range(len(route)):
        r1 = []
        for k in range(len(route[i])):
            r2=[]
            r2.append('e' + o[i])
            for j in range(len(route[i][k])):
                r2.append('n' + route[i][k][j])
            r2.append('e' + d[i])
            r1.append(r2)
        routes.append(r1)
    rc_num=0
    for i in range(len(routes)):
        rc_num+=len(routes[i])-1
    return routes, rc_num
def relevant_ss(build=False):
    # 相邻交叉口
    if build==True:
        csv_reader = csv.reader(open("SiouxFalls_net.csv"))
    else:
        csv_reader = csv.reader(open("Sioux/data/SiouxFalls_net.csv"))

    node,_=get_nodes(build=build)

    relevant_ss_map = {}
    for row in csv_reader:
        if row[0] == 'LINK' or row[0] == 'e' or 'n' + row[1] not in node or 'n' + row[2] not in node:
            continue

        if 'n' + row[1] not in relevant_ss_map.keys():
            relevant_ss_map['n' + row[1]] = []
        relevant_ss_map['n' + row[1]].append('n' + row[2])
    return relevant_ss_map

def get_nodes(build=False):

    if build==True:
        csv_reader = csv.reader(open("SiouxFalls_node.csv"))
    else:
        csv_reader = csv.reader(open("Sioux/data/SiouxFalls_node.csv"))
    node = []
    notlnode=[]
    for row in csv_reader:
        if row[0] == 'Node':
            continue
        if row[3]=='traffic_light':
            node.append(row[0])
        elif row[3]=='priority':
            notlnode.append(row[0])
    return node,notlnode

def get_routes_xml(route,o,d):
    routes_output = []

    for i in range(len(route)):
        routes = []
        for k in range(len(route[i])):
            routes.append('e' + o[i] + '_' + 'n' + route[i][k][0] + ' ')
            for j in range(len(route[i][k]) - 1):
                routes[k] += 'n' + route[i][k][j] + '_' + 'n' + route[i][k][j + 1] + ' '
            routes[k] += ('n' + route[i][k][len(route[i][k]) - 1] + '_' + 'e' + d[i])
        routes_output.append(routes)
    return routes_output

def output_flows(demand_flows,seed=None,build=False,load=False):
    if build==True:
        csv_reader = csv.reader(open("SiouxFalls_od.csv"))
    else:
        csv_reader = csv.reader(open("Sioux/data/SiouxFalls_od.csv"))

    route = []
    o = []
    d = []

    for row in csv_reader:
        if row[0] == 'OD':
            continue
        r = []
        for i in range(int(row[4])):
            r.append(row[int(5 + i)].split(' '))
        route.append(r)
        o.append(row[1])
        d.append(row[2])

    routes=get_routes_xml(route, o, d)

    demand_begin = 0
    demand_end = 3600
    demand_gap=360
    #0-3600s 每360s一个间隔
    ratio1=[0.3,0.5,0.4,0.3,0.8,0.9,1,0.9,0.7,0.2]
    ratio2=[0.3,0.2,0.4,0.3,0.5,0.5,0.8,0.8,0.9,1]
    ratio3=[0.2,0.2,0.3,0.3,0.6,0.8,1,0.9,0.4,0.3]
    ratio4=[0.9,1,0.9,0.9,0.6,0.4,0.2,0.2,0.4,0.3]
    ratios=[ratio1, ratio2,ratio3,ratio2,ratio1,ratio3,ratio4,ratio1,ratio3,ratio2,ratio3,ratio4,ratio1, ratio2,ratio3,ratio4,ratio2,ratio1,ratio3,ratio4,ratio1,ratio2,ratio3,ratio2,ratio4,ratio1]

    flows = []
    for i in range(len(ratios)):
        # 第i个flow
        flows.append([])
        for j in range(10):
            # 第j个时段
            flows[i].append(0)
    for i in range(len(ratios)):
        # 第i个flow
        for j in range(10):
            # 第j个时段
            flows[i][j] = ratios[i][j] * demand_flows[i]

    if seed is not None:
        np.random.seed(seed)
    str_flows = '<routes>\n'
    str_flows += '<vType id="type1" length="5" accel="5" decel="10"/>\n'
    ext_flow = '<flow id="f_%s_%s" route="f_%s" departPos="random_free" begin="%d" end="%d" vehsPerHour="%d" type="type1"/>\n'
    ext_route='<route id="f_%s" edges="%s"/>\n'
    for i in range(len(ratios)):
        # 第i个flow
        str_flows += ext_route % (i+1, routes[i][0])

    for j in range(10):
    # 第j个时段
        for i in range(len(ratios)):
        # 第i个flow
            t_begin=j*demand_gap
            t_end=(j+1)*demand_gap
            if type(load)==bool and load==False:
                str_flows += ext_flow % (i+1,j+1,i+1,t_begin, t_end, flows[i][j])
            else:
                str_flows+=ext_flow % (i+1,j+1,i+1,t_begin, t_end, int(flows[i][j]*load))
    str_flows += '</routes>\n'
    return str_flows

def gen_rou_file(path,demand_flows, seed=None, thread=None,load=False):
    if thread is None:
        flow_file = 'exp.rou.xml'
    else:
        flow_file = 'exp_%d.rou.xml' % int(thread)
    write_file(path + flow_file, output_flows(demand_flows,seed=seed,load=load))
    sumocfg_file = path + ('exp_%d.sumocfg' % thread)
    write_file(sumocfg_file, output_config(thread=thread))
    return sumocfg_file


def output_config(thread=None):
    if thread is None:
        out_file = 'exp.rou.xml'
    else:
        out_file = 'exp_%d.rou.xml' % int(thread)
    str_config = '<configuration>\n  <input>\n'
    str_config += '    <net-file value="exp.net.xml"/>\n'
    str_config += '    <route-files value="%s"/>\n' % out_file
    str_config += '    <additional-files value="exp.add.xml"/>\n'
    str_config += '  </input>\n  <time>\n'
    str_config += '    <begin value="0"/>\n    <end value="%d"/>\n'%time_length
    str_config += '  </time>\n</configuration>\n'
    return str_config

'''
def get_ild_str(from_node, to_node, ild_str, lane_i=0):
    edge = '%s_%s' % (from_node, to_node)
    return ild_str % (edge, lane_i, edge, lane_i)


def output_ild(ild):
    str_adds = '<additional>\n'
    in_edges = [1,2,3,4,5,6,1,3,4,6]
    out_edges = [1,2,3,8,9,10,4,5,6,7]
    # external edges
    for k, (i, j) in enumerate(zip(in_edges, out_edges)):
        node1 = 'nt' + str(i)
        node2 = 'np' + str(j)
        str_adds += get_ild_str(node2, node1, ild)
        if k >=6:
            # streets
            str_adds += get_ild_str(node2, node1, ild, lane_i=1)
    # streets
    for i in range(1, 6, 3):
        for j in range(2):
            node1 = 'nt' + str(i + j)
            node2 = 'nt' + str(i + j + 1)
            str_adds += get_ild_str(node1, node2, ild)
            str_adds += get_ild_str(node2, node1, ild)
            str_adds += get_ild_str(node1, node2, ild, lane_i=1)
            str_adds += get_ild_str(node2, node1, ild, lane_i=1)
    # avenues
    for i in range(1, 4):
        for j in range(1):
            node1 = 'nt' + str(i + j)
            node2 = 'nt' + str(i + j + 3)
            str_adds += get_ild_str(node1, node2, ild)
            str_adds += get_ild_str(node2, node1, ild)
    str_adds += '</additional>\n'
    return str_adds
'''

def output_tls(tls, phase,NODE):
    str_adds = '<additional>\n'
    #t字形路口:主线 支线 2相位
    phase_duration = [30, 5]
    # n1 n2 n4 n9 n12 n14
    phases=['GGggrrrrGGGg','yyyyrrrryyyy','rrrrGGGgGrrr','rrrryyyyyrrr']
    n=['n1', 'n2', 'n4', 'n9', 'n12','n14']
    for i in set(n).intersection(set(NODE)) :
        node = i
        str_adds += tls % node
        for k, p in enumerate(phases):
            str_adds += phase % (phase_duration[k % 2], p)
        str_adds += '  </tlLogic>\n'
    #n13 n21 n24
    phases=['rrrrGGGgGGgg','rrrryyyyyyyy','GGGgGrrrrrrr','yyyyyrrrrrrr']
    n=['n13','n21','n24']
    for i in set(n).intersection(set(NODE)):
        node = i
        str_adds += tls % node
        for k, p in enumerate(phases):
            str_adds += phase % (phase_duration[k % 2], p)
        str_adds += '  </tlLogic>\n'

    #n6 n7 n17 n18 n19
    phases=['GGGgGGggrrrr','yyyyyyyyrrrr','GrrrrrrrGGGg','yrrrrrrryyyy']
    n=['n6','n7','n17','n18','n19']
    for i in set(n).intersection(set(NODE)):
        node = i
        str_adds += tls % node
        for k, p in enumerate(phases):
            str_adds += phase % (phase_duration[k % 2], p)
        str_adds += '  </tlLogic>\n'

    #十字路口 4相位
    #n3 n5 n8 n11 n15 n16 n20 n22 n23
    phases=['GGrrrrGGrrrr','GyrrrrGyrrrr',
            'GrGGrrGrGGrr','yryGrryryGrr',
            'rrrGGrrrrGGr','rrrGyrrrrGyr',
            'GrrGrGGrrGrG','GrryryGrryry']
    n=['n3','n5' ,'n8', 'n11', 'n15', 'n16', 'n20' ,'n22' ,'n23']
    for i in set(n).intersection(set(NODE)):
        node = i
        str_adds += tls % node
        for k, p in enumerate(phases):
            str_adds += phase % (phase_duration[k % 2], p)
        str_adds += '  </tlLogic>\n'
    #五叉交叉口
    #n10
    phases = ['GGrrrrrrrrrrGGGrrrrr', 'yyrrrrrrrrrryyyrrrrr',
              'rrGGrrrrrrrrrrrGrrrr', 'rryyrrrrrrrrrrryrrrr',
              'rrrrGGrrrrrrrrrrGGGr', 'rrrryyrrrrrrrrrryyyr',
              'rrrrrrGGrrrrrrrrrrrG', 'rrrrrryyrrrrrrrrrrry',
              'rrrrrrrrGGGGGrrrrrrr','rrrrrrrryyyyyrrrrrrr']
    n=['n10']
    for i in set(n).intersection(set(NODE)):
        node = i
        str_adds += tls % node
        for k, p in enumerate(phases):
            str_adds += phase % (phase_duration[k % 2], p)
        str_adds += '  </tlLogic>\n'
    str_adds += '</additional>\n'
    return str_adds

def main():
    # nod.xml file
    build=True
    relevant_ss(build=build)

    node = '  <node id="%s" x="%.2f" y="%.2f" type="%s"/>\n'
    write_file('./exp.nod.xml', output_nodes(node,build))

    # typ.xml file
    write_file('./exp.typ.xml', output_road_types())

    # edg.xml file
    edge = '  <edge id="%s" from="%s" to="%s" type="%s"/>\n'
    write_file('./exp.edg.xml', output_edges(edge,build))

    # con.xml file
 
    con = '  <connection from="%s" to="%s" fromLane="%d" toLane="%d"/>\n'
    write_file('./exp.con.xml', output_connections(con))

    # tls.xml file
    tls = '  <tlLogic id="%s" programID="0" offset="0" type="static">\n'
    phase = '    <phase duration="%d" state="%s"/>\n'
    node,_=get_nodes(build=build)
    write_file('./exp.tll.xml', output_tls(tls, phase,node))

    # net config file
    write_file('./exp2.0.netccfg', output_netconfig())

    # generate net.xml file
    os.system('netconvert -c exp2.0.netccfg')

    demand_flows=get_flows(build)

    # raw.rou.xml file
    write_file('./exp.rou.xml', output_flows(demand_flows,seed=None,build=build))

    # add.xml file
    ild = '  <inductionLoop file="ild_out.xml" freq="3600" id="%s" lane="%s" pos="30"/>\n'
    write_file('./exp.add.xml', output_ild(ild,build))


    # config file
    write_file('./exp.sumocfg', output_config())

if __name__ == '__main__':
    main()