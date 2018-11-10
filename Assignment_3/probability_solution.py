"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys

if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
from exampleinference import inferenceExample

inferenceExample()
# Should output:
# ('The marginal probability of sprinkler=false:', 0.80102921)
#('The marginal probability of wetgrass=false | cloudy=False, rain=True:', 0.055)

'''
WRITE YOUR CODE BELOW. DO NOT CHANGE ANY FUNCTION HEADERS FROM THE NOTEBOOK.
'''


from Node import BayesNode
from Graph import BayesNet
from numpy import zeros, float32
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Inference import JunctionTreeEngine, EnumerationEngine
import random as rand
import numpy as np


def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    
    G_node = BayesNode(0,2,name='gauge')
    FG_node = BayesNode(1,2,name='faulty gauge')
    T_node = BayesNode(2,2,name='temperature')
    A_node = BayesNode(3,2,name='alarm')
    FA_node = BayesNode(4,2,name='faulty alarm')
    
    T_node.add_child(G_node)
    T_node.add_child(FG_node)
    G_node.add_parent(T_node)
    FG_node.add_parent(T_node)
    FG_node.add_child(G_node)
    G_node.add_parent(FG_node)
    G_node.add_child(A_node)
    A_node.add_parent(G_node)
    FA_node.add_child(A_node)
    A_node.add_parent(FA_node)
    
    
    nodes = [G_node, FG_node, T_node, FA_node, A_node]
    # TODO: finish this function    
    return BayesNet(nodes)

def make_final_net():
    A = BayesNode(0,2,name='A')
    B = BayesNode(1,2,name='B')
    C = BayesNode(2,2,name='C')
    D = BayesNode(3,2,name='D')
    E = BayesNode(4,2,name='E')
    
    A.add_child(C)
    C.add_parent(A)
    B.add_child(C)
    C.add_parent(B)
    B.add_child(D)
    D.add_parent(B)
    C.add_child(E)
    E.add_parent(C)
    D.add_child(E)
    E.add_parent(D)
    
    nodes = [A,B,C,D,E]
    net = BayesNet(nodes)
    
    A = net.get_node_by_name('A')
    B = net.get_node_by_name('B')
    C = net.get_node_by_name('C')
    D = net.get_node_by_name('D')
    E = net.get_node_by_name('E')
    
    A_distribution = DiscreteDistribution(A)
    index = A_distribution.generate_index([],[])
    A_distribution[index] = [0.4, 0.6]
    A.set_dist(A_distribution)
    
    B_distribution = DiscreteDistribution(B)
    index = B_distribution.generate_index([],[])
    B_distribution[index] = [0.8, 0.2]
    B.set_dist(B_distribution)
    
    print B, D
    dist = zeros([B.size(), D.size()], dtype=float32)
    dist[0,:] = [0.87, 0.13]
    dist[1,:] = [0.24, 0.76]
    D_distribution = ConditionalDiscreteDistribution(nodes=[B,D], table=dist)
    D.set_dist(D_distribution)
    
    dist = zeros([A.size(), B.size(), C.size()], dtype=float32)
    dist[0,0,:] = [0.85, 0.15]
    dist[0,1,:] = [0.68, 0.32]
    dist[1,0,:] = [0.16, 0.84]
    dist[1,1,:] = [0.05, 0.95]
    C_distribution = ConditionalDiscreteDistribution(nodes=[A, B, C], table=dist)
    C.set_dist(C_distribution)
    
    dist = zeros([C.size(), D.size(), E.size()], dtype=float32)
    dist[0,0,:] = [0.8, 0.2]
    dist[0,1,:] = [0.37, 0.63]
    dist[1,0,:] = [0.08, 0.92]
    dist[1,1,:] = [0.07, 0.93]
    E_distribution = ConditionalDiscreteDistribution(nodes=[C, D, E], table=dist)
    E.set_dist(E_distribution)
    
    engine = EnumerationEngine(net)
#    engine1.evidence[wear] = False
#    engine1.evidence[weap] = True
    engine.evidence[A] = True
    engine.evidence[C] = True
    Q = engine.marginal(E)[0]
    index = Q.generate_index([True], range(Q.nDims))
    prob = Q[index]
    print prob
    
    
    
def make_exam_net():
    H = BayesNode(0,2,name='H')
    G = BayesNode(1,2,name='G')
    B = BayesNode(2,2,name='B')
    O = BayesNode(3,2,name='O')
    D = BayesNode(4,2,name='D')
    C = BayesNode(5,2,name='C')
    
    H.add_child(B)
    B.add_parent(H)
    B.add_parent(G)
    G.add_child(B)
    G.add_child(O)
    O.add_parent(G)
    O.add_child(D)
    O.add_child(C)
    D.add_parent(O)
    C.add_parent(O)
    B.add_child(D)
    D.add_parent(B)
    
    nodes = [B,C,D,G,H,O]
    net = BayesNet(nodes)
    
    sci = BayesNode(0,2,name='sci')
    inf = BayesNode(1,2,name='inf')
    weap = BayesNode(2,2,name='weap')
    car = BayesNode(3,2,name='car')
    wear = BayesNode(4,2,name='wear')
    
    sci.add_child(weap)
    weap.add_parent(sci)
    inf.add_child(weap)
    weap.add_parent(inf)
    weap.add_child(car)
    car.add_parent(weap)
    weap.add_child(wear)
    wear.add_parent(weap)
    
    nodes1 = [sci, weap, wear, inf, car]
    net1 = BayesNet(nodes1)
    
    B = net.get_node_by_name('B')
    C = net.get_node_by_name('C')
    D = net.get_node_by_name('D')
    G = net.get_node_by_name('G')
    H = net.get_node_by_name('H')
    O = net.get_node_by_name('O')
    
    
    H_distribution = DiscreteDistribution(H)
    index = H_distribution.generate_index([],[])
    H_distribution[index] = [0.6, 0.4]
    H.set_dist(H_distribution)
    
    G_distribution = DiscreteDistribution(G)
    index = G_distribution.generate_index([],[])
    G_distribution[index] = [0.75, 0.25]
    G.set_dist(G_distribution)
    
    dist = zeros([O.size(), C.size()], dtype=float32)
    dist[0,:] = [0.55, 0.45]
    dist[1,:] = [0.75, 0.25]
    C_distribution = ConditionalDiscreteDistribution(nodes=[O,C], table=dist)
    C.set_dist(C_distribution)
    
    dist = zeros([G.size(), O.size()], dtype=float32)
    dist[0,:] = [0.55, 0.45]
    dist[1,:] = [0.45, 0.55]
    O_distribution = ConditionalDiscreteDistribution(nodes=[G,O], table=dist)
    O.set_dist(O_distribution)
    
    dist = zeros([B.size(), O.size(), D.size()], dtype=float32)
    dist[0,0,:] = [0.72, 0.28]
    dist[0,1,:] = [0.38, 0.62]
    dist[1,0,:] = [0.85, 0.15]
    dist[1,1,:] = [0.65, 0.35]
    D_distribution = ConditionalDiscreteDistribution(nodes=[B, O, D], table=dist)
    D.set_dist(D_distribution)
    
    dist = zeros([H.size(), G.size(), B.size()], dtype=float32)
    dist[0,0,:] = [0.92, 0.08]
    dist[0,1,:] = [0.75, 0.25]
    dist[1,0,:] = [0.55, 0.45]
    dist[1,1,:] = [0.35, 0.65]
    B_distribution = ConditionalDiscreteDistribution(nodes=[H, G, B], table=dist)
    B.set_dist(B_distribution)
    
    sci = net1.get_node_by_name('sci')
    weap = net1.get_node_by_name('weap')
    wear = net1.get_node_by_name('wear')
    inf = net1.get_node_by_name('inf')
    car = net1.get_node_by_name('car')
    
    sci_distribution = DiscreteDistribution(sci)
    index = sci_distribution.generate_index([],[])
    sci_distribution[index] = [0.2, 0.8]
    sci.set_dist(sci_distribution)
    
    inf_distribution = DiscreteDistribution(inf)
    index = inf_distribution.generate_index([],[])
    inf_distribution[index] = [0.4, 0.6]
    inf.set_dist(inf_distribution)
    
    dist = zeros([sci.size(), inf.size(), weap.size()], dtype=float32)
    dist[0,0,:] = [0.6, 0.4]
    dist[0,1,:] = [0.8, 0.2]
    dist[1,0,:] = [0.1, 0.9]
    dist[1,1,:] = [0.3, 0.7]
    weap_distribution = ConditionalDiscreteDistribution(nodes=[sci, inf, weap], table=dist)
    weap.set_dist(weap_distribution)
    
    dist = zeros([weap.size(), wear.size()], dtype=float32)
    dist[0,:] = [0.88, 0.12]
    dist[1,:] = [0.15, 0.85]
    wear_distribution = ConditionalDiscreteDistribution(nodes=[weap, wear], table=dist)
    wear.set_dist(wear_distribution)
    
    dist = zeros([weap.size(), car.size()], dtype=float32)
    dist[0,:] = [0.75, 0.25]
    dist[1,:] = [0.55, 0.45]
    car_distribution = ConditionalDiscreteDistribution(nodes=[weap, car], table=dist)
    car.set_dist(car_distribution)
    
    
##    engine = JunctionTreeEngine(net)
#    engine = EnumerationEngine(net)
##    engine.evidence[B] = True
#    Q = engine.marginal(C)[0]
#    index = Q.generate_index([True], range(Q.nDims))
#    prob = Q[index]
#    print "Thr ptob of O = T given B = T is  ", prob
    
    engine1 = EnumerationEngine(net1)
    engine1.evidence[wear] = False
    engine1.evidence[weap] = True
 #   engine1.evidence[sci] = False
    Q = engine1.marginal(car)[0]
    index = Q.generate_index([True], range(Q.nDims))
    prob = Q[index]
    print prob
    
    

def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system."""    
    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")
    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]
    # TODO: set the probability distribution for each node
    
    #Set distribution of node FA
    FA_distribution = DiscreteDistribution(F_A_node)
    index = FA_distribution.generate_index([], [])
    FA_distribution[index] = [0.85, 0.15]
    F_A_node.set_dist(FA_distribution)
    
    #Set distribution of node T
    T_distribution = DiscreteDistribution(T_node)
    index = T_distribution.generate_index([], [])
    T_distribution[index] = [0.8, 0.2]
    T_node.set_dist(T_distribution)
    
    #Set distribution of FG given T
    dist = zeros([T_node.size(), F_G_node.size()], dtype=float32)
    dist[0, :] = [0.95, 0.05]
    dist[1, :] = [0.2, 0.8]
    FG_distribution = ConditionalDiscreteDistribution(nodes=[T_node, F_G_node], table=dist)
    F_G_node.set_dist(FG_distribution)
    
    #set distribution of A given FA and G
    dist = zeros([F_A_node.size(), G_node.size(), A_node.size()], dtype=float32)
    dist[0,0,:] = [0.9, 0.1]
    dist[0,1,:] = [0.1, 0.9]
    dist[1,0,:] = [0.55, 0.45]
    dist[1,1,:] = [0.45, 0.55]
    A_distribution = ConditionalDiscreteDistribution(nodes=[F_A_node, G_node, A_node], table=dist)
    A_node.set_dist(A_distribution)
    
    #Set distribution of G given FG and T
    dist = zeros([F_G_node.size(), T_node.size(), G_node.size()], dtype=float32)
    dist[0,0,:] = [0.95, 0.05]
    dist[0,1,:] = [0.05, 0.95]
    dist[1,0,:] = [0.2, 0.8]
    dist[1,1,:] = [0.8, 0.2]
    G_distribution = ConditionalDiscreteDistribution(nodes=[F_G_node, T_node, G_node], table=dist)
    G_node.set_dist(G_distribution)
    
    return bayes_net

def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal 
    probability of the alarm 
    ringing (T/F) in the 
    power plant system."""
    # TODO: finish this function
    A_node = bayes_net.get_node_by_name('alarm')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([alarm_rings], range(Q.nDims))
    alarm_prob = Q[index]
    return alarm_prob


def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge 
    showing hot (T/F) in the 
    power plant system."""
    # TOOD: finish this function
    G_node = bayes_net.get_node_by_name('gauge')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(G_node)[0]
    index = Q.generate_index([gauge_hot], range(Q.nDims))
    gauge_prob = Q[index]
    return gauge_prob


def get_temperature_prob(bayes_net,temp_hot):
    """Calculate the conditional probability 
    of the temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    T_node = bayes_net.get_node_by_name('temperature')
    A_node = bayes_net.get_node_by_name('alarm')
    FA_node = bayes_net.get_node_by_name('faulty alarm')
    FG_node = bayes_net.get_node_by_name('faulty gauge')
    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A_node] = True
    engine.evidence[FA_node] = False
    engine.evidence[FG_node] = False
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([temp_hot], range(Q.nDims))
    temp_prob = Q[index]
    return temp_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    
    # TODO: fill this out
    A = BayesNode(0,4,name='A')
    B = BayesNode(1,4,name='B')
    C = BayesNode(2,4,name='C')
    AB = BayesNode(3,3,name='AvB')
    BC = BayesNode(4,3,name='BvC')
    CA = BayesNode(5,3,name='CvA')
    A.add_child(AB)
    A.add_child(CA)
    B.add_child(AB)
    B.add_child(BC)
    C.add_child(BC)
    C.add_child(CA)
    AB.add_parent(A)
    AB.add_parent(B)
    BC.add_parent(B)
    BC.add_parent(C)
    CA.add_parent(A)
    CA.add_parent(C)
    
    # Set A distribution
    A_distribution = DiscreteDistribution(A)
    index = A_distribution.generate_index([],[])
    A_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    A.set_dist(A_distribution)
    
    # Set B distribution
    B_distribution = DiscreteDistribution(B)
    index = B_distribution.generate_index([],[])
    B_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    B.set_dist(B_distribution)
    
    # Set C distribution
    C_distribution = DiscreteDistribution(C)
    index = C_distribution.generate_index([],[])
    C_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    C.set_dist(C_distribution)
    
    # Set distribution of AvB given A and B
    dist = zeros([A.size(), B.size(), AB.size()], dtype=float32)
    dist[0,0,:] = [0.10, 0.10, 0.80]
    dist[0,1,:] = [0.20, 0.60, 0.20]
    dist[0,2,:] = [0.15, 0.75, 0.10]
    dist[0,3,:] = [0.05, 0.90, 0.05]
    dist[1,0,:] = [0.60, 0.20, 0.20]
    dist[1,1,:] = [0.10, 0.10, 0.80]
    dist[1,2,:] = [0.20, 0.60, 0.20]
    dist[1,3,:] = [0.15, 0.75, 0.10]
    dist[2,0,:] = [0.75, 0.15, 0.10]
    dist[2,1,:] = [0.60, 0.20, 0.20]
    dist[2,2,:] = [0.10, 0.10, 0.80]
    dist[2,3,:] = [0.20, 0.60, 0.20]
    dist[3,0,:] = [0.90, 0.05, 0.05]
    dist[3,1,:] = [0.75, 0.15, 0.10]
    dist[3,2,:] = [0.60, 0.20, 0.20]
    dist[3,3,:] = [0.10, 0.10, 0.80]
    AB_distribution = ConditionalDiscreteDistribution(nodes=[A, B, AB], table=dist)
    AB.set_dist(AB_distribution)
    
    # Set distribution of BvC given B and C
    dist = zeros([B.size(), C.size(), BC.size()], dtype=float32)
    dist[0,0,:] = [0.10, 0.10, 0.80]
    dist[0,1,:] = [0.20, 0.60, 0.20]
    dist[0,2,:] = [0.15, 0.75, 0.10]
    dist[0,3,:] = [0.05, 0.90, 0.05]
    dist[1,0,:] = [0.60, 0.20, 0.20]
    dist[1,1,:] = [0.10, 0.10, 0.80]
    dist[1,2,:] = [0.20, 0.60, 0.20]
    dist[1,3,:] = [0.15, 0.75, 0.10]
    dist[2,0,:] = [0.75, 0.15, 0.10]
    dist[2,1,:] = [0.60, 0.20, 0.20]
    dist[2,2,:] = [0.10, 0.10, 0.80]
    dist[2,3,:] = [0.20, 0.60, 0.20]
    dist[3,0,:] = [0.90, 0.05, 0.05]
    dist[3,1,:] = [0.75, 0.15, 0.10]
    dist[3,2,:] = [0.60, 0.20, 0.20]
    dist[3,3,:] = [0.10, 0.10, 0.80]
    BC_distribution = ConditionalDiscreteDistribution(nodes=[B, C, BC], table=dist)
    BC.set_dist(BC_distribution)
    
    # Set distribution of CA giveen C and A
    dist = zeros([C.size(), A.size(), CA.size()], dtype=float32)
    dist[0,0,:] = [0.10, 0.10, 0.80]
    dist[0,1,:] = [0.20, 0.60, 0.20]
    dist[0,2,:] = [0.15, 0.75, 0.10]
    dist[0,3,:] = [0.05, 0.90, 0.05]
    dist[1,0,:] = [0.60, 0.20, 0.20]
    dist[1,1,:] = [0.10, 0.10, 0.80]
    dist[1,2,:] = [0.20, 0.60, 0.20]
    dist[1,3,:] = [0.15, 0.75, 0.10]
    dist[2,0,:] = [0.75, 0.15, 0.10]
    dist[2,1,:] = [0.60, 0.20, 0.20]
    dist[2,2,:] = [0.10, 0.10, 0.80]
    dist[2,3,:] = [0.20, 0.60, 0.20]
    dist[3,0,:] = [0.90, 0.05, 0.05]
    dist[3,1,:] = [0.75, 0.15, 0.10]
    dist[3,2,:] = [0.60, 0.20, 0.20]
    dist[3,3,:] = [0.10, 0.10, 0.80]
    CA_distribution = ConditionalDiscreteDistribution(nodes=[C, A, CA], table=dist)
    CA.set_dist(CA_distribution)
    
    nodes = [A, B, C, AB, BC, CA]    
    return BayesNet(nodes)

def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function    
    A = bayes_net.get_node_by_name('A')
    B = bayes_net.get_node_by_name('B')
    C = bayes_net.get_node_by_name('C')
    AB = bayes_net.get_node_by_name('AvB')
    BC = bayes_net.get_node_by_name('BvC')
    CA = bayes_net.get_node_by_name('CvA')
    
    engine = EnumerationEngine(bayes_net)
    engine.evidence[AB] = 0
    engine.evidence[CA] = 2
    Q = engine.marginal(BC)[0]
    for val in range(3):
        index = Q.generate_index([val], range(Q.nDims))
        posterior[val] = Q[index]
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    # TODO: finish this function
    A = bayes_net.get_node_by_name('A')
    A_table = A.dist.table
    B = bayes_net.get_node_by_name('B')
    B_table = B.dist.table
    C = bayes_net.get_node_by_name('C')
    C_table = C.dist.table
    AB = bayes_net.get_node_by_name('AvB')
    AB_table = AB.dist.table
    BC = bayes_net.get_node_by_name('BvC')
    BC_table = BC.dist.table
    CA = bayes_net.get_node_by_name('CvA')
    CA_table = CA.dist.table
    if (len(initial_state) == 0) or (initial_state == None):
        initial_state = []
        for i in range(3):
            initial_val = rand.randint(0,3)
            initial_state.append(initial_val)
        for i in range(3,6):
            initial_val = rand.randint(0,2)
            initial_state.append(initial_val)
    initial_state[3] = 0
    initial_state[5] = 2
    val_to_change = rand.choice([0,1,2,4])
    if val_to_change == 0:
        pA = zeros([A.size()], dtype=float32)
        for indx in range(4):
            pA[indx] = A_table[indx]*AB_table[indx,initial_state[1],0]*CA_table[initial_state[2],indx,2]
        pA = np.divide(pA, np.sum(pA, dtype=float32))
        new_val = np.random.choice([0,1,2,3], p=pA)
        initial_state[val_to_change] = new_val
    elif val_to_change == 1:
        pB = zeros([B.size()], dtype=float32)
        for indx in range(4):
            pB[indx] = B_table[indx]*AB_table[initial_state[0],indx,0]*BC_table[indx,initial_state[2],initial_state[4]]
        pB = np.divide(pB, np.sum(pB, dtype=float32))
        new_val = np.random.choice([0,1,2,3], p=pB)
        initial_state[val_to_change] = new_val
    elif val_to_change == 2:
        pC = zeros([C.size()], dtype=float32)
        for indx in range(4):
            pC[indx] = C_table[indx]*BC_table[initial_state[2],indx,initial_state[4]]*CA_table[indx,initial_state[0],2]
        pC = np.divide(pC, np.sum(pC, dtype=float32))
        new_val = np.random.choice([0,1,2,3], p=pC)
        initial_state[val_to_change] = new_val
    elif val_to_change == 4:
        pBC = BC_table[initial_state[1], initial_state[2],:]
        new_val = np.random.choice([0,1,2], p=pBC)
        initial_state[val_to_change] = new_val
    sample = tuple(initial_state)    
    return sample

def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A = bayes_net.get_node_by_name('A')
    A_table = A.dist.table
    B = bayes_net.get_node_by_name('B')
    B_table = B.dist.table
    C = bayes_net.get_node_by_name('C')
    C_table = C.dist.table
    AB = bayes_net.get_node_by_name('AvB')
    AB_table = AB.dist.table
    BC = bayes_net.get_node_by_name('BvC')
    BC_table = BC.dist.table
    CA = bayes_net.get_node_by_name('CvA')
    CA_table = CA.dist.table
    if (len(initial_state) == 0) or (initial_state == None):
        initial_state = []
        for i in range(3):
            initial_val = rand.randint(0,3)
            initial_state.append(initial_val)
        for i in range(3,6):
            initial_val = rand.randint(0,2)
            initial_state.append(initial_val)
    initial_state[3] = 0
    initial_state[5] = 2
#    print "initial state", initial_state
    previous_initial_state = initial_state[:]
#    vals_to_change = np.random.choice([0,1,2,4],4)
#    for val in vals_to_change:
    for val in [0,1,2,4]:
        if (val == 0) or (val == 1) or (val == 2):
            initial_state[val] = rand.randint(0,3)
        elif val == 4:
            initial_state[val] = rand.randint(0,2)
#    A_indx, B_indx, C_indx, AB_indx, BC_indx, CA_indx = previous_initial_state
    A_indx = previous_initial_state[0]
    B_indx = previous_initial_state[1]
    C_indx = previous_initial_state[2]
    AB_indx = previous_initial_state[3]
    BC_indx = previous_initial_state[4]
    CA_indx = previous_initial_state[5]
    previous_pi = A_table[A_indx]*B_table[B_indx]*C_table[C_indx]*AB_table[A_indx,B_indx,0]*BC_table[B_indx,C_indx,BC_indx]*CA_table[C_indx,A_indx,2]
#    A_indx, B_indx, C_indx, AB_indx, BC_indx, CA_indx = initial_state
    A_indx = initial_state[0]
    B_indx = initial_state[1]
    C_indx = initial_state[2]
    AB_indx = initial_state[3]
    BC_indx = initial_state[4]
    CA_indx = initial_state[5]
    pi = A_table[A_indx]*B_table[B_indx]*C_table[C_indx]*AB_table[A_indx,B_indx,0]*BC_table[B_indx,C_indx,BC_indx]*CA_table[C_indx,A_indx,2]
    alpha = min(1,(pi/previous_pi))
    if alpha == 1:
        sample = tuple(initial_state)
    else:
        if rand.uniform(0,1) < alpha:
            sample = tuple(initial_state)
        else:
            sample = tuple(previous_initial_state)
#            print "rejected"
#            print "previous initial state", previous_initial_state
    # TODO: finish this function
#    print initial_state
#    print previous_initial_state
    return sample


def compare_sampling(bayes_net,initial_state, delta):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    
    N_Gibbs = 0
    prev_Gibbs_convergence = [0,0,0]
    N = [0,0,0]
    burn_in = 0
    initial_state_Gibbs = initial_state[:]
    initial_state_MH = initial_state[:]
    while True:
        Gibbs_count +=1
        state = Gibbs_sampler(bayes_net,initial_state_Gibbs)
        x = state[4]
        N[x] += 1
#        print N
        Gibbs_convergence = N[:]
        Gibbs_convergence = np.divide(Gibbs_convergence,float(sum(Gibbs_convergence)))
        if ((N[0] and N[1]) or (N[0] and N[2]) or (N[1] and N[2])) and (burn_in > 1000):
            if (abs(Gibbs_convergence[0]-prev_Gibbs_convergence[0]) <= 0.00001) and (abs(Gibbs_convergence[1]-prev_Gibbs_convergence[1]) <= 0.00001) and (abs(Gibbs_convergence[2]-prev_Gibbs_convergence[2]) <= 0.00001):
                N_Gibbs += 1
                if N_Gibbs == 150:
                    break
            else:
                N_Gibbs = 0
        prev_Gibbs_convergence = Gibbs_convergence[:]
        burn_in += 1
#        initial_state_Gibbs = list(state)
    N_MH = 0
    prev_MH_convergence = [0,0,0]
    N = [0,0,0]
    burn_in = 0
    while True:
        MH_count +=1
        state = MH_sampler(bayes_net, initial_state_MH)
        x = state[4]
        N[x] += 1
#        print N
        MH_convergence = N[:]
        MH_convergence = np.divide(MH_convergence, float(sum(MH_convergence)))
        if ((N[0] and N[1]) or (N[0] and N[2]) or (N[1] and N[2])) and (burn_in > 1000):
            if (abs(MH_convergence[0]-prev_MH_convergence[0]) <= 0.00001) and (abs(MH_convergence[1]-prev_MH_convergence[1]) <= 0.00001) and (abs(MH_convergence[2]-prev_MH_convergence[2]) <= 0.00001):
                N_MH += 1
                if N_MH == 150:
                    break
            else:
                N_MH = 0
        prev_MH_convergence = MH_convergence[:]
        burn_in += 1
#        print "initial state", initial_state_MH
#        print "state", state
        if state == tuple(initial_state_MH):
            MH_rejection_count += 1
        initial_state_MH = list(state)
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count

def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 1
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1.068
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Dan Monga Kilanga"
