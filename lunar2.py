import gym
import numpy as np
import random
import time
import math
from NeuroEvolve import *


# Create a Neural Network using Custom API.
nn=NeuroES(8,4,softmax)
#nn.add_layer(4,relu)
#nn.add_layer(10,sigmoid)
#nn.add_layer(10,relu)
nn.completed_network()

#Load the environment
env = gym.make('LunarLander-v2')
env.reset()
done = False
action = 0
reward = 0

#Genetic Algorithm specific hyperparameters
pop_size = 150
cr_rate = 0.4
m_rate = 3
consecutive=5

# Debugging Variabes, not used!
Min_reward=1
Optimal_reward=-100
best=Min_reward
act=[0,1,2,3]
counter=2

# Get the number of weights from the neural network.
gene_size=nn.get_weight_count()




# Initialize population of weights values between (0,1)
def gen_pop():
    pop = []
    for i in range(pop_size):
        weights = np.random.uniform(size=gene_size)
        pop.append(weights)

    return pop


# The crossover of Top members, can be anything as long as new variations of best genes are available.
def crossover(po):
    pop = []
    topp = int(cr_rate * len(po))
    top = [po[x][0] for x in range(0, topp)]
    for i in range(0, topp):
        pop.append(top[i])
    j = 0
    s_topp = topp + (topp) / 2
    for i in range(topp, topp + (topp) / 2):
        a = top[j]
        b = top[j + 1]
        c = []
        for k in range(len(a)):
            c.append((a[k] + b[k]) / 2)
        j = j + 2
        pop.append(c)
    j = 0
    for i in range(pop_size - (s_topp)):
        k=random.randint(1,topp-1)

        j=random.randint(1,topp-1)
        s=[x for x in top[k]]      # x can be multiplied with some noise.
        b=[x for x in top[j]]      # x can be multiplied with some noise.
        r=[]
        sw=0
        for l in range(len(s)):
            if sw==0:
                r.append(s[l])
                sw=1
            else:
                r.append(b[l])
                sw=0
        pop.append(r)
    return pop


# Mutation , slightly mutate weights of some members.
def mutation(po):
    for i in range(len(po)):
        c = po[i]
        for j in range(len(c)):
            if m_rate > random.randint(0, 120):
                c[j] = c[j]+(np.random.uniform()*2-1)/10
                #print "Mutated"
    return po

#Some more variables.
avg=0
pop = gen_pop()
gen = 0
observation = [0,0,0,0,0,0,0,0]
ep_no = 0
#To Store winning neural network.
winner = []
win = 0
actp=0

#The main loop
while True:
    #Training part, else is the winner gameplay.
    if win == 0:
        gen += 1
        i = 0
        n_pop = []
        while i < pop_size:
            if win == 1:
                break
            cand = pop[i]
            av=0
            #Play the same neural network for consecutive games to check its average perfomance, which is also its fitness
            for j in range(consecutive):
                total_reward = 0
                t_r=0
                #Each observation of the game is fed into the network and each action is given to the environment.
                while done == False:
                    nn.set_weights(cand)
                    outp=nn.evaluate(observation)
                    actp=np.argmax(outp)
                    observation, reward, done, info = env.step(actp)
                    t_r+=reward
                av+=t_r
                #Restart the environment.
                env.reset()
                ep_no += 1
                done = False
            #Add the candidate and its perfomance to a list
            n_pop.append([cand, float(av)/consecutive])
            #Check if there is a winner candidate with average perfomance above 200.
            if float(av)/consecutive>=200:
                winner = cand
                win = 1
                print "DONE:", t_r, ep_no
                break
            i += 1
        if win==1:
            continue

        # Sort the population based on their perfomance.
        n_pop = sorted(n_pop, key=lambda x: x[1])
        n_pop = n_pop[::-1]
        avg=0
        #Calculate average perfomance through the population.
        for i in range(len(n_pop)):
            avg+=n_pop[i][1]
        avg=avg/float(pop_size)
        best_reward = n_pop[0][1]
        #Check if there is a winner.
        if best_reward>=200:
            winner = n_pop[0][0]
            win = 1
            print "DONE:", t_r, ep_no
            continue
        # Crossover and mutation of the population
        pop = crossover(n_pop)
        pop = mutation(pop)
        #Print data to check for convergence.
        if ep_no%50==0:
            print gen, best_reward, ep_no,avg
    else:
        # The winner plays here. Winner weights are stored in winner variable.
        if done == True:
            print winner
            env.reset()
            done = False
        nn.set_weights(winner)
        outp = nn.evaluate(observation)
        actp = np.argmax(outp)
        observation, reward, done, info = env.step(actp)
        time.sleep(0.05)
        env.render()

