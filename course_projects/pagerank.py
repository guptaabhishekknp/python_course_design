"""
Assumption about data set: Line i in data file, provides the list to the pages that page i has outlinks to. This must be said explicitly.
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import math
import copy

linelist = [line.rstrip('\n') for line in open("data.txt")]
links = [l.split(':', 1)[-1].split(',') for l in linelist]

num_vert = len(links)
H = np.zeros((num_vert,num_vert))
A = np.zeros((num_vert,num_vert))

for j in range(num_vert):
    if(links[j] == ['']): #for a dangling node, populate the appropriate column of A
        for i in range(num_vert):
            A[i][j] = 1/num_vert
        continue
    f = 1/len(links[j])
    for p in links[j]:
        H[int(p)][j] = f #for a non-dangling node, populate the appropriate column of H

alpha = 0.85
O = np.full((num_vert,num_vert),1/num_vert)
I = np.multiply(1/num_vert, np.ones(num_vert))
G = np.multiply(alpha,H) + np.multiply(alpha,A) + np.multiply(1-alpha,O)
H_original = copy.deepcopy(H)

while True:
    I_k = np.matmul(G,I)
    if(all(I == I_k)):
        break
    I = I_k
    
I_original = I

plt.plot(sorted(I, reverse=True))
plt.ylabel('PageRank Importance')
plt.xlabel('Ordinal Rank')
plt.savefig('PageRankImportance_vs_OrdinalRank.png')
plt.show()

d = []
I_k = np.multiply(1/num_vert, np.ones(num_vert))

while any(I-I_k): 
    d.append(math.sqrt(reduce(lambda x, y: x**2+y**2, I-I_k)))
    I_k = np.matmul(G,I_k)
    
plt.plot(np.array(d))
plt.ylabel('Euclidean distance d_k.png')
plt.xlabel('Iteration number k')
plt.savefig('Convergence_for_alpha_0.85.png')
plt.show()

d = []
d_values = []
alpha_start = 0.01 
alpha_stop = 0.99
alpha_step = 0.01

alpha = alpha_start

while alpha <= alpha_stop+0.001: #really lame reason for that 0.01 - floating points are not precise
    G = np.multiply(alpha,H) + np.multiply(alpha,A) + np.multiply(1-alpha,O)
    I = np.multiply(1/num_vert, np.ones(num_vert))
    while True: 
        I_k = np.matmul(G,I)
        d.append(math.sqrt(reduce(lambda x, y: x**2+y**2, I-I_k)))
        if d[-1] < 10**-15:
            break
        I = I_k
    d_values.append(len(d))
    d = []
    alpha += alpha_step
    
alpha_values = np.arange(alpha_start, alpha_stop+alpha_step, alpha_step)
plt.plot(alpha_values, np.array(d_values))
plt.ylabel('Required iterations k.png')
plt.xlabel('alpha')
plt.savefig('Convergence_dependence_on_alpha.png')
plt.show()

website_id = 10 #choose between 0 and 999
ordinal_rank = num_vert-[x[0] for x in sorted([(i,I_original[i]) for i in range(num_vert)], key= lambda im: im[1])].index(website_id)
print("Website id: "+str(website_id)+" Original Ordinal Rank:", ordinal_rank, "Original Importance", I_original[website_id])
threshold_percentage = 0.05
alpha = 0.85
s = sorted([(i,I_original[i]) for i in range(num_vert)], key=lambda vec: vec[1])
s = s[:int(threshold_percentage*num_vert)]
new_pagerank = []

for v in s:
    if str(website_id) in links[v[0]]:
        new_pagerank.append((v[0],I_original[website_id],ordinal_rank)) 
        continue
    if links[int(v[0])] == ['']:
        new_outlinks = 1
    else:
        new_outlinks = len(links[int(v[0])])+1
    H = np.copy(H_original)
    A = np.zeros((num_vert,num_vert))
    for i in range(num_vert):
        if(H[i][int(v[0])] != 0 or i == website_id): #populate the column of H corresponding to this website
            H[i][int(v[0])] = 1/new_outlinks
    for j in range(num_vert):
        if any([H[i][j] for i in range(num_vert)]): #If any entry in the j-th column is non zero then move to the next column
            continue
        for i in range(num_vert): #if all the entries are zero then populate the corresponding column of A
            A[i][j] = 1/num_vert
    I = np.multiply(1/num_vert, np.ones(num_vert))        
    G = np.multiply(alpha,H) + np.multiply(alpha,A) + np.multiply(1-alpha,O)
    
    while True:
        I_k = np.matmul(G,I)
        if(all(I==I_k)): 
            
            print(v[0],sum(abs(I-I_k)))
            break
        I = I_k
    new_ordinal_rank = num_vert-[x[0] for x in sorted([(i,I[i]) for i in range(num_vert)], key= lambda im: im[1])].index(website_id) #sort the new importance vector by importance, 1000-index of website id is the ordinal rank. Instead of 1000-*, we can give reverse=True in the sorted function
    new_pagerank.append((v[0],I[website_id],new_ordinal_rank)) #id of website, new importance after paying that website, new ordinal rank

result = sorted(new_pagerank, key=lambda vec: vec[2]) #sort the result by the ordinal ranking that paying that particular website yields
for v in result: 
    print(str(website_id)+" Pays:", v[0], "New Ordinal Rank:", v[2], "New Importance:", v[1])
    