import numpy as np

def selection(mask_all, fitness): 
    prob = fitness / np.sum(fitness)
    rnd = np.random.rand(1)
    try:
        best_prob = np.max(fitness[prob>rnd])
        idx= np.where(prob == best_prob)[0].item()
    except:
        idx = np.argmax(prob)
    mask_=mask_all[idx]
    return mask_.copy()

def crossover(mask_all, fitness, L):
    individual1_mask = selection(mask_all, fitness)
    individual2_mask = selection(mask_all, fitness)
    idx = np.random.randint(0, L, 2)
    start_idx, end_idx = np.min(idx), np.max(idx)
    individual1_mask_copy = individual1_mask.copy()
    individual2_mask_copy = individual2_mask.copy()
    individual1_mask_copy[start_idx: end_idx] = individual2_mask[start_idx: end_idx]
    individual2_mask_copy[start_idx: end_idx] = individual1_mask[start_idx: end_idx]
    return individual1_mask_copy, individual2_mask_copy

def mutation(mask_all, fitness,L):
    individual_mask = selection(mask_all, fitness)
    idx = np.random.randint(0, L, 2)
    start_idx, end_idx = np.min(idx), np.max(idx)
    individual_mask_copy=individual_mask.copy()
    individual_mask_copy[start_idx: end_idx] = -individual_mask[start_idx: end_idx]
    return individual_mask_copy
