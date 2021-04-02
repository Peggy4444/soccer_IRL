#!/usr/bin/env python
# coding: utf-8

# This script compute feature expectation, applies GIRL algorithm, recovers reward weights, and computes loss


import numpy as np
from qpsolvers import solve_qp
from scipy import linalg
from scipy import optimize
import scipy
from tqdm import tqdm
import scipy.linalg as scila
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    args = parser.parse_args()


path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path

directory= 'soccer_dataset/'



df_rew_features= pd.read_excel(directory + 'df_rew_features.xlsx')


reward_features_df= df_rew_features[['bayern_rank_inverse','opponent_rank_inverse', 'bayern_side_id', 'goal_diff', 'norm_time_remaining', 'player_market_value' ]]



reward_features_array= reward_features_df.to_numpy()


# Feature expectation



GAMMA = args.gamma
def feature_expectations(reward_features_array, GAMMA):
    discount_factor_timestep = np.power(GAMMA * np.ones(reward_features_array.shape[0]),
                                        range(reward_features_array.shape[0]))
    discounted_return = discount_factor_timestep[np.newaxis, :, np.newaxis] * reward_features_array
    reward_est_timestep = np.sum(discounted_return, axis=1)
    return reward_est_timestep



# import gradients




gradients_df= pd.read_excel(directory + 'gradients_df.xlsx')
gradients_arr= gradients_df.to_numpy()
mean_gradients = np.mean(gradients_arr, axis=0)


# compute psi: feature expectation * gradients




mean_gradients= mean_gradients.reshape(4,1)
psi= np.dot(mean_gradients,reward_feature_vector)



# recover feature weights: omegas, and compute IRL loss



#taken from the paper: Truly Batch Model-Free Inverse Reinforcement Learning about Multiple Intentions: http://proceedings.mlr.press/v108/ramponi20a/ramponi20a.pdf 


def solve_PGIRL(estimated_gradients, verbose=False, solver='quadprog', seed=1234,):
    num_episodes, num_parameters, num_objectives = estimated_gradients.shape[:]
    mean_gradients = np.mean(estimated_gradients, axis=0)
    ns = scipy.linalg.null_space(mean_gradients)
    P = np.dot(mean_gradients.T, mean_gradients)
    if ns.shape[1] > 0:
        if (ns >= 0).all() or (ns <= 0).all():

            print("Jacobian has a null space:", ns[:, 0] / np.sum(ns[:, 0]))
            weights = ns[:, 0] / np.sum(ns[:, 0])
            loss = np.dot(np.dot(weights.T, P), weights)
            return weights, loss
        else:
            weights = solve_polyhedra(ns)
            print("Null space:", ns)
            if weights is not None and (weights!=0).any():
                print("Linear programming sol:", weights)
                weights = np.dot(ns, weights.T)
                weights = weights / np.sum(weights)
                loss = np.dot(np.dot(weights.T, P), weights)
                print("Weights from non positive null space:", weights)
                return weights, loss
            else:
                print("Linear prog did not find positive weights")

    q = np.zeros(num_objectives)
    A = np.ones((num_objectives, num_objectives))
    b = np.ones(num_objectives)
    G = np.diag(np.diag(A))
    h = np.zeros(num_objectives)
    normalized_P = P / np.linalg.norm(P)
    try:
        weights = solve_qp(P, q, -G, h, A=A, b=b, solver=solver)
    except ValueError:
        try:
            weights = solve_qp(normalized_P, q, -G, h, A=A, b=b, solver=solver)
        except:
            #normalize matrix

            print("Error in Girl")
            print(P)
            print(normalized_P)
            u, s, v = np.linalg.svd(P)
            print("Singular Values:", s)
            ns = scipy.linalg.null_space(mean_gradients)
            print("Null space:", ns)

            weights, loss = solve_girl_approx(P, seed=seed)
    loss = np.dot(np.dot(weights.T, P), weights)
    if verbose:
        print('loss:', loss)
        print(weights)
    return weights, loss

