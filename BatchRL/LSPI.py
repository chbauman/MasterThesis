import numpy as np
import scipy
import scipy.stats

def phi_linear(s, a):
    """
    returns the evaluations of all the basis functions
    in a vector.
    Linear model.
    """
    return np.hstack([1, s, a])

def phi_quadratic(s, a):
    """
    returns the evaluations of all the basis functions
    in a vector.
    Quadratic model.
    """
    return np.hstack([1, s, a, s * s, a * s, a * a])


def phi_quadratic(s, a):
    """
    returns the evaluations of all the basis functions
    in a vector.
    Quadratic model.
    """

    return np.hstack([1, s, a, s * s, a * s, a * a])

def phi_handcraft(s, a):
    """
    returns the evaluations of all the basis functions
    in a vector.
    Handcrafted model.
    """
    # Handcrafted features
    ind1 = 1 if s[1] < 3 else 0
    ind2 = 1 if s[1] > 0 else 0
    ind3 = 1 if s[0] == 3 or s[0] == 4 else 0
    ind4 = 1 if s[0] == 8 else 0
    a_one_hot = np.zeros((3))
    a_one_hot[a] = 1
    return np.hstack([1, s, s * s, a_one_hot, ind1, ind2, ind3, ind4])


def phi_state_action01(s, a):
    """
    returns the evaluations of all the basis functions
    in a vector.
    Handcrafted model.
    """
    # Handcrafted features
    tot_n_sa_pairs = 10 * 4 * 3 + 1
    ind = int(s[0]) * 4 * 3 + int(s[1]) * 3 + a
    a_one_hot = np.zeros((tot_n_sa_pairs))
    a_one_hot[ind] = 1
    a_one_hot[-1] = 1
    return a_one_hot

class LSPI:
    """
    From the paper: 
        Least-Squares Policy Iteration, 
        by: Michail G. Lagoudakis, Ronald Parr 

    If stoch_policy_imp = True, then the stochastic 
    policy improvement is used. From the paper:
        Non-Deterministic Policy Improvement Stabilizes Approximated Reinforcement Learning,
        by: Wendelin Böhmer, Rong Guo and Klaus Obermayer
    """

    def __init__(self,
                 state_dim,
                 nb_actions,
                 phi = phi_state_action01,
                 discount_factor = 0.7,
                 max_iters=500,
                 stoch_policy_imp = False,
                 stochasticity_beta = 2,
                 accur = 0.01):

        """
        Init function, stores all required parameters.
        """
        self.state_dim = state_dim
        self.nb_actions = nb_actions
        self.phi = phi
        self.disc_fac = discount_factor
        self.max_iters = max_iters    
        self.accur = accur
        s_dum = np.zeros(state_dim)
        self.n_basis_fun = phi(s_dum, 0).shape[0]
        self.stoch_policy_imp = stoch_policy_imp
        self.stochasticity_beta = stochasticity_beta

    def fit(self, D_s, D_a, D_r, D_s_prime):
        """
        Fit the Q-function.
        """

        # Initialize 
        n_samples = D_r.shape[0]
        n = self.n_basis_fun
        w = np.zeros((n))
        A = np.zeros((n, n))
        b = np.zeros((n))
        for k in range(n_samples):
            b += self.phi(D_s[k], D_a[k])
        
        phi_s_a = np.zeros((self.nb_actions))

        elements = np.arange(self.nb_actions)

        # Iterate
        for i in range(self.max_iters):

            # Assemble A
            for k in range(n_samples):

                # Compute \phi(s, a)
                phi_res = np.reshape(self.phi(D_s[k], D_a[k]), (n, 1))

                # Compute \pi(s')
                for j in range(self.nb_actions):
                    phi_s_a[j] = np.dot(self.phi(D_s_prime[k], j), w)

                if self.stoch_policy_imp:
                    # Stochastic Policy improvement
                    phi_s_a = scipy.stats.zscore(phi_s_a)
                    prob_actions = scipy.special.softmax(self.stochasticity_beta * phi_s_a)
                    pi_s = np.random.choice(elements, 1, p=prob_actions)
                else:
                    # Greedy policy improvement
                    pi_s = np.argmax(phi_s_a)

                # Compute \gamma * \phi(s', \pi(s'))
                phi_pi = np.reshape(self.phi(D_s_prime[k], pi_s), (n, 1))
                phi_pi = phi_pi * self.disc_fac
                
                # Update A
                A += phi_res * np.transpose(phi_res - phi_pi)

            # Solve LSE
            #w_new = np.linalg.solve(A, b, rcond=None)
            w_new = np.linalg.lstsq(A, b, rcond=None)[0]

            # Test convergence
            w_diff = np.linalg.norm(w - w_new)
            if w_diff < self.accur:
                print("Converged!!")
                break
            lam = 0.2
            w = lam * w_new + (1 - lam) * w
            print("Difference:", w_diff, "Lam:", lam)

            # Reset for next iteration
            A = np.zeros((n, n))

        self.w = w
        pass

    def get_policy(self):
        """
        Returns the fitted policy:
        a = \argmax_a \phi(s, a)^T w
        """
        def policy(s_t):
            # Computes \phi(s, a) for all a and finds
            # argmax.
            phi_s_a = np.zeros((self.nb_actions))
            for j in range(self.nb_actions):
                phi_s_a[j] = np.dot(self.phi(s_t, j), self.w)
            return np.argmax(phi_s_a)
        
        return policy