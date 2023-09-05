import numpy as np
import matplotlib.pyplot as plt

class CMAES:
    def __init__(self,objective_function, nVar,VarMin,VarMax):
        self.objective_function = objective_function
        self.nVar = nVar
        self.VarMin = VarMin
        self.VarMax = VarMax
        self.MaxIt = 1000
        self.e = 0.0001

        # Population Size (and Number of Offsprings)
        self.lambda_ = (4 + round(3 * np.log(self.nVar))) * 10 * 4
        # Number of Parents
        self.mu = round(self.lambda_ / 2)

        # Parent Weights
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / np.sum(w)

        # Number of Effective Solutions
        self.mu_eff = 1 / np.sum(self.w ** 2)

        # Step Size Control Parameters (c_sigma and d_sigma)
        self.sigma0 = 0.3 * (VarMax - VarMin)  # 目前是多余的
        self.cs = (self.mu_eff + 2) / (self.nVar + self.mu_eff + 5)
        self.ds = 1 + self.cs + 2 * max(np.sqrt((self.mu_eff - 1) / (self.nVar + 1)) - 1, 0)
        self.ENN = np.sqrt(self.nVar) * (1 - 1 / (4 * self.nVar) + 1 / (21 * self.nVar ** 2))

        # Covariance Update Parameters
        self.cc = (4 + self.mu_eff / nVar) / (4 + nVar + 2 * self.mu_eff / nVar)
        self.c1 = 2 / ((nVar + 1.3) ** 2 + self.mu_eff)
        self.alpha_mu = 2
        # min?
        self.cmu = min(1 - self.c1, self.alpha_mu * (self.mu_eff - 2 + 1 / self.mu_eff) / ((nVar + 2) ** 2 + self.alpha_mu * self.mu_eff / 2))
        self.hth = (1.4 + 2 / (nVar + 1)) * self.ENN

        # Initialization
        self.ps = np.zeros((self.MaxIt, self.nVar))
        self.pc = np.zeros((self.MaxIt, self.nVar))
        self.C = np.array([np.eye(self.nVar) for _ in range(self.MaxIt)])
        self.sigma = np.full(self.MaxIt, self.sigma0)

        self.M = [{'Position': np.random.uniform(self.VarMin, self.VarMax, self.nVar),
                   'Step': np.zeros(self.nVar),
                   'Cost': float('inf')} for _ in range(self.MaxIt)]

        self.M[0]['Cost'] = self.objective_function(self.M[0]['Position'])

        self.BestSol = self.M[0].copy()
        self.BestCost = np.zeros(self.MaxIt)

    # CMA-ES Main Loop
    def run(self):
        out_cnt = 0  # stop sign
        for g in range(self.MaxIt):
            # Generate lambda offspring
            pop = []
            for i in range(self.lambda_):
                step = np.random.multivariate_normal(np.zeros(self.nVar), self.C[g])
                position = self.M[g]['Position'] + self.sigma[g] * step
                cost = self.objective_function(position)
                pop.append({'Position': position, 'Step': step, 'Cost': cost})

            # Sort population
            pop.sort(key=lambda x: x['Cost'])

            # Update Best Solution Ever Found
            if pop[0]['Cost'] < self.BestSol['Cost']:
                self.BestSol = pop[0].copy()

            self.BestCost[g] = self.BestSol['Cost']

            # Check for early stop
            if g > 0 and abs(self.BestCost[g] - self.BestCost[g - 1]) < self.e:
                out_cnt += 1
                if out_cnt >= 10:
                    break
            else:
                out_cnt = 0

            # Update other parameters
            # Update Mean (M)
            new_step = np.zeros(self.nVar)
            for j in range(self.mu):
                new_step += self.w[j] * pop[j]['Step']

            new_position = self.M[g]['Position'] + self.sigma[g] * new_step
            new_cost = self.objective_function(new_position)

            self.M[g + 1]['Position'] = new_position
            self.M[g + 1]['Step'] = new_step
            self.M[g + 1]['Cost'] = new_cost

            if new_cost < self.BestSol['Cost']:
                self.BestSol = self.M[g + 1].copy()

            # Update Step Size (sigma)
            self.ps[g + 1] = (1 - self.cs) * self.ps[g] + np.sqrt(
                self.cs * (2 - self.cs) * self.mu_eff) * np.dot(new_step,np.linalg.inv(np.linalg.cholesky(self.C[g]).T))
            self.sigma[g + 1] = self.sigma[g] * np.exp(
                (self.cs / self.ds) * (np.linalg.norm(self.ps[g + 1]) / self.ENN - 1)) ** 0.3

            # Update Covariance Matrix (C)
            #hth = (1.4 + 2 / (self.nVar + 1)) * self.ENN
            if np.linalg.norm(self.ps[g + 1]) / np.sqrt(1 - (1 - self.cs) ** (2 * (g + 1))) < self.hth:
                hs = 1
            else:
                hs = 0

            #cc = (4 + self.mu_eff / self.nVar) / (self.nVar + 4 + 2 * self.mu_eff / self.nVar)
            #c1 = 2 / ((self.nVar + 1.3) ** 2 + self.mu_eff)
            #cmu = min(1 - c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.nVar + 2) ** 2 + self.mu_eff))
            delta = (1 - hs) * self.cc * (2 - self.cc)

            pc = (1 - self.cc) * self.pc[g] + hs * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * new_step
            self.pc[g + 1] = pc
            self.C[g + 1] = (1 - self.c1 - self.cmu) * self.C[g] + self.c1 * (
                        pc.reshape(-1, 1) @ pc.reshape(1, -1) + delta * self.C[g])

            for j in range(self.mu):
                self.C[g + 1] += self.cmu * self.w[j] * pop[j]['Step'].reshape(-1, 1) @ pop[j]['Step'].reshape(1, -1)

            # Eigen Decomposition for Repairing C
            eig_values, eig_vectors = np.linalg.eig(self.C[g + 1])
            # If Covariance Matrix is not Positive Defenite or Near Singular
            if np.any(np.diag(eig_values) < 0):
                eig_values = np.maximum(eig_values, 0)
                self.C[g + 1] = eig_vectors @ np.diag(eig_values) @ np.linalg.inv(eig_vectors)
            #self.C[g + 1] = eig_vectors @ np.diag(np.maximum(eig_values, 0)) @ np.linalg.inv(eig_vectors)
        print(self.BestSol)

        # plt.figure()
        # plt.semilogy(-self.BestCost, linewidth=2)
        # plt.xlabel('Iteration')
        # plt.ylabel('Best Cost')
        # plt.grid(True)
        # plt.show()
        return self.BestSol

######################################## Test Function
def rosenbrock_function(x):
    x1, x2, x3 = x
    a = 1
    b = 100
    c = 100
    return (a - x1) ** 2 + b * (x2 - x1 ** 2) ** 2 + c * (x3 - x2 ** 2) ** 2

# multi varibale -2
def schaffer(x):
    '''
    This function has plenty of local minimum, with strong shocks
    global minimum at (0,0) with value 0
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    '''
    x1, x2 = x
    part1 = np.square(x1) - np.square(x2)
    part2 = np.square(x1) + np.square(x2)
    return 0.5 + (np.square(np.sin(part1)) - 0.5) / np.square(1 + 0.001 * part2)

# multi varibale -6
def test_function(x):
    x1, x2, x3, x4, x5, x6 = x
    return (
        -np.exp(-((x1 - 2)**2 + (x2 - 1)**2)) +
        -np.exp(-((x3 + 3)**2 + (x4 + 1)**2)) +
        -np.exp(-((x5 - 4)**2 + (x6 - 2)**2))
    )

if __name__ == "__main__":
    VarMin =-5
    VarMax = 5
    cmaes = CMAES(test_function,6,VarMin,VarMax)
    cmaes.run()
