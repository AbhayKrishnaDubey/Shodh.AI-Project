# fixed_lending_rl_with_baselines.py
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ----------------------------
# 1) Mock data (keep runnable)
# ----------------------------
np.random.seed(0)
N_SAMPLES = 20000
n_features = 17

X = np.random.randn(N_SAMPLES, n_features).astype(np.float32)

financials = np.zeros((N_SAMPLES, 3), dtype=np.float32)
for i in range(N_SAMPLES):
    loan_amt = np.random.randint(5000, 30000)
    is_good = 1 if np.random.rand() > 0.2 else 0
    if is_good:
        payment = loan_amt * (1 + np.random.uniform(0.05, 0.20))
    else:
        payment = loan_amt * np.random.uniform(0.0, 0.4)
    financials[i] = [loan_amt, payment, is_good]

# ----------------------------
# 2) Train/test & scaler
# ----------------------------
X_train, X_test, fin_train, fin_test = train_test_split(X, financials, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# 3) Portfolio Environment
# ----------------------------
class LendingPortfolioEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, X, financials, portfolio_size=32, deny_cost=-0.01, risk_penalty=1.0, seed=None):
        super().__init__()
        self.X = X.astype(np.float32)
        self.financials = financials.astype(np.float32)
        self.n = X.shape[0]
        self.portfolio_size = int(portfolio_size)
        self.deny_cost = float(deny_cost)
        self.risk_penalty = float(risk_penalty)
        self.rng = np.random.RandomState(seed)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.X.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.indices = None
        self.ptr = None

    def reset(self, seed=None, options=None):
        self.indices = self.rng.choice(self.n, size=self.portfolio_size, replace=False)
        self.ptr = 0
        idx = self.indices[self.ptr]
        return self.X[idx], {}

    def step(self, action):
        idx = self.indices[self.ptr]
        loan_amt = float(self.financials[idx, 0])
        total_pymnt = float(self.financials[idx, 1])
        profit = total_pymnt - loan_amt
        scaled_profit = profit / 1000.0

        if action == 0:
            reward = self.deny_cost
            info = {"outcome": "denied", "loan_idx": int(idx), "profit": 0.0}
        else:
            if scaled_profit >= 0:
                reward = scaled_profit
            else:
                reward = scaled_profit - self.risk_penalty * (abs(scaled_profit) ** 2)
            info = {"outcome": "approved", "loan_idx": int(idx), "profit": profit}

        self.ptr += 1
        done = False
        if self.ptr >= self.portfolio_size:
            done = True
            next_obs = np.zeros_like(self.X[0], dtype=np.float32)
        else:
            next_idx = self.indices[self.ptr]
            next_obs = self.X[next_idx]

        # gymnasium returns (obs, reward, terminated, truncated, info)
        return next_obs, float(reward), bool(done), False, info

# ----------------------------
# 4) Create VecEnv and model
# ----------------------------
def make_env():
    return LendingPortfolioEnv(X_train, fin_train, portfolio_size=32, deny_cost=-0.01, risk_penalty=1.0, seed=None)

n_envs = 8
vec_env = DummyVecEnv([make_env for _ in range(n_envs)])

model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4, n_steps=2048//n_envs, batch_size=64, gae_lambda=0.95)

print("Training PPO... (this may take a few minutes)")
model.learn(total_timesteps=200000)
print("Done training.")

# ----------------------------
# 5) Evaluation function (robust predict handling)
# ----------------------------
def evaluate_policy_on_test(model, X_test, fin_test, portfolio_size=32, n_episodes=500):
    env = LendingPortfolioEnv(X_test, fin_test, portfolio_size=portfolio_size, deny_cost=-0.01, risk_penalty=1.0)
    total_profit = 0.0
    loans_approved = 0
    defaults_approved = 0
    returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # robustly handle different policy types (SB3, DummyPolicy, etc.)
            pred = model.predict(obs, deterministic=True)
            # pred might be (action, state) or a direct integer (in some dummy cases)
            if pred is None:
                action = int(np.random.randint(0, 2))
            elif isinstance(pred, tuple) and len(pred) >= 1:
                action = pred[0]
                if isinstance(action, (np.ndarray, list)):
                    action = int(np.array(action).ravel()[0])
                else:
                    action = int(action)
            else:
                # fallback
                action = int(pred)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            if action == 1:
                profit = info.get("profit", 0.0)
                total_profit += float(profit)
                loans_approved += 1
                returns.append(float(profit))
                if profit < 0:
                    defaults_approved += 1

            obs = next_obs

    default_rate = (defaults_approved / loans_approved) if loans_approved else 0.0
    avg_profit_per_approved = (total_profit / loans_approved) if loans_approved else 0.0
    return {
        "total_profit": total_profit,
        "loans_approved": loans_approved,
        "defaults_approved": defaults_approved,
        "default_rate": default_rate,
        "avg_profit_per_approved": avg_profit_per_approved,
        "returns": returns
    }

# Random baseline evaluator
def evaluate_random(X_test, fin_test, portfolio_size=32, n_episodes=800):
    env = LendingPortfolioEnv(X_test, fin_test, portfolio_size=portfolio_size)
    total_profit = 0.0
    loans_approved = 0
    defaults_approved = 0
    returns = []
    rng = np.random.RandomState(123)
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = int(rng.randint(0, 2))
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            if action == 1:
                profit = info.get("profit", 0.0)
                total_profit += float(profit)
                loans_approved += 1
                returns.append(float(profit))
                if profit < 0:
                    defaults_approved += 1
            obs = next_obs
    default_rate = (defaults_approved / loans_approved) if loans_approved else 0.0
    avg_profit_per_approved = (total_profit / loans_approved) if loans_approved else 0.0
    return {
        "total_profit": total_profit,
        "loans_approved": loans_approved,
        "defaults_approved": defaults_approved,
        "default_rate": default_rate,
        "avg_profit_per_approved": avg_profit_per_approved,
        "returns": returns
    }

# ----------------------------
# 6) Evaluate PPO and baselines (fixed)
# ----------------------------
print("Evaluating PPO policy on test set...")
ppo_results = evaluate_policy_on_test(model, X_test, fin_test, portfolio_size=32, n_episodes=800)

# Baselines:
class DummyPolicy:
    def __init__(self, action):
        self.action = int(action)
    def predict(self, obs, deterministic=True):
        return int(self.action), None

print("Evaluating baselines...")
approve_all_results = evaluate_policy_on_test(DummyPolicy(1), X_test, fin_test, portfolio_size=32, n_episodes=800)
random_results = evaluate_random(X_test, fin_test, portfolio_size=32, n_episodes=800)

# Print summary
def print_results(name, res):
    print(f"{name:<12} | Total Profit: ${res['total_profit']:,.2f} | Loans Approved: {res['loans_approved']:<6} | Default Rate: {res['default_rate']:.2%} | Avg Profit/Approved: ${res['avg_profit_per_approved']:,.2f}")

print_results("PPO Agent", ppo_results)
print_results("Approve All", approve_all_results)
print_results("Random", random_results)

# Plot histogram of returns (dollars) for PPO and Approve All
plt.figure(figsize=(10,6))
plt.hist(ppo_results['returns'], bins=60, alpha=0.7, label='PPO Agent Returns')
plt.hist(approve_all_results['returns'], bins=60, alpha=0.2, label='Approve All Returns')
plt.axvline(0, color='red', linestyle='--')
plt.title("Distribution of Returns on Approved Loans (PPO Agent vs Approve All)")
plt.xlabel("Profit/Loss ($)")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# -----------------------------------------
# Compute average reward from PPO results
# -----------------------------------------

def compute_average_reward(model, X_test, fin_test, portfolio_size=32, n_episodes=500):
    env = LendingPortfolioEnv(X_test, fin_test, portfolio_size=portfolio_size)
    
    total_reward = 0.0
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = int(action.ravel()[0])

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            total_reward += reward
            total_steps += 1

    avg_reward = total_reward / total_steps
    return avg_reward, total_reward, total_steps

ppo_avg_reward, ppo_total_reward, ppo_steps = compute_average_reward(
    model, X_test, fin_test, portfolio_size=32, n_episodes=500
)

print("\n--- RL Reward Statistics ---")
print(f"Average Reward per Decision: {ppo_avg_reward:.4f}")
print(f"Total Reward Accumulated: {ppo_total_reward:.4f}")
print(f"Total Steps: {ppo_steps}")
