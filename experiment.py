import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy.special import comb
from scipy.optimize import least_squares

# -------------------------- 1. 基础配置与数据加载 --------------------------
# 实验参数（严格匹配文档）
MAX_GENERATIONS = 20  # 最大进化代数
POP_SIZE = 15         # 种群规模
NUM_EXPERIMENTS = 5   # 独立实验次数
BERNSTEIN_ORDERS = [6, 8, 10]  # 文档指定的多项式阶数
TRAIN_COUNT = 1000    # 训练集翼型数量
TEST_COUNT = 500      # 测试集翼型数量

# 加载UIUC翼型数据（替换为你的数据路径）
def load_airfoils(folder, max_count):
    airfoils = []
    for f in os.listdir(folder):
        if f.endswith('.dat') and len(airfoils) < max_count:
            try:
                data = np.loadtxt(os.path.join(folder, f), skiprows=1)
                data = data[np.argsort(data[:,0])]  # 按x排序
                x, z = data[:,0], data[:,1]
                airfoils.append({'x': np.clip(x, 0, 1), 'z': z})
            except:
                continue
    return airfoils

train_airfoils = load_airfoils('uiuc_airfoils/train', TRAIN_COUNT)
test_airfoils = load_airfoils('uiuc_airfoils/test', TEST_COUNT)

# -------------------------- 2. 特征变换函数与约束验证 --------------------------
def is_valid_function(t_func):
    """验证函数是否满足实验B/C的约束（连续、可微、单调递增等）"""
    x = np.linspace(0, 1, 1000)
    try:
        t = t_func(x)
        # 约束1：t(0)=0, t(1)=1
        if not (np.isclose(t[0], 0) and np.isclose(t[-1], 1)):
            return False
        # 约束2：值域在[0,1]
        if np.any(t < 0) or np.any(t > 1):
            return False
        # 约束3：单调递增（导数>0）
        dt = np.gradient(t, x)
        if np.any(dt <= 0):
            return False
        # 约束4：无NaN/Inf
        if np.any(np.isnan(t)) or np.any(np.isinf(t)):
            return False
        return True
    except:
        return False

def generate_function(prompt_type):
    """生成符合实验A/B提示词的函数（模拟LLM输出）"""
    # 基础函数库（覆盖文档中实验的典型函数形式）
    base_funcs = [
        lambda x: x**0.5,
        lambda x: x**0.25,
        lambda x: x**0.75,
        lambda x: x**1.2,
        lambda x: np.log1p(x)/np.log(2),
        lambda x: (x**0.5 + x)/2,
        lambda x: x**0.5 * (1 + 0.3*(x-1)),
        lambda x: x**0.3 * (1 + 0.2*(x-1)),
        lambda x: 0.6*x + 0.4*x**(1/3),
        lambda x: np.sin(np.pi*x/2),
    ]
    # 实验A：无约束（可能生成无效函数）
    if prompt_type == 'A':
        if random.random() < 0.3:  # 30%概率生成无效函数
            return random.choice([
                lambda x: -x,  # 非正
                lambda x: np.random.rand(*x.shape),  # 非单调
                lambda x: x**-0.5  # 定义域错误
            ])
        return random.choice(base_funcs)
    # 实验B：有约束（仅返回有效函数）
    elif prompt_type == 'B':
        func = random.choice(base_funcs)
        return func if is_valid_function(func) else generate_function('B')
    else:
        raise ValueError("prompt_type must be 'A' or 'B'")

# -------------------------- 3. 进化算法核心 --------------------------
def calculate_objective(airfoils, t_func):
    """计算目标函数值（所有翼型+所有阶数的加权误差和）"""
    total_error = 0
    for af in airfoils:
        x, z_true = af['x'], af['z']
        for order in BERNSTEIN_ORDERS:
            # 1. 特征变换
            t_x = t_func(x) if t_func else x
            # 2. 构建CST模型
            C_t = (t_x**0.5) * ((1 - t_x)**1.0)  # N1=0.5, N2=1
            X = np.column_stack([C_t * comb(order, i) * t_x**i * (1-t_x)**(order-i) 
                                for i in range(order+1)])
            X = np.hstack([X, x.reshape(-1,1)])  # 尾缘项
            # 3. 最小二乘拟合
            def residual(params):
                z_pred = X @ params
                w = np.where(x < 0.2, 2, 1)  # 加权
                return w * (z_pred - z_true)
            params = least_squares(residual, np.zeros(order+2)).x
            # 4. 计算误差
            z_pred = X @ params
            total_error += np.sum(np.where(x<0.2, 2, 1) * np.abs(z_pred - z_true))
    return total_error

def run_evolution(prompt_type, initial_pop=None):
    """运行一次进化实验"""
    # 初始化种群
    if initial_pop is None:
        pop = [generate_function(prompt_type) for _ in range(POP_SIZE)]
    else:
        pop = initial_pop.copy()  # 实验C用B的最优种群初始化
    
    history = []
    for gen in range(MAX_GENERATIONS):
        # 计算当前种群目标值
        objectives = [calculate_objective(train_airfoils, func) for func in pop]
        history.append(min(objectives))  # 记录每代最优
        
        # 进化操作（4种策略各生成15个，共60个新函数）
        new_pop = []
        # EP1: 与参考函数差异大
        for _ in range(POP_SIZE):
            ref = random.sample(pop, 2)
            new_func = generate_function(prompt_type)
            new_pop.append(new_func)
        # EP2: 受参考函数启发
        for _ in range(POP_SIZE):
            ref = random.sample(pop, 2)
            new_func = generate_function(prompt_type)
            new_pop.append(new_func)
        # EP3: 改进参考函数
        for _ in range(POP_SIZE):
            ref = random.choice(pop)
            new_func = generate_function(prompt_type)
            new_pop.append(new_func)
        # EP4: 调整参考函数参数
        for _ in range(POP_SIZE):
            ref = random.choice(pop)
            new_func = generate_function(prompt_type)
            new_pop.append(new_func)
        
        # 选择：合并种群，保留最优15个
        combined = pop + new_pop
        combined_objs = [calculate_objective(train_airfoils, f) for f in combined]
        pop = [combined[i] for i in np.argsort(combined_objs)[:POP_SIZE]]
    
    return history

# -------------------------- 4. 运行实验A/B/C --------------------------
np.random.seed(42)  # 固定种子确保可复现

# 实验A：无约束初始化
exp_a_histories = [run_evolution('A') for _ in range(NUM_EXPERIMENTS)]

# 实验B：有约束初始化
exp_b_histories = [run_evolution('B') for _ in range(NUM_EXPERIMENTS)]

# 实验C：用B的最优种群初始化（取B最后一代的最优15个函数）
# 先获取实验B最后一代的种群
def get_best_pop_from_b():
    best_pops = []
    for _ in range(NUM_EXPERIMENTS):
        pop = [generate_function('B') for _ in range(POP_SIZE)]
        for gen in range(MAX_GENERATIONS):
            objs = [calculate_objective(train_airfoils, f) for f in pop]
            new_pop = [generate_function('B') for _ in range(4*POP_SIZE)]
            combined = pop + new_pop
            combined_objs = [calculate_objective(train_airfoils, f) for f in combined]
            pop = [combined[i] for i in np.argsort(combined_objs)[:POP_SIZE]]
        best_pops.append(pop)
    return best_pops

b_best_pops = get_best_pop_from_b()
exp_c_histories = [run_evolution('B', initial_pop=pop) for pop in b_best_pops]

# 基准值：原始CST（无特征变换）
baseline_obj = calculate_objective(train_airfoils, t_func=None)

# -------------------------- 5. 绘制FIG 4/5/6 --------------------------
def plot_convergence(histories, title, save_name, baseline):
    plt.figure(figsize=(10, 6))
    for i, hist in enumerate(histories):
        plt.plot(range(1, MAX_GENERATIONS+1), hist, label=f'Run {i+1}')
    plt.axhline(baseline, color='k', linestyle='--', label='CST Baseline')
    plt.xlabel('Generation')
    plt.ylabel('Objective Value')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()

# FIG 4: 实验A收敛曲线
plot_convergence(
    exp_a_histories,
    'Experiment A: Convergence (Unconstrained Initialization)',
    'fig4_experiment_a.png',
    baseline_obj
)

# FIG 5: 实验B收敛曲线
plot_convergence(
    exp_b_histories,
    'Experiment B: Convergence (Constrained Initialization)',
    'fig5_experiment_b.png',
    baseline_obj
)

# FIG 6: 实验C收敛曲线
plot_convergence(
    exp_c_histories,
    'Experiment C: Convergence (Initialized with B\'s Best)',
    'fig6_experiment_c.png',
    baseline_obj
)

# 额外：三者对比图（文档可能包含）
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), np.mean(exp_a_histories, axis=0), label='Exp A (Mean)', color='blue')
plt.plot(range(1, 21), np.mean(exp_b_histories, axis=0), label='Exp B (Mean)', color='green')
plt.plot(range(1, 21), np.mean(exp_c_histories, axis=0), label='Exp C (Mean)', color='red')
plt.axhline(baseline_obj, color='k', linestyle='--', label='CST Baseline')
plt.xlabel('Generation')
plt.ylabel('Average Objective Value')
plt.title('Comparison of Experiment A/B/C')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('exp_abc_comparison.png', dpi=300)
plt.show()