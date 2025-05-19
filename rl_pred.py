import torch
import torch.nn as nn
from torch.distributions import Bernoulli # Для бинарного выбора (ядро включено/выключено)
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim): # action_dim = num_cpu_cores (битовая маска ядер)
        super(ActorCritic, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Actor - предсказывает вероятности для каждого ядра (Bernoulli)
        self.actor_probs = nn.Linear(128, action_dim)

        self.critic_head = nn.Linear(128, 1)

    def forward(self, state):
        x = self.shared_layer(state)
        action_probs = torch.sigmoid(self.actor_probs(x)) # Sigmoid for probabilities
        state_value = self.critic_head(x)
        return action_probs, state_value

GLOBAL_PRIO = 1000000000 # Начальное значение

class PPOAgent:
    def __init__(self, state_dim, num_cpu_cores, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=80, eps_clip=0.2, device="cpu", affinity_threshold=0.5):
        self.device = torch.device(device)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.num_cpu_cores = num_cpu_cores # Number of CPU cores
        self.action_dim = num_cpu_cores  # Action is a bitmask for cores
        self.affinity_threshold = affinity_threshold # Threshold for setting affinity
        self.MseLoss = nn.MSELoss()
        # Создаем Actor-Critic сеть
        self.policy = ActorCritic(state_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.shared_layer.parameters(), 'lr': lr_actor},
                        {'params': self.policy.actor_probs.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic_head.parameters(), 'lr': lr_critic}]
                    )

        self.policy_old = ActorCritic(state_dim, self.action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.episode_buffer = {'states': [], 'actions': [], 'log_probs': [], 'values': []}

    def _get_state(self, node):
        """Преобразует узел графа в вектор состояния."""
        # Пример простого состояния:
        cpu_times = [pm.length_mcs for pm in node.perfmodels if pm.worker_type == 'CPU']
        avg_cpu_time = np.mean(cpu_times) if cpu_times else 1e9
        n_cpu_perf = len(cpu_times)
        n_in = len(node.inc)
        n_out = len(node.out)

        # Добавим количество CPU cores в состояние
        state = np.array([avg_cpu_time, n_cpu_perf, n_in, n_out, self.num_cpu_cores], dtype=np.float32)
        return state

    def select_action(self, state, node):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_probs, state_value = self.policy_old(state_tensor)

        # Bernoulli distribution for each core
        dist = Bernoulli(action_probs)

        # Sample which cores to use (0 or 1 for each core)
        actions = dist.sample() # shape: (num_cpu_cores)
        log_probs = dist.log_prob(actions).sum() # Sum log probs for all cores

        self.episode_buffer['states'].append(state)
        self.episode_buffer['actions'].append(actions.cpu().numpy()) # Store numpy array
        self.episode_buffer['log_probs'].append(log_probs.item())
        self.episode_buffer['values'].append(state_value.item())

        return actions.cpu().numpy() # Return action

    def schedule(self, graph):
        self.global_prio_counter = GLOBAL_PRIO
        self.episode_buffer = {'states': [], 'actions': [], 'log_probs': [], 'values': []}

        nodes_to_schedule = list(graph.nodes.items())

        for _, node in nodes_to_schedule:
            if node.name == 'NULL':
                continue

            # Get state
            state = self._get_state(node)

            # Select which cores to use
            core_selection = self.select_action(state, node) # core_selection: numpy array of 0s and 1s

            # Set device affinity based on core selection
            # Важно: Здесь нужно знать, как workerid соотносятся с ядрами CPU
            # Предположим, что workerid 0 соответствует ядру 0, workerid 1 - ядру 1, и т.д.
            device_affinity = []
            for i, pm in enumerate(node.perfmodels):
                if pm.worker_type == 'CPU':
                    core_index = int(pm.workerid) # workerid to core index

                    # Если ядро выбрано агентом, разрешаем его использование
                    if core_index < self.num_cpu_cores and core_selection[core_index] >= self.affinity_threshold:
                        device_affinity.append(True)
                    else:
                        device_affinity.append(False)
                else:
                    device_affinity.append(False) # Disable non-CPU workers

            # Если affinity пустой (агент не выбрал ни одного ядра), разрешаем все CPU
            if not any(device_affinity):
                cpu_affinity = [pm.worker_type == 'CPU' for pm in node.perfmodels]
                device_affinity = [c or False for c in cpu_affinity]
                if not any(device_affinity): # If still empty, allow all
                    device_affinity = [True] * len(node.perfmodels)

            node.device_affinity = device_affinity

            # Set priority (базовая приоритизация)
            node.prio = self.global_prio_counter
            self.global_prio_counter -= 1

    def update(self, buffer):
         # --- Расчет Returns и Advantages (немного меняется для continuous actions)---
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.stack([torch.from_numpy(s) for s in buffer.states]).to(self.device).detach()
        # Action is a *vector* of binary choices, so we don't need to unsqueeze
        old_actions = torch.tensor(np.stack(buffer.actions), dtype=torch.float32).to(self.device).detach()  # Stack numpy arrays
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32).to(self.device).detach()
        old_values = torch.tensor(buffer.values, dtype=torch.float32).to(self.device).detach()

        advantages = rewards - old_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        for _ in range(self.K_epochs):
            # Get values from current policy
            action_probs, state_values = self.policy(old_states)
            dist = Bernoulli(action_probs) # Pass probabilities
            log_probs = dist.log_prob(old_actions).sum(dim=1) # Calculate log prob, sum over cores
            dist_entropy = dist.entropy().mean() # Mean entropy
            state_values = torch.squeeze(state_values)

            # Calculate ratio
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = 0.5 * self.MseLoss(state_values, rewards)

            # Total loss
            loss = loss_actor + loss_critic - 0.01*dist_entropy # Subtract entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())

#Trainer

from collections import deque # Для FIFO-буфера

class AgentTrainer:
    def __init__(self, agent, ppo_buffer, gamma=0.99, update_timestep=2000, log_interval=10): # agent - это PPOAgent
        self.agent = agent
        self.ppo_buffer = ppo_buffer
        self.gamma = gamma
        self.update_timestep = update_timestep # Как часто обновляем политику
        self.log_interval = log_interval # Как часто логируем метрики
        self.timestep_counter = 0
        self.episode_reward = 0
        self.episode_length = 0
        self.episodes = 0

        self.reward_queue = deque(maxlen=100)  # Очередь для усреднения наград
        self.length_queue = deque(maxlen=100)

    def step(self, graph, timings, inds):
        """
        Выполняется после получения таймингов от выполнения графа.
        Вычисляет награду, сохраняет опыт в буфер и обновляет агента.
        """

        # 1. Вычисляем награду
        # Начнем с простой награды: -makespan (общее время выполнения)
        reward = -timings  # Инвертируем, чтобы минимизировать время
        self.episode_reward += reward

        # 2. Определяем, является ли это концом эпизода
        # В нашем случае, эпизод заканчивается после выполнения всего графа.
        done = True # Всегда True, потому что один граф = один эпизод
        terminal = True # Используем для GAE

        # 3. Достаем данные из agent.episode_buffer и сохраняем в ppo_buffer
        # Важно: agent.episode_buffer уже содержит state, action, log_prob, value
        states = self.agent.episode_buffer['states']
        actions = self.agent.episode_buffer['actions']
        log_probs = self.agent.episode_buffer['log_probs']
        values = self.agent.episode_buffer['values']

        # Проверяем, что длины совпадают
        if not (len(states) == len(actions) == len(log_probs) == len(values)):
            print("WARNING: Длины states, actions, log_probs, values не совпадают!")
            print(f"{len(states)=}, {len(actions)=}, {len(log_probs)=}, {len(values)=}")

        # Сохраняем данные в ppo_buffer для *каждого* узла в графе
        for state, action, log_prob, value in zip(states, actions, log_probs, values):
            self.ppo_buffer.store(state, action, log_prob, value, reward, done, terminal)
            self.timestep_counter += 1
            self.episode_length += 1

        # 4. Обновляем агента, если накопили достаточно опыта
        if self.timestep_counter >= self.update_timestep:
            self.agent.update(self.ppo_buffer)
            self.ppo_buffer.clear() # Очищаем буфер после обновления
            self.timestep_counter = 0 # Сбрасываем счетчик

        # 5. Логирование (периодическое)
        if done:  # Эпизод закончился
            self.episodes += 1
            self.reward_queue.append(self.episode_reward)
            self.length_queue.append(self.episode_length)
            avg_reward = sum(self.reward_queue) / len(self.reward_queue)
            avg_length = sum(self.length_queue) / len(self.length_queue)

            if self.episodes % self.log_interval == 0:
                print(f"Episode {self.episodes}: Reward = {self.episode_reward:.2f}, Avg Reward = {avg_reward:.2f}, Length = {self.episode_length}, Avg Length = {avg_length:.2f}")

            # Сбрасываем статистики эпизода
            self.episode_reward = 0
            self.episode_length = 0

#Runner            
state_dim = 5  # Заменить на фактическую размерность вектора состояния
file = f"{HOMEDIR}/graph_tracer/traced_graph_2_small_layers.json"
ncpus = 4
ncuda = 0
niters = 20
manual_sampling = True
verbose = True
with open(file, "r") as f:
    config = json.load(f)

operations = config["calls"]
total_operations = len(operations)
print(total_operations)
agent = PPOAgent(state_dim=5, device="cpu" if ncuda==0 else "cuda", num_cpu_cores=ncpus) # n_cpus и n_gpus важно указать
ppo_buffer = PPOBuffer()
trainer = AgentTrainer(agent, ppo_buffer, update_timestep=total_operations) # Обновляем после каждого графа
gr = GraphRunner(file, ncpus, ncuda, niters, manual_sampling, verbose=verbose, home_dir=HOMEDIR)
gr.run(interactive=True, starpu_home=STARPU_HOME)
_ = gr.skip_initialization_routine()
rng = np.random.default_rng()
timings_dynamic = []
for i in range(gr.niters):
    # inds = get_subset_inds(rng, total_operations)
    inds = "all" # just use this if you want whole graph
    print(f"{i=} {inds=}")
    graph = gr.next(inds=inds)

    # print("Nodes: ", len(graph.nodes))

    agent.schedule(graph)

    gr.serialize_priorities(graph)
    timings = gr.wait_get_timings()

    trainer.step(graph, timings, inds) # Передаем timings и inds в step

    timings_dynamic.append(timings)
