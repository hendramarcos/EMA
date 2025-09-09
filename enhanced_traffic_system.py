from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import optparse
import random
import serial
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from datetime import datetime
from collections import deque, defaultdict
import pickle

# SUMO imports
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

class PerformanceLogger:
    """
    Comprehensive logging system for tracking agent performance and system metrics
    """
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Initialize logging dictionaries
        self.episode_logs = []
        self.step_logs = []
        self.agent_logs = defaultdict(list)
        self.communication_logs = []
        
        # Performance metrics tracking
        self.metrics = {
            'total_waiting_time': [],
            'average_speed': [],
            'throughput': [],
            'queue_lengths': [],
            'fuel_consumption': [],
            'emissions': [],
            'coordination_efficiency': [],
            'reward_evolution': defaultdict(list)
        }
        
    def log_step(self, step, agents_data, global_metrics):
        """Log data for each simulation step"""
        step_data = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'agents': agents_data,
            'global_metrics': global_metrics
        }
        self.step_logs.append(step_data)
        
    def log_episode(self, episode, episode_metrics):
        """Log episode-level metrics"""
        episode_data = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'metrics': episode_metrics
        }
        self.episode_logs.append(episode_data)
        
    def log_agent_action(self, agent_id, state, action, reward, next_state, info):
        """Log individual agent actions and decisions"""
        agent_data = {
            'agent_id': agent_id,
            'timestamp': datetime.now().isoformat(),
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': action,
            'reward': reward,
            'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            'info': info
        }
        self.agent_logs[agent_id].append(agent_data)
        
    def log_communication(self, sender_id, receiver_id, message_type, content):
        """Log inter-agent communication"""
        comm_data = {
            'sender': sender_id,
            'receiver': receiver_id,
            'message_type': message_type,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        self.communication_logs.append(comm_data)
        
    def update_metrics(self, metric_name, value, agent_id=None):
        """Update performance metrics"""
        if agent_id is None:
            self.metrics[metric_name].append(value)
        else:
            self.metrics['reward_evolution'][agent_id].append(value)
            
    def save_logs(self, filename_prefix):
        """Save all logs to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save step logs
        with open(f"{self.log_dir}/{filename_prefix}_steps_{timestamp}.json", 'w') as f:
            json.dump(self.step_logs, f, indent=2)
            
        # Save episode logs
        with open(f"{self.log_dir}/{filename_prefix}_episodes_{timestamp}.json", 'w') as f:
            json.dump(self.episode_logs, f, indent=2)
            
        # Save agent logs
        with open(f"{self.log_dir}/{filename_prefix}_agents_{timestamp}.json", 'w') as f:
            json.dump(dict(self.agent_logs), f, indent=2)
            
        # Save communication logs
        with open(f"{self.log_dir}/{filename_prefix}_communication_{timestamp}.json", 'w') as f:
            json.dump(self.communication_logs, f, indent=2)
            
        # Save metrics
        with open(f"{self.log_dir}/{filename_prefix}_metrics_{timestamp}.pickle", 'wb') as f:
            pickle.dump(self.metrics, f)
            
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        report = {
            'summary': {
                'total_episodes': len(self.episode_logs),
                'total_steps': len(self.step_logs),
                'agents_count': len(self.agent_logs),
                'communication_events': len(self.communication_logs)
            },
            'performance_trends': {},
            'agent_analysis': {}
        }
        
        # Analyze performance trends
        for metric, values in self.metrics.items():
            if values and metric != 'reward_evolution':
                report['performance_trends'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'trend': 'improving' if values[-1] < values[0] else 'degrading'
                }
                
        # Analyze agent performance
        for agent_id, rewards in self.metrics['reward_evolution'].items():
            if rewards:
                report['agent_analysis'][agent_id] = {
                    'total_actions': len(self.agent_logs[agent_id]),
                    'average_reward': np.mean(rewards),
                    'reward_trend': np.polyfit(range(len(rewards)), rewards, 1)[0],
                    'learning_stability': np.std(rewards[-100:]) if len(rewards) > 100 else np.std(rewards)
                }
                
        return report

class AdvancedRewardFunction:
    """
    Sophisticated reward function that considers multiple traffic objectives
    """
    def __init__(self, weights=None):
        # Default weights for different reward components
        self.weights = weights or {
            'waiting_time': -1.0,      # Minimize waiting time
            'queue_length': -0.5,      # Minimize queue length
            'throughput': 2.0,         # Maximize throughput
            'coordination': 1.5,       # Reward coordination with neighbors
            'fairness': 1.0,          # Promote fairness across lanes
            'fuel_efficiency': 0.8,    # Minimize fuel consumption
            'safety': 2.0,            # Safety bonus
            'green_efficiency': 0.3    # Efficient use of green time
        }
        
    def calculate_reward(self, junction_id, current_state, action, prev_state, 
                        neighbor_actions, global_info):
        """
        Calculate comprehensive reward considering multiple objectives
        """
        reward_components = {}
        
        # 1. Waiting time component
        current_waiting = sum(current_state['waiting_times'])
        prev_waiting = sum(prev_state['waiting_times']) if prev_state else current_waiting
        waiting_improvement = prev_waiting - current_waiting
        reward_components['waiting_time'] = self.weights['waiting_time'] * current_waiting
        
        # 2. Queue length component
        queue_lengths = current_state['queue_lengths']
        max_queue = max(queue_lengths)
        avg_queue = np.mean(queue_lengths)
        reward_components['queue_length'] = self.weights['queue_length'] * avg_queue
        
        # 3. Throughput component (vehicles passed through)
        throughput = current_state.get('vehicles_passed', 0)
        reward_components['throughput'] = self.weights['throughput'] * throughput
        
        # 4. Coordination component (alignment with neighbors)
        coordination_bonus = self._calculate_coordination_bonus(action, neighbor_actions)
        reward_components['coordination'] = self.weights['coordination'] * coordination_bonus
        
        # 5. Fairness component (balanced service across lanes)
        fairness_bonus = self._calculate_fairness_bonus(queue_lengths, action)
        reward_components['fairness'] = self.weights['fairness'] * fairness_bonus
        
        # 6. Fuel efficiency component
        fuel_consumption = current_state.get('fuel_consumption', 0)
        reward_components['fuel_efficiency'] = self.weights['fuel_efficiency'] * (-fuel_consumption)
        
        # 7. Safety component
        safety_bonus = self._calculate_safety_bonus(current_state)
        reward_components['safety'] = self.weights['safety'] * safety_bonus
        
        # 8. Green time efficiency
        green_efficiency = self._calculate_green_efficiency(current_state, action)
        reward_components['green_efficiency'] = self.weights['green_efficiency'] * green_efficiency
        
        # Calculate total reward
        total_reward = sum(reward_components.values())
        
        return total_reward, reward_components
        
    def _calculate_coordination_bonus(self, action, neighbor_actions):
        """Calculate bonus for coordinated actions with neighboring intersections"""
        if not neighbor_actions:
            return 0
            
        # Reward synchronized green waves
        coordination_score = 0
        for neighbor_action in neighbor_actions.values():
            if action == neighbor_action:
                coordination_score += 0.5
            elif abs(action - neighbor_action) <= 1:  # Adjacent phases
                coordination_score += 0.2
                
        return coordination_score / max(len(neighbor_actions), 1)
        
    def _calculate_fairness_bonus(self, queue_lengths, action):
        """Calculate fairness bonus to prevent starvation of certain lanes"""
        if len(queue_lengths) == 0:
            return 0
            
        # Calculate standard deviation of queue lengths
        queue_std = np.std(queue_lengths)
        max_queue_idx = np.argmax(queue_lengths)
        
        # Bonus if serving the lane with longest queue
        fairness_bonus = 0
        if action == max_queue_idx:
            fairness_bonus += 1.0
            
        # Penalty for high variance in queue lengths
        fairness_bonus -= queue_std * 0.1
        
        return fairness_bonus
        
    def _calculate_safety_bonus(self, current_state):
        """Calculate safety bonus based on traffic conditions"""
        safety_score = 0
        
        # Bonus for preventing extremely long queues (accident risk)
        max_queue = max(current_state['queue_lengths']) if current_state['queue_lengths'] else 0
        if max_queue < 10:
            safety_score += 1.0
        elif max_queue > 20:
            safety_score -= 1.0
            
        # Bonus for smooth traffic flow
        speed_variance = current_state.get('speed_variance', 0)
        if speed_variance < 5:  # Low speed variance indicates smooth flow
            safety_score += 0.5
            
        return safety_score
        
    def _calculate_green_efficiency(self, current_state, action):
        """Calculate efficiency of green time usage"""
        if 'green_time_utilization' not in current_state:
            return 0
            
        utilization = current_state['green_time_utilization']
        
        # Reward high utilization but penalize waste
        if utilization > 0.8:
            return 1.0
        elif utilization < 0.3:
            return -0.5
        else:
            return utilization - 0.5

class CommunicationProtocol:
    """
    Inter-agent communication system for coordination
    """
    def __init__(self, agent_ids):
        self.agent_ids = agent_ids
        self.message_queue = defaultdict(list)
        self.neighbor_graph = self._build_neighbor_graph()
        
    def _build_neighbor_graph(self):
        """Build graph of neighboring intersections"""
        # This should be built based on actual road network topology
        # For now, we'll use a simple proximity-based approach
        neighbor_graph = defaultdict(list)
        
        # Example: assuming agents are arranged in a grid
        for i, agent_id in enumerate(self.agent_ids):
            # Add adjacent agents as neighbors
            if i > 0:
                neighbor_graph[agent_id].append(self.agent_ids[i-1])
            if i < len(self.agent_ids) - 1:
                neighbor_graph[agent_id].append(self.agent_ids[i+1])
                
        return neighbor_graph
        
    def send_message(self, sender_id, message_type, content, recipient_id=None):
        """Send message to specific agent or broadcast to neighbors"""
        message = {
            'sender': sender_id,
            'type': message_type,
            'content': content,
            'timestamp': time.time()
        }
        
        if recipient_id:
            self.message_queue[recipient_id].append(message)
        else:
            # Broadcast to all neighbors
            for neighbor in self.neighbor_graph[sender_id]:
                self.message_queue[neighbor].append(message)
                
    def receive_messages(self, agent_id):
        """Retrieve and clear messages for an agent"""
        messages = self.message_queue[agent_id]
        self.message_queue[agent_id] = []
        return messages
        
    def get_neighbor_actions(self, agent_id, recent_actions):
        """Get recent actions of neighboring agents"""
        neighbor_actions = {}
        for neighbor in self.neighbor_graph[agent_id]:
            if neighbor in recent_actions:
                neighbor_actions[neighbor] = recent_actions[neighbor]
        return neighbor_actions

class EnhancedModel(nn.Module):
    """
    Enhanced neural network with attention mechanism and communication processing
    """
    def __init__(self, lr, input_dims, hidden_dims, n_actions, communication_dims=10):
        super(EnhancedModel, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.communication_dims = communication_dims
        
        # Main state processing network
        self.state_net = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Communication processing network
        self.comm_net = nn.Sequential(
            nn.Linear(communication_dims, hidden_dims // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims // 2, hidden_dims // 2),
            nn.ReLU()
        )
        
        # Attention mechanism for combining state and communication
        self.attention = nn.MultiheadAttention(hidden_dims, num_heads=4, batch_first=True)
        
        # Final action network
        self.action_net = nn.Sequential(
            nn.Linear(hidden_dims + hidden_dims // 2, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, n_actions)
        )
        
        # Value network for advantage calculation
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dims + hidden_dims // 2, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, state, communication_input=None):
        # Process state
        state_features = self.state_net(state)
        
        # Process communication if available
        if communication_input is not None and communication_input.numel() > 0:
            comm_features = self.comm_net(communication_input)
            
            # Apply attention mechanism
            state_features_expanded = state_features.unsqueeze(1)
            attended_features, _ = self.attention(state_features_expanded, 
                                                state_features_expanded, 
                                                state_features_expanded)
            state_features = attended_features.squeeze(1)
            
            # Combine features
            combined_features = torch.cat([state_features, comm_features], dim=-1)
        else:
            # Use zero padding for communication features
            comm_features = torch.zeros(state.shape[0], self.hidden_dims // 2, 
                                      device=self.device)
            combined_features = torch.cat([state_features, comm_features], dim=-1)
        
        # Generate action probabilities and state value
        actions = self.action_net(combined_features)
        value = self.value_net(combined_features)
        
        return actions, value

class MultiAgent:
    """
    Enhanced multi-agent system with communication and coordination
    """
    def __init__(self, agent_configs, reward_function, communication_protocol, logger):
        self.agents = {}
        self.agent_ids = list(agent_configs.keys())
        self.reward_function = reward_function
        self.communication = communication_protocol
        self.logger = logger
        
        # Initialize individual agents
        for agent_id, config in agent_configs.items():
            self.agents[agent_id] = self._create_agent(agent_id, config)
            
        # Shared memory for coordination
        self.recent_actions = {}
        self.global_state_history = deque(maxlen=100)
        
    def _create_agent(self, agent_id, config):
        """Create individual agent with enhanced capabilities"""
        agent = {
            'id': agent_id,
            'model': EnhancedModel(
                lr=config['lr'],
                input_dims=config['input_dims'],
                hidden_dims=config['hidden_dims'],
                n_actions=config['n_actions'],
                communication_dims=config.get('communication_dims', 10)
            ),
            'memory': deque(maxlen=config.get('memory_size', 10000)),
            'epsilon': config.get('epsilon', 0.1),
            'epsilon_decay': config.get('epsilon_decay', 0.995),
            'epsilon_min': config.get('epsilon_min', 0.01),
            'gamma': config.get('gamma', 0.99),
            'target_update_freq': config.get('target_update_freq', 100),
            'update_counter': 0,
            'performance_history': deque(maxlen=1000)
        }
        
        # Create target network
        agent['target_model'] = EnhancedModel(
            lr=config['lr'],
            input_dims=config['input_dims'],
            hidden_dims=config['hidden_dims'],
            n_actions=config['n_actions'],
            communication_dims=config.get('communication_dims', 10)
        )
        agent['target_model'].load_state_dict(agent['model'].state_dict())
        
        return agent
        
    def select_actions(self, states, step):
        """Select actions for all agents considering communication"""
        actions = {}
        
        for agent_id in self.agent_ids:
            agent = self.agents[agent_id]
            state = states[agent_id]
            
            # Get communication input from recent messages
            messages = self.communication.receive_messages(agent_id)
            communication_input = self._process_messages(messages)
            
            # Select action using epsilon-greedy with neural network
            if np.random.random() < agent['epsilon']:
                action = np.random.randint(agent['model'].n_actions)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent['model'].device)
                comm_tensor = torch.FloatTensor(communication_input).unsqueeze(0).to(agent['model'].device) if communication_input.size > 0 else None
                
                with torch.no_grad():
                    q_values, _ = agent['model'](state_tensor, comm_tensor)
                    action = q_values.argmax().item()
            
            actions[agent_id] = action
            
            # Send communication to neighbors about intended action
            self.communication.send_message(
                agent_id, 
                'action_intent', 
                {'action': action, 'state_summary': state[:4].tolist()}
            )
            
        # Update recent actions history
        self.recent_actions.update(actions)
        
        return actions
        
    def _process_messages(self, messages):
        """Process received messages into communication input vector"""
        if not messages:
            return np.array([])
            
        # Extract relevant information from messages
        comm_vector = []
        
        for message in messages[-3:]:  # Consider last 3 messages
            if message['type'] == 'action_intent':
                comm_vector.extend([
                    message['content']['action'],
                    np.mean(message['content']['state_summary'])
                ])
                
        # Pad or truncate to fixed size
        comm_vector = comm_vector[:10]  # Max 10 dimensions
        comm_vector.extend([0] * (10 - len(comm_vector)))  # Pad with zeros
        
        return np.array(comm_vector)
        
    def update_agents(self, experiences, step):
        """Update all agents using their experiences"""
        for agent_id, experience in experiences.items():
            if len(experience) == 0:
                continue
                
            agent = self.agents[agent_id]
            
            # Store experience in agent's memory
            agent['memory'].extend(experience)
            
            # Train agent if enough experience available
            if len(agent['memory']) >= 64:  # Minimum batch size
                self._train_agent(agent, step)
                
            # Update epsilon
            agent['epsilon'] = max(
                agent['epsilon_min'], 
                agent['epsilon'] * agent['epsilon_decay']
            )
            
            # Update target network periodically
            agent['update_counter'] += 1
            if agent['update_counter'] % agent['target_update_freq'] == 0:
                agent['target_model'].load_state_dict(agent['model'].state_dict())
                
    def _train_agent(self, agent, step):
        """Train individual agent using Double DQN with communication"""
        if len(agent['memory']) < 64:
            return
            
        # Sample batch from memory
        batch = random.sample(agent['memory'], min(64, len(agent['memory'])))
        
        states = torch.FloatTensor([e['state'] for e in batch]).to(agent['model'].device)
        actions = torch.LongTensor([e['action'] for e in batch]).to(agent['model'].device)
        rewards = torch.FloatTensor([e['reward'] for e in batch]).to(agent['model'].device)
        next_states = torch.FloatTensor([e['next_state'] for e in batch]).to(agent['model'].device)
        dones = torch.BoolTensor([e['done'] for e in batch]).to(agent['model'].device)
        
        # Process communication inputs
        comm_inputs = []
        next_comm_inputs = []
        for e in batch:
            comm_input = e.get('communication_input', np.array([]))
            next_comm_input = e.get('next_communication_input', np.array([]))
            
            if comm_input.size == 0:
                comm_input = np.zeros(10)
            if next_comm_input.size == 0:
                next_comm_input = np.zeros(10)
                
            comm_inputs.append(comm_input)
            next_comm_inputs.append(next_comm_input)
            
        comm_tensor = torch.FloatTensor(comm_inputs).to(agent['model'].device)
        next_comm_tensor = torch.FloatTensor(next_comm_inputs).to(agent['model'].device)
        
        # Current Q values
        current_q_values, current_values = agent['model'](states, comm_tensor)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q values using Double DQN
        with torch.no_grad():
            next_q_values, next_values = agent['model'](next_states, next_comm_tensor)
            next_actions = next_q_values.argmax(1)
            
            target_next_q_values, _ = agent['target_model'](next_states, next_comm_tensor)
            target_q_values = target_next_q_values.gather(1, next_actions.unsqueeze(1))
            
            target_q_values = rewards.unsqueeze(1) + (agent['gamma'] * target_q_values * ~dones.unsqueeze(1))
            
        # Calculate loss
        q_loss = F.mse_loss(current_q_values, target_q_values)
        value_loss = F.mse_loss(current_values, target_q_values)
        total_loss = q_loss + 0.5 * value_loss
        
        # Update network
        agent['model'].optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent['model'].parameters(), 1.0)
        agent['model'].optimizer.step()
        
        # Log training metrics
        self.logger.update_metrics('training_loss', total_loss.item(), agent['id'])
        
    def get_agent_states(self, junction_states):
        """Get states for all agents"""
        agent_states = {}
        for agent_id in self.agent_ids:
            # Convert junction state to agent state format
            junction_id = f"gneJ{agent_id}" if not agent_id.startswith("gneJ") else agent_id
            if junction_id in junction_states:
                agent_states[agent_id] = junction_states[junction_id]
            else:
                # Default state if junction not found
                agent_states[agent_id] = np.zeros(12)  # Assuming 12-dimensional state
                
        return agent_states
        
    def save_models(self, model_dir):
        """Save all agent models"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        for agent_id, agent in self.agents.items():
            model_path = os.path.join(model_dir, f"agent_{agent_id}_model.pth")
            torch.save(agent['model'].state_dict(), model_path)
            
            target_path = os.path.join(model_dir, f"agent_{agent_id}_target.pth")
            torch.save(agent['target_model'].state_dict(), target_path)
            
    def load_models(self, model_dir):
        """Load all agent models"""
        for agent_id, agent in self.agents.items():
            model_path = os.path.join(model_dir, f"agent_{agent_id}_model.pth")
            target_path = os.path.join(model_dir, f"agent_{agent_id}_target.pth")
            
            if os.path.exists(model_path):
                agent['model'].load_state_dict(torch.load(model_path, map_location=agent['model'].device))
            if os.path.exists(target_path):
                agent['target_model'].load_state_dict(torch.load(target_path, map_location=agent['model'].device))

def get_enhanced_traffic_metrics(all_junctions):
    """
    Collect comprehensive traffic metrics for reward calculation and logging
    """
    metrics = {}
    
    for junction in all_junctions:
        controlled_lanes = traci.trafficlight.getControlledLanes(junction)
        
        # Basic metrics
        waiting_times = [traci.lane.getWaitingTime(lane) for lane in controlled_lanes]
        queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes]
        vehicle_counts = [traci.lane.getLastStepVehicleNumber(lane) for lane in controlled_lanes]
        
        # Advanced metrics
        speeds = []
        fuel_consumption = 0
        co2_emissions = 0
        
        for lane in controlled_lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            lane_speeds = []
            
            for veh_id in vehicle_ids:
                try:
                    speed = traci.vehicle.getSpeed(veh_id)
                    lane_speeds.append(speed)
                    
                    # Get fuel consumption and emissions if available
                    fuel_consumption += traci.vehicle.getFuelConsumption(veh_id)
                    co2_emissions += traci.vehicle.getCO2Emission(veh_id)
                except:
                    continue  # Vehicle might have left the network
                    
            speeds.extend(lane_speeds)
            
        # Calculate derived metrics
        avg_speed = np.mean(speeds) if speeds else 0
        speed_variance = np.var(speeds) if len(speeds) > 1 else 0
        total_vehicles = sum(vehicle_counts)
        
        # Green time utilization (approximation)
        current_phase = traci.trafficlight.getPhase(junction)
        phase_definition = traci.trafficlight.getAllProgramLogics(junction)[0].phases[current_phase]
        green_lanes = [i for i, state in enumerate(phase_definition.state) if state.lower() == 'g']
        
        green_utilization = 0
        if green_lanes:
            green_lane_vehicles = sum(vehicle_counts[i] for i in green_lanes if i < len(vehicle_counts))
            green_utilization = min(green_lane_vehicles / max(len(green_lanes), 1), 1.0)
            
        metrics[junction] = {
            'waiting_times': waiting_times,
            'queue_lengths': queue_lengths,
            'vehicle_counts': vehicle_counts,
            'avg_speed': avg_speed,
            'speed_variance': speed_variance,
            'total_vehicles': total_vehicles,
            'fuel_consumption': fuel_consumption,
            'co2_emissions': co2_emissions,
            'green_time_utilization': green_utilization,
            'vehicles_passed': 0  # Would need to track over time
        }
        
    return metrics

def run_enhanced_simulation(train=True, model_name="enhanced_model", epochs=50, steps=500, ard=False):
    """
    Run enhanced multi-agent traffic light simulation
    """
    # Initialize logging system
    logger = PerformanceLogger(log_dir=f"logs/{model_name}")
    
    # Arduino setup if needed
    arduino = None
    if ard:
        try:
            arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)
            def write_read(x):
                arduino.write(bytes(x, 'utf-8'))
                time.sleep(0.05)
                data = arduino.readline()
                return data
        except Exception as e:
            print(f"Arduino connection failed: {e}")
            ard = False
    
    # Initialize SUMO
    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg", 
                "--tripinfo-output", "tripinfo.xml"])
    
    all_junctions = traci.trafficlight.getIDList()
    print(f"Found {len(all_junctions)} traffic light junctions: {all_junctions}")
    
    # Initialize enhanced reward function
    reward_function = AdvancedRewardFunction()
    
    # Initialize communication protocol
    communication = CommunicationProtocol(all_junctions)
    
    # Configure agents
    agent_configs = {}
    for junction_id in all_junctions:
        agent_configs[junction_id] = {
            'lr': 0.001,
            'input_dims': 12,  # Enhanced state representation
            'hidden_dims': 256,
            'n_actions': 4,
            'communication_dims': 10,
            'memory_size': 10000,
            'epsilon': 0.1 if train else 0.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'gamma': 0.99,
            'target_update_freq': 100
        }
    
    # Initialize multi-agent system
    multi_agent = MultiAgent(agent_configs, reward_function, communication, logger)
    
    # Load models if not training
    if not train:
        try:
            multi_agent.load_models(f"models/{model_name}")
            print("Models loaded successfully")
        except Exception as e:
            print(f"Could not load models: {e}")
    
    traci.close()
    
    # Training/Testing loop
    best_performance = float('inf')
    performance_history = []
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        
        # Start SUMO for this epoch
        if train:
            traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg", 
                        "--tripinfo-output", f"tripinfo_epoch_{epoch}.xml"])
        else:
            traci.start([checkBinary("sumo-gui"), "-c", "configuration.sumocfg", 
                        "--tripinfo-output", f"tripinfo_test.xml"])
        
        # Initialize epoch metrics
        epoch_metrics = {
            'total_waiting_time': 0,
            'total_throughput': 0,
            'total_fuel_consumption': 0,
            'total_emissions': 0,
            'coordination_events': 0,
            'agent_rewards': {agent_id: [] for agent_id in all_junctions}
        }
        
        # Phase definitions for each action
        phase_definitions = [
            ["yyyyrrrrrrrr", "GGGGrrrrrrrr"],  # North-South
            ["rrrryyyyrrrr", "rrrrGGGGrrrr"],  # East-West  
            ["rrrrrrrryyyy", "rrrrrrrrGGGG"],  # Left turns
            ["yyyyyyyyrrrr", "GGGGGGGGrrrr"]   # All directions
        ]
        
        # Initialize junction states
        junction_timers = {junction: 0 for junction in all_junctions}
        previous_states = {junction: None for junction in all_junctions}
        current_experiences = {junction: [] for junction in all_junctions}
        
        # Simulation loop
        step = 0
        while step < steps:
            traci.simulationStep()
            
            # Collect current traffic metrics
            current_metrics = get_enhanced_traffic_metrics(all_junctions)
            
            # Process each junction
            for junction_id in all_junctions:
                if junction_timers[junction_id] <= 0:
                    # Time to make a decision
                    
                    # Prepare enhanced state representation
                    junction_metrics = current_metrics[junction_id]
                    current_state = np.array([
                        *junction_metrics['waiting_times'][:4],      # Waiting times per lane
                        *junction_metrics['queue_lengths'][:4],      # Queue lengths per lane  
                        junction_metrics['avg_speed'],               # Average speed
                        junction_metrics['speed_variance'],          # Speed variance
                        junction_metrics['green_time_utilization'],  # Green time efficiency
                        junction_metrics['total_vehicles']           # Total vehicles
                    ])
                    
                    # Ensure state has correct dimensions
                    if len(current_state) < 12:
                        current_state = np.pad(current_state, (0, 12 - len(current_state)))
                    current_state = current_state[:12]
                    
                    # Get neighbor actions for coordination
                    neighbor_actions = communication.get_neighbor_actions(
                        junction_id, multi_agent.recent_actions
                    )
                    
                    # Calculate reward if we have a previous state
                    if previous_states[junction_id] is not None:
                        reward, reward_components = reward_function.calculate_reward(
                            junction_id, 
                            junction_metrics,
                            multi_agent.recent_actions.get(junction_id, 0),
                            previous_states[junction_id],
                            neighbor_actions,
                            current_metrics
                        )
                        
                        # Log detailed reward information
                        logger.log_agent_action(
                            junction_id,
                            previous_states[junction_id]['state'],
                            multi_agent.recent_actions.get(junction_id, 0),
                            reward,
                            current_state,
                            {
                                'reward_components': reward_components,
                                'junction_metrics': junction_metrics,
                                'neighbor_actions': neighbor_actions
                            }
                        )
                        
                        # Store experience for training
                        if train and len(current_experiences[junction_id]) > 0:
                            last_experience = current_experiences[junction_id][-1]
                            last_experience['reward'] = reward
                            last_experience['next_state'] = current_state
                            last_experience['done'] = (step >= steps - 1)
                            
                        epoch_metrics['agent_rewards'][junction_id].append(reward)
                        logger.update_metrics('reward', reward, junction_id)
                    
                    # Store current state for next iteration
                    previous_states[junction_id] = {
                        'state': current_state.copy(),
                        'metrics': junction_metrics.copy()
                    }
                    
                    junction_timers[junction_id] = 0  # Will be set after action selection
            
            # Select actions for all agents
            agent_states = {}
            for junction_id in all_junctions:
                if previous_states[junction_id] is not None:
                    agent_states[junction_id] = previous_states[junction_id]['state']
                else:
                    agent_states[junction_id] = np.zeros(12)
            
            if agent_states:
                actions = multi_agent.select_actions(agent_states, step)
                
                # Execute actions and set timers
                for junction_id, action in actions.items():
                    if junction_timers[junction_id] <= 0:
                        # Apply yellow phase first
                        yellow_state = phase_definitions[action][0]
                        traci.trafficlight.setRedYellowGreenState(junction_id, yellow_state)
                        traci.trafficlight.setPhaseDuration(junction_id, 3)
                        
                        # Then apply green phase
                        green_state = phase_definitions[action][1]
                        # Schedule green phase (will be applied in next steps)
                        
                        # Set timer for this junction
                        junction_timers[junction_id] = 20  # Total phase duration
                        
                        # Store experience for training
                        if train and previous_states[junction_id] is not None:
                            experience = {
                                'state': previous_states[junction_id]['state'],
                                'action': action,
                                'reward': 0,  # Will be filled in next iteration
                                'next_state': None,  # Will be filled in next iteration
                                'done': False,
                                'communication_input': communication.get_neighbor_actions(
                                    junction_id, multi_agent.recent_actions
                                )
                            }
                            current_experiences[junction_id].append(experience)
                        
                        # Arduino communication
                        if ard and junction_id == all_junctions[0]:  # Only for first junction
                            try:
                                ph = str(action)
                                value = write_read(ph)
                            except Exception as e:
                                print(f"Arduino communication error: {e}")
                
                # Log communication events
                for sender, neighbors in communication.neighbor_graph.items():
                    for neighbor in neighbors:
                        if sender in actions and neighbor in actions:
                            logger.log_communication(
                                sender, neighbor, 'action_coordination',
                                {'sender_action': actions[sender], 'receiver_action': actions[neighbor]}
                            )
            
            # Update junction timers
            for junction_id in all_junctions:
                if junction_timers[junction_id] > 0:
                    junction_timers[junction_id] -= 1
            
            # Update epoch metrics
            total_waiting = sum(sum(m['waiting_times']) for m in current_metrics.values())
            total_vehicles = sum(m['total_vehicles'] for m in current_metrics.values())
            total_fuel = sum(m['fuel_consumption'] for m in current_metrics.values())
            total_co2 = sum(m['co2_emissions'] for m in current_metrics.values())
            
            epoch_metrics['total_waiting_time'] += total_waiting
            epoch_metrics['total_throughput'] += total_vehicles
            epoch_metrics['total_fuel_consumption'] += total_fuel
            epoch_metrics['total_emissions'] += total_co2
            
            # Log step data
            step_data = {
                'junction_states': {jid: previous_states[jid]['state'].tolist() 
                                  if previous_states[jid] else [] for jid in all_junctions},
                'actions': actions if 'actions' in locals() else {},
                'rewards': {jid: epoch_metrics['agent_rewards'][jid][-1] 
                           if epoch_metrics['agent_rewards'][jid] else 0 for jid in all_junctions}
            }
            
            global_step_metrics = {
                'total_waiting_time': total_waiting,
                'total_vehicles': total_vehicles,
                'total_fuel_consumption': total_fuel,
                'total_emissions': total_co2,
                'average_speed': np.mean([m['avg_speed'] for m in current_metrics.values()])
            }
            
            logger.log_step(step, step_data, global_step_metrics)
            
            step += 1
        
        # End of epoch processing
        traci.close()
        
        # Calculate epoch performance
        epoch_performance = epoch_metrics['total_waiting_time']
        performance_history.append(epoch_performance)
        
        print(f"Epoch {epoch + 1} Performance:")
        print(f"  Total Waiting Time: {epoch_performance:.2f}")
        print(f"  Total Throughput: {epoch_metrics['total_throughput']}")
        print(f"  Total Fuel Consumption: {epoch_metrics['total_fuel_consumption']:.2f}")
        print(f"  Average Agent Rewards: {np.mean([np.mean(rewards) if rewards else 0 for rewards in epoch_metrics['agent_rewards'].values()]):.2f}")
        
        # Update agents with collected experiences
        if train:
            multi_agent.update_agents(current_experiences, epoch)
            
            # Save best model
            if epoch_performance < best_performance:
                best_performance = epoch_performance
                multi_agent.save_models(f"models/{model_name}")
                print(f"  New best performance! Model saved.")
        
        # Log epoch data
        logger.log_episode(epoch, epoch_metrics)
        
        # Update global metrics
        logger.update_metrics('total_waiting_time', epoch_performance)
        logger.update_metrics('throughput', epoch_metrics['total_throughput'])
        logger.update_metrics('fuel_consumption', epoch_metrics['total_fuel_consumption'])
        logger.update_metrics('emissions', epoch_metrics['total_emissions'])
        
        if not train:
            break  # Only run one epoch for testing
    
    # Generate and save performance report
    performance_report = logger.generate_performance_report()
    
    print("\n=== Final Performance Report ===")
    print(f"Total Episodes: {performance_report['summary']['total_episodes']}")
    print(f"Total Communication Events: {performance_report['summary']['communication_events']}")
    print(f"Best Performance: {best_performance:.2f}")
    
    # Print agent analysis
    print("\nAgent Performance Analysis:")
    for agent_id, analysis in performance_report['agent_analysis'].items():
        print(f"  {agent_id}:")
        print(f"    Average Reward: {analysis['average_reward']:.3f}")
        print(f"    Learning Stability: {analysis['learning_stability']:.3f}")
        print(f"    Total Actions: {analysis['total_actions']}")
    
    # Save all logs and generate plots
    logger.save_logs(model_name)
    
    if train:
        # Plot performance evolution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(range(len(performance_history)), performance_history)
        plt.title('Total Waiting Time per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Waiting Time')
        
        plt.subplot(2, 3, 2)
        throughput_history = logger.metrics['throughput']
        plt.plot(range(len(throughput_history)), throughput_history)
        plt.title('Throughput per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Total Vehicles')
        
        plt.subplot(2, 3, 3)
        fuel_history = logger.metrics['fuel_consumption']
        plt.plot(range(len(fuel_history)), fuel_history)
        plt.title('Fuel Consumption per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Fuel Consumption')
        
        plt.subplot(2, 3, 4)
        # Plot reward evolution for each agent
        for agent_id, rewards in logger.metrics['reward_evolution'].items():
            if rewards:
                plt.plot(range(len(rewards)), rewards, label=f'Agent {agent_id}', alpha=0.7)
        plt.title('Reward Evolution per Agent')
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        plt.legend()
        
        plt.subplot(2, 3, 5)
        # Plot coordination efficiency (communication events over time)
        comm_events_per_epoch = []
        current_count = 0
        epoch_size = len(logger.step_logs) // epochs if epochs > 0 else 1
        
        for i in range(0, len(logger.communication_logs), epoch_size):
            epoch_comms = len(logger.communication_logs[i:i+epoch_size])
            comm_events_per_epoch.append(epoch_comms)
            
        plt.plot(range(len(comm_events_per_epoch)), comm_events_per_epoch)
        plt.title('Communication Events per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Communication Events')
        
        plt.subplot(2, 3, 6)
        # Plot learning curves (epsilon decay for first agent)
        if logger.metrics['reward_evolution']:
            first_agent = list(logger.metrics['reward_evolution'].keys())[0]
            rewards = logger.metrics['reward_evolution'][first_agent]
            if len(rewards) > 100:
                # Moving average
                window_size = 50
                moving_avg = [np.mean(rewards[i:i+window_size]) 
                             for i in range(len(rewards)-window_size)]
                plt.plot(range(len(moving_avg)), moving_avg)
                plt.title(f'Learning Curve (Agent {first_agent})')
                plt.xlabel('Training Steps')
                plt.ylabel('Moving Average Reward')
        
        plt.tight_layout()
        plt.savefig(f'plots/enhanced_performance_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Clean up
    if arduino:
        arduino.close()
    
    print(f"\nLogs and performance data saved to logs/{model_name}/")
    print(f"Models saved to models/{model_name}/")
    
    return performance_report

def get_enhanced_options():
    """Enhanced command line options"""
    optParser = optparse.OptionParser()
    optParser.add_option(
        "-m", "--model",
        dest='model_name',
        type='string',
        default="enhanced_model",
        help="Name of the model for saving/loading"
    )
    optParser.add_option(
        "--train",
        action='store_true',
        default=False,
        help="Training mode (default: testing mode)"
    )
    optParser.add_option(
        "-e", "--epochs",
        dest='epochs',
        type='int',
        default=50,
        help="Number of training epochs"
    )
    optParser.add_option(
        "-s", "--steps",
        dest='steps',
        type='int',
        default=500,
        help="Number of simulation steps per epoch"
    )
    optParser.add_option(
        "--arduino",
        action='store_true',
        default=False,
        help="Enable Arduino communication"
    )
    optParser.add_option(
        "--log-level",
        dest='log_level',
        type='string',
        default='INFO',
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    optParser.add_option(
        "--reward-weights",
        dest='reward_weights',
        type='string',
        default=None,
        help="JSON string of custom reward weights"
    )
    
    options, args = optParser.parse_args()
    return options

# Main execution
if __name__ == "__main__":
    options = get_enhanced_options()
    
    # Parse custom reward weights if provided
    reward_weights = None
    if options.reward_weights:
        try:
            reward_weights = json.loads(options.reward_weights)
        except json.JSONDecodeError:
            print("Warning: Invalid reward weights JSON, using defaults")
    
    print("=== Enhanced Multi-Agent Traffic Light System ===")
    print(f"Mode: {'Training' if options.train else 'Testing'}")
    print(f"Model: {options.model_name}")
    print(f"Epochs: {options.epochs}")
    print(f"Steps per epoch: {options.steps}")
    print(f"Arduino: {'Enabled' if options.arduino else 'Disabled'}")
    
    # Run the enhanced simulation
    try:
        performance_report = run_enhanced_simulation(
            train=options.train,
            model_name=options.model_name,
            epochs=options.epochs,
            steps=options.steps,
            ard=options.arduino
        )
        
        print("\n=== Simulation completed successfully ===")
        
    except KeyboardInterrupt:
        print("\n=== Simulation interrupted by user ===")
    except Exception as e:
        print(f"\n=== Simulation failed with error: {e} ===")
        import traceback
        traceback.print_exc()