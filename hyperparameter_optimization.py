#!/usr/bin/env python3
"""
Hyperparameter Optimization for Enhanced Multi-Agent Traffic Light System
"""

import optuna
import json
import os
import sys
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Import our enhanced system
from enhanced_config import SystemConfig, HyperparameterOptimizer
from train import run_enhanced_simulation

class OptimizationStudy:
    """Main class for conducting hyperparameter optimization studies"""
    
    def __init__(self, study_name, storage_url=None):
        self.study_name = study_name
        self.storage_url = storage_url or f"sqlite:///{study_name}.db"
        
        # Create optuna study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=self.storage_url,
            direction='minimize',  # Minimize waiting time
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Initialize optimizer
        base_config = {
            'DEFAULT_REWARD_WEIGHTS': SystemConfig.DEFAULT_REWARD_WEIGHTS,
            'NETWORK_CONFIG': SystemConfig.NETWORK_CONFIG,
            'TRAINING_CONFIG': SystemConfig.TRAINING_CONFIG,
            'TRAFFIC_CONFIG': SystemConfig.TRAFFIC_CONFIG,
            'LOGGING_CONFIG': SystemConfig.LOGGING_CONFIG
        }
        
        self.optimizer = HyperparameterOptimizer(base_config)
        
    def objective(self, trial):
        """Objective function for optimization"""
        try:
            # Get suggested hyperparameters
            config = self.optimizer.suggest_hyperparameters(trial)
            
            # Create temporary config file
            temp_config_file = f"temp_config_trial_{trial.number}.json"
            with open(temp_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Update system config
            SystemConfig.load_from_file(temp_config_file)
            
            # Run simulation with suggested hyperparameters
            print(f"\nRunning trial {trial.number}...")
            print(f"Hidden dims: {config['NETWORK_CONFIG']['hidden_dims']}")
            print(f"Learning rate: {config['NETWORK_CONFIG']['learning_rate']:.6f}")
            print(f"Gamma: {config['TRAINING_CONFIG']['gamma']:.3f}")
            
            performance_report = run_enhanced_simulation(
                train=True,
                model_name=f"optim_trial_{trial.number}",
                epochs=30,  # Reduced for faster optimization
                steps=300   # Reduced for faster optimization
            )
            
            # Extract performance metrics
            waiting_time = performance_report['performance_trends']['total_waiting_time']['mean']
            throughput = performance_report['performance_trends']['throughput']['mean']
            fuel_consumption = performance_report['performance_trends']['fuel_consumption']['mean']
            
            # Multi-objective optimization (weighted sum)
            objective_value = (
                waiting_time * 1.0 +           # Primary: minimize waiting time
                (-throughput) * 0.3 +          # Secondary: maximize throughput  
                fuel_consumption * 0.2         # Tertiary: minimize fuel consumption
            )
            
            # Log additional metrics
            trial.set_user_attr('waiting_time', waiting_time)
            trial.set_user_attr('throughput', throughput)
            trial.set_user_attr('fuel_consumption', fuel_consumption)
            
            # Clean up temporary config file
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
            
            print(f"Trial {trial.number} completed - Objective: {objective_value:.2f}")
            
            return objective_value
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            # Clean up on failure
            temp_config_file = f"temp_config_trial_{trial.number}.json"
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
            return float('inf')
    
    def run_optimization(self, n_trials, timeout=None):
        """Run optimization study"""
        print(f"Starting optimization study: {self.study_name}")
        print(f"Number of trials: {n_trials}")
        print(f"Timeout: {timeout} seconds" if timeout else "No timeout")
        
        # Add callbacks for progress tracking
        def callback(study, trial):
            if trial.number % 5 == 0:
                print(f"\nProgress: {trial.number}/{n_trials} trials completed")
                print(f"Best value so far: {study.best_value:.2f}")
                print(f"Best params: {study.best_params}")
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[callback]
        )
        
        # Print results
        self.print_results()
        
        return self.study
    
    def print_results(self):
        """Print optimization results"""
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS")
        print("="*60)
        
        trial = self.study.best_trial
        
        print(f"Best objective value: {trial.value:.2f}")
        print(f"Best parameters:")
        for key, value in trial.params.items():
            print(f"  {key}: {value}")
        
        print(f"\nPerformance metrics:")
        print(f"  Waiting time: {trial.user_attrs.get('waiting_time', 'N/A')}")
        print(f"  Throughput: {trial.user_attrs.get('throughput', 'N/A')}")
        print(f"  Fuel consumption: {trial.user_attrs.get('fuel_consumption', 'N/A')}")
        
        print(f"\nTotal trials completed: {len(self.study.trials)}")
        print(f"Failed trials: {len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    def save_best_config(self, output_file):
        """Save best configuration to file"""
        best_config = self.optimizer.suggest_hyperparameters(self.study.best_trial)
        
        with open(output_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"Best configuration saved to: {output_file}")
    
    def plot_optimization_history(self, save_path=None):
        """Plot optimization history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Objective value history
        objectives = [trial.value for trial in self.study.trials if trial.value is not None]
        axes[0, 0].plot(objectives)
        axes[0, 0].set_title('Objective Value History')
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('Objective Value')
        
        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            params = list(importance.keys())
            values = list(importance.values())
            
            axes[0, 1].barh(params, values)
            axes[0, 1].set_title('Parameter Importance')
            axes[0, 1].set_xlabel('Importance')
        except:
            axes[0, 1].text(0.5, 0.5, 'Parameter importance\nnot available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # Performance metrics scatter
        waiting_times = [trial.user_attrs.get('waiting_time') for trial in self.study.trials 
                        if trial.user_attrs.get('waiting_time') is not None]
        throughputs = [trial.user_attrs.get('throughput') for trial in self.study.trials 
                      if trial.user_attrs.get('throughput') is not None]
        
        if waiting_times and throughputs:
            axes[1, 0].scatter(waiting_times, throughputs, alpha=0.6)
            axes[1, 0].set_xlabel('Waiting Time')
            axes[1, 0].set_ylabel('Throughput')
            axes[1, 0].set_title('Performance Trade-off')
        
        # Best trial progression
        best_values = []
        current_best = float('inf')
        for trial in self.study.trials:
            if trial.value is not None and trial.value < current_best:
                current_best = trial.value
            best_values.append(current_best if current_best != float('inf') else None)
        
        valid_best_values = [v for v in best_values if v is not None]
        if valid_best_values:
            axes[1, 1].plot(valid_best_values)
            axes[1, 1].set_title('Best Value Progression')
            axes[1, 1].set_xlabel('Trial')
            axes[1, 1].set_ylabel('Best Objective Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def generate_optimization_report(self, output_file):
        """Generate comprehensive optimization report"""
        report = {
            'study_name': self.study_name,
            'optimization_date': datetime.now().isoformat(),
            'total_trials': len(self.study.trials),
            'completed_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'failed_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'best_trial': {
                'number': self.study.best_trial.number,
                'value': self.study.best_trial.value,
                'params': self.study.best_trial.params,
                'user_attrs': self.study.best_trial.user_attrs
            },
            'parameter_importance': {},
            'trial_history': []
        }
        
        # Add parameter importance if available
        try:
            importance = optuna.importance.get_param_importances(self.study)
            report['parameter_importance'] = importance
        except:
            pass
        
        # Add trial history
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data = {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'user_attrs': trial.user_attrs
                }
                report['trial_history'].append(trial_data)
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Optimization report saved to: {output_file}")
        return report

def multi_objective_optimization(study_name, n_trials=100):
    """Run multi-objective optimization"""
    print("Starting multi-objective optimization...")
    
    # Create multi-objective study
    study = optuna.create_study(
        study_name=f"{study_name}_multi",
        directions=['minimize', 'maximize', 'minimize'],  # waiting_time, throughput, fuel
        sampler=optuna.samplers.NSGAIISampler(seed=42)
    )
    
    def multi_objective(trial):
        """Multi-objective function"""
        # Get configuration
        base_config = {
            'DEFAULT_REWARD_WEIGHTS': SystemConfig.DEFAULT_REWARD_WEIGHTS,
            'NETWORK_CONFIG': SystemConfig.NETWORK_CONFIG,
            'TRAINING_CONFIG': SystemConfig.TRAINING_CONFIG
        }
        
        optimizer = HyperparameterOptimizer(base_config)
        config = optimizer.suggest_hyperparameters(trial)
        
        # Update system config
        temp_config_file = f"temp_multi_config_trial_{trial.number}.json"
        with open(temp_config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        SystemConfig.load_from_file(temp_config_file)
        
        try:
            # Run simulation
            performance_report = run_enhanced_simulation(
                train=True,
                model_name=f"multi_optim_trial_{trial.number}",
                epochs=25,
                steps=250
            )
            
            # Extract objectives
            waiting_time = performance_report['performance_trends']['total_waiting_time']['mean']
            throughput = performance_report['performance_trends']['throughput']['mean']
            fuel_consumption = performance_report['performance_trends']['fuel_consumption']['mean']
            
            # Clean up
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
            
            return waiting_time, throughput, fuel_consumption
            
        except Exception as e:
            print(f"Multi-objective trial {trial.number} failed: {e}")
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
            return float('inf'), 0, float('inf')
    
    # Run optimization
    study.optimize(multi_objective, n_trials=n_trials)
    
    # Plot Pareto front
    plot_pareto_front(study, f"pareto_front_{study_name}.png")
    
    return study

def plot_pareto_front(study, save_path=None):
    """Plot Pareto front for multi-objective optimization"""
    fig = plt.figure(figsize=(15, 5))
    
    # Extract trial data
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trials_data.append({
                'waiting_time': trial.values[0],
                'throughput': trial.values[1], 
                'fuel_consumption': trial.values[2],
                'trial_number': trial.number
            })
    
    if not trials_data:
        print("No completed trials for Pareto front plotting")
        return None
    
    df = pd.DataFrame(trials_data)
    
    # Plot 1: Waiting Time vs Throughput
    ax1 = fig.add_subplot(131)
    scatter1 = ax1.scatter(df['waiting_time'], df['throughput'], 
                          c=df['fuel_consumption'], cmap='viridis', alpha=0.7)
    ax1.set_xlabel('Waiting Time')
    ax1.set_ylabel('Throughput')
    ax1.set_title('Waiting Time vs Throughput')
    plt.colorbar(scatter1, ax=ax1, label='Fuel Consumption')
    
    # Plot 2: Waiting Time vs Fuel Consumption
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(df['waiting_time'], df['fuel_consumption'], 
                          c=df['throughput'], cmap='plasma', alpha=0.7)
    ax2.set_xlabel('Waiting Time')
    ax2.set_ylabel('Fuel Consumption')
    ax2.set_title('Waiting Time vs Fuel Consumption')
    plt.colorbar(scatter2, ax=ax2, label='Throughput')
    
    # Plot 3: Throughput vs Fuel Consumption
    ax3 = fig.add_subplot(133)
    scatter3 = ax3.scatter(df['throughput'], df['fuel_consumption'], 
                          c=df['waiting_time'], cmap='coolwarm', alpha=0.7)
    ax3.set_xlabel('Throughput')
    ax3.set_ylabel('Fuel Consumption')
    ax3.set_title('Throughput vs Fuel Consumption')
    plt.colorbar(scatter3, ax=ax3, label='Waiting Time')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def automated_hyperparameter_search(base_study_name, search_strategies=None):
    """Run automated hyperparameter search with multiple strategies"""
    if search_strategies is None:
        search_strategies = ['tpe', 'random', 'cmaes']
    
    results = {}
    
    for strategy in search_strategies:
        print(f"\n{'='*60}")
        print(f"Running optimization with {strategy.upper()} sampler")
        print(f"{'='*60}")
        
        # Create sampler based on strategy
        if strategy == 'tpe':
            sampler = optuna.samplers.TPESampler(seed=42)
        elif strategy == 'random':
            sampler = optuna.samplers.RandomSampler(seed=42)
        elif strategy == 'cmaes':
            sampler = optuna.samplers.CmaEsSampler(seed=42)
        else:
            print(f"Unknown strategy: {strategy}")
            continue
        
        # Create study
        study_name = f"{base_study_name}_{strategy}"
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            sampler=sampler
        )
        
        # Run optimization
        optimization_study = OptimizationStudy(study_name)
        optimization_study.study = study
        optimization_study.run_optimization(n_trials=50)
        
        # Store results
        results[strategy] = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'study': study
        }
        
        # Save best config for this strategy
        optimization_study.save_best_config(f"best_config_{strategy}.json")
        optimization_study.plot_optimization_history(f"optimization_history_{strategy}.png")
    
    # Compare strategies
    print(f"\n{'='*60}")
    print("STRATEGY COMPARISON")
    print(f"{'='*60}")
    
    for strategy, result in results.items():
        print(f"{strategy.upper():>10}: {result['best_value']:.2f} (trials: {result['n_trials']})")
    
    # Find best strategy
    best_strategy = min(results.keys(), key=lambda x: results[x]['best_value'])
    print(f"\nBest strategy: {best_strategy.upper()} with value {results[best_strategy]['best_value']:.2f}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Best values comparison
    strategies_list = list(results.keys())
    best_values = [results[s]['best_value'] for s in strategies_list]
    
    ax1.bar(strategies_list, best_values)
    ax1.set_title('Best Objective Value by Strategy')
    ax1.set_ylabel('Objective Value')
    
    # Convergence comparison
    for strategy, result in results.items():
        study = result['study']
        objectives = [trial.value for trial in study.trials if trial.value is not None]
        best_so_far = []
        current_best = float('inf')
        
        for obj in objectives:
            if obj < current_best:
                current_best = obj
            best_so_far.append(current_best)
        
        ax2.plot(best_so_far, label=strategy.upper(), alpha=0.8)
    
    ax2.set_title('Convergence Comparison')
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Best Objective Value')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"strategy_comparison_{base_study_name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

def sensitivity_analysis(base_config, parameter_ranges, n_samples=20):
    """Perform sensitivity analysis on key parameters"""
    print("Performing sensitivity analysis...")
    
    results = {}
    
    for param_name, param_range in parameter_ranges.items():
        print(f"Analyzing sensitivity for: {param_name}")
        
        param_results = []
        param_values = np.linspace(param_range[0], param_range[1], n_samples)
        
        for i, param_value in enumerate(param_values):
            print(f"  Sample {i+1}/{n_samples}: {param_name} = {param_value:.4f}")
            
            # Create modified config
            config = base_config.copy()
            
            # Navigate nested dictionaries to set parameter
            config_path = param_name.split('.')
            current_config = config
            for key in config_path[:-1]:
                current_config = current_config[key]
            current_config[config_path[-1]] = param_value
            
            # Save temporary config
            temp_config_file = f"temp_sensitivity_{param_name.replace('.', '_')}_{i}.json"
            with open(temp_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            try:
                # Update system config
                SystemConfig.load_from_file(temp_config_file)
                
                # Run simulation
                performance_report = run_enhanced_simulation(
                    train=True,
                    model_name=f"sensitivity_{param_name.replace('.', '_')}_{i}",
                    epochs=15,  # Reduced for faster analysis
                    steps=200
                )
                
                # Extract performance
                waiting_time = performance_report['performance_trends']['total_waiting_time']['mean']
                param_results.append({'value': param_value, 'performance': waiting_time})
                
            except Exception as e:
                print(f"    Failed: {e}")
                param_results.append({'value': param_value, 'performance': float('inf')})
            
            finally:
                # Clean up
                if os.path.exists(temp_config_file):
                    os.remove(temp_config_file)
        
        results[param_name] = param_results
    
    # Plot sensitivity results
    plot_sensitivity_analysis(results, f"sensitivity_analysis.png")
    
    return results

def plot_sensitivity_analysis(sensitivity_results, save_path=None):
    """Plot sensitivity analysis results"""
    n_params = len(sensitivity_results)
    cols = min(3, n_params)
    rows = (n_params + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if n_params == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (param_name, results) in enumerate(sensitivity_results.items()):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        values = [r['value'] for r in results if r['performance'] != float('inf')]
        performances = [r['performance'] for r in results if r['performance'] != float('inf')]
        
        if values and performances:
            ax.plot(values, performances, 'bo-', alpha=0.7)
            ax.set_xlabel(param_name)
            ax.set_ylabel('Waiting Time')
            ax.set_title(f'Sensitivity: {param_name}')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params, rows * cols):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        elif cols > 1:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def main():
    """Main function for hyperparameter optimization"""
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for Enhanced Traffic Light System')
    
    parser.add_argument('--study-name', default='traffic_optimization', 
                       help='Name of the optimization study')
    parser.add_argument('--trials', type=int, default=100, 
                       help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=None, 
                       help='Timeout in seconds')
    parser.add_argument('--multi-objective', action='store_true', 
                       help='Run multi-objective optimization')
    parser.add_argument('--strategy-comparison', action='store_true', 
                       help='Compare different optimization strategies')
    parser.add_argument('--sensitivity-analysis', action='store_true', 
                       help='Perform sensitivity analysis')
    parser.add_argument('--output-dir', default='optimization_results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"{args.study_name}_{timestamp}"
    
    if args.multi_objective:
        print("Running multi-objective optimization...")
        multi_study = multi_objective_optimization(study_name, args.trials)
        
    elif args.strategy_comparison:
        print("Running strategy comparison...")
        strategy_results = automated_hyperparameter_search(study_name)
        
    elif args.sensitivity_analysis:
        print("Running sensitivity analysis...")
        
        # Define parameter ranges for sensitivity analysis
        param_ranges = {
            'NETWORK_CONFIG.learning_rate': [1e-5, 1e-2],
            'NETWORK_CONFIG.hidden_dims': [128, 512],
            'TRAINING_CONFIG.gamma': [0.9, 0.999],
            'TRAINING_CONFIG.epsilon_decay': [0.99, 0.999],
            'DEFAULT_REWARD_WEIGHTS.waiting_time': [-2.0, -0.1],
            'DEFAULT_REWARD_WEIGHTS.coordination': [0.5, 3.0]
        }
        
        base_config = {
            'DEFAULT_REWARD_WEIGHTS': SystemConfig.DEFAULT_REWARD_WEIGHTS,
            'NETWORK_CONFIG': SystemConfig.NETWORK_CONFIG,
            'TRAINING_CONFIG': SystemConfig.TRAINING_CONFIG
        }
        
        sensitivity_results = sensitivity_analysis(base_config, param_ranges)
        
        # Save sensitivity results
        with open(f"sensitivity_results_{study_name}.json", 'w') as f:
            json.dump(sensitivity_results, f, indent=2)
        
    else:
        # Standard single-objective optimization
        print("Running single-objective optimization...")
        
        optimization_study = OptimizationStudy(study_name)
        optimization_study.run_optimization(args.trials, args.timeout)
        
        # Save results
        optimization_study.save_best_config(f"best_config_{study_name}.json")
        optimization_study.plot_optimization_history(f"optimization_history_{study_name}.png")
        optimization_study.generate_optimization_report(f"optimization_report_{study_name}.json")
    
    print(f"\nOptimization completed! Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()