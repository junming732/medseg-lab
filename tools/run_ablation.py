"""
Run Ablation Study

Runs multiple experiments with different prompt encoders and compares results.

Usage:
    python tools/run_ablation.py --config configs/ablation_study.yaml
"""

import os
import sys
import argparse
import yaml
import subprocess
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path):
    """Load ablation study configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_experiment_config(base_config_path, experiment_config, output_dir):
    """
    Create experiment-specific config by modifying base config.

    Args:
        base_config_path: Path to base configuration
        experiment_config: Dict with experiment-specific settings
        output_dir: Where to save the new config

    Returns:
        config_path: Path to the created config file
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update with experiment-specific settings
    if 'encoder' in experiment_config:
        config['data']['encoder'] = experiment_config['encoder']

    if 'name' in experiment_config:
        config['experiment']['name'] = experiment_config['name']

    # Save modified config
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, f"{experiment_config['name']}.yaml")

    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    return config_path


def run_experiment(config_path, seed=42):
    """
    Run a single training experiment.

    Args:
        config_path: Path to experiment config
        seed: Random seed

    Returns:
        success: Boolean indicating if training succeeded
    """
    # Modify seed in config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['experiment']['seed'] = seed

    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Run training
    cmd = [
        'python', 'tools/train.py',
        '--config', config_path
    ]

    print(f"\nRunning command: {' '.join(cmd)}")
    print("="*60)

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")
        return False


def collect_results(output_dir, experiment_name):
    """
    Collect results from a completed experiment.

    Args:
        output_dir: Base output directory
        experiment_name: Name of the experiment

    Returns:
        results: Dict with metrics
    """
    exp_dir = os.path.join(output_dir, experiment_name)

    # Look for results file (you'd need to save this during training)
    results_file = os.path.join(exp_dir, 'results.json')

    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    else:
        print(f"Warning: No results found for {experiment_name}")
        return None


def generate_report(all_results, output_path):
    """
    Generate comparison report.

    Args:
        all_results: Dict of experiment_name -> results
        output_path: Where to save the report
    """
    report = []
    report.append("="*80)
    report.append("ABLATION STUDY RESULTS")
    report.append("="*80)
    report.append("")

    # Table header
    report.append(f"{'Experiment':<30} {'Dice↑':<12} {'HD95↓':<12} {'Time':<12}")
    report.append("-"*80)

    # Sort by Dice score
    sorted_experiments = sorted(
        all_results.items(),
        key=lambda x: x[1].get('val_dice', 0) if x[1] else 0,
        reverse=True
    )

    for exp_name, results in sorted_experiments:
        if results:
            dice = results.get('val_dice', 0.0)
            hd95 = results.get('val_hd95', 999.9)
            time = results.get('training_time_hours', 0.0)

            report.append(f"{exp_name:<30} {dice:<12.4f} {hd95:<12.2f} {time:<12.2f}h")
        else:
            report.append(f"{exp_name:<30} {'FAILED':<12}")

    report.append("="*80)

    # Save report
    report_text = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to ablation study config')
    parser.add_argument('--base_config', type=str, default='configs/gaussian_s3.yaml',
                       help='Base experiment config to modify')
    parser.add_argument('--output_dir', type=str, default='./ablation_results',
                       help='Directory for ablation study outputs')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, only collect results')
    args = parser.parse_args()

    # Load ablation config
    ablation_config = load_config(args.config)

    print("="*80)
    print(f"ABLATION STUDY: {ablation_config['ablation_study']['name']}")
    print("="*80)

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_dir = os.path.join(args.output_dir, f"{ablation_config['ablation_study']['name']}_{timestamp}")
    os.makedirs(study_dir, exist_ok=True)

    print(f"Study directory: {study_dir}")
    print(f"Number of experiments: {len(ablation_config['ablation_study']['experiments'])}")
    print(f"Number of seeds: {ablation_config.get('n_seeds', 1)}")

    # Run experiments
    all_results = {}
    experiments = ablation_config['ablation_study']['experiments']
    seeds = ablation_config.get('seeds', [42])

    if not args.skip_training:
        for exp_idx, exp_config in enumerate(experiments):
            exp_name = exp_config['name']

            print(f"\n{'='*80}")
            print(f"EXPERIMENT {exp_idx+1}/{len(experiments)}: {exp_name}")
            print('='*80)

            for seed_idx, seed in enumerate(seeds):
                print(f"\nSeed {seed_idx+1}/{len(seeds)}: {seed}")

                # Create experiment config
                exp_name_with_seed = f"{exp_name}_seed{seed}"
                config_path = create_experiment_config(
                    args.base_config,
                    {**exp_config, 'name': exp_name_with_seed},
                    study_dir
                )

                # Run experiment
                success = run_experiment(config_path, seed=seed)

                if not success:
                    print(f"Failed: {exp_name_with_seed}")
                    all_results[exp_name_with_seed] = None

    # Collect results
    print("\n" + "="*80)
    print("COLLECTING RESULTS")
    print("="*80)

    for exp_config in experiments:
        exp_name = exp_config['name']
        for seed in seeds:
            exp_name_with_seed = f"{exp_name}_seed{seed}"
            results = collect_results(
                ablation_config['experiment']['output_dir'],
                exp_name_with_seed
            )
            all_results[exp_name_with_seed] = results

    # Generate report
    report_path = os.path.join(study_dir, 'ablation_report.txt')
    generate_report(all_results, report_path)

    print(f"\nAblation study complete!")
    print(f"Results saved to: {study_dir}")


if __name__ == "__main__":
    main()