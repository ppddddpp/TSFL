from eval.eval import eval_full_pipeline
import numpy as np

# Poison ratios to test: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30

def run_case_4_generic(ratio_val, seeds_to_test, defense_to_run, dataset_to_run):
    num_seeds = len(seeds_to_test)
    all_results = {}

    alpha_default = 0.5 
    attack_to_use = "spa"

    all_results[ratio_val] = {}
    
    print(f"\n==================================================")
    print(f"Starting Test: Poison Ratio = {ratio_val*100}% (Trigger injected in {ratio_val*100}% of target class images)")
    print(f"==================================================")
    
    for s in seeds_to_test:
        seed_results_dict = eval_full_pipeline(
            dataset_name=dataset_to_run,
            defense_name=defense_to_run,
            alpha=alpha_default,
            attack_category=attack_to_use,  
            update_method="fedavg",
            percent_attackers=0.25,
            num_rounds=200,             
            n_clients=100,
            clients_per_round=25,
            seed=s,
            batch_size=64,
            poison_ratio=ratio_val
        )

        for defense_name, results_dict in seed_results_dict.items():
            if defense_name not in all_results[ratio_val]:
                all_results[ratio_val][defense_name] = []

            val = results_dict.get('acc', results_dict.get('accuracy', 0.0))
            all_results[ratio_val][defense_name].append(val)
            
    print("\n\n==================================================")
    print(f"FINAL RESULTS: {num_seeds}-SEED AVERAGE (SPA Attack)")
    print("==================================================")
    
    print(f"--- Poison Ratio: {ratio_val} ---")
    print(f"{'Defense':<15} | {'Avg Accuracy':<15} | {'Std Dev':<15}")
    print("-" * 50)
    
    for defense_name, scores in all_results[ratio_val].items():
        avg_acc = np.mean(scores)
        std_dev = np.std(scores)
        print(f"{defense_name.upper():<15} | {avg_acc:.2f}%          | ±{std_dev:.2f}%")
        
    print("==================================================\n")

def case_4_poison_ratio_0_0_5(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_4_generic(0.05, seeds_to_test, defense_to_run, dataset_to_run)

def case_4_poison_ratio_0_1_0(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_4_generic(0.10, seeds_to_test, defense_to_run, dataset_to_run)

def case_4_poison_ratio_0_1_5(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_4_generic(0.15, seeds_to_test, defense_to_run, dataset_to_run)

def case_4_poison_ratio_0_2_0(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_4_generic(0.20, seeds_to_test, defense_to_run, dataset_to_run)

def case_4_poison_ratio_0_2_5(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_4_generic(0.25, seeds_to_test, defense_to_run, dataset_to_run)

def case_4_poison_ratio_0_3_0(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_4_generic(0.30, seeds_to_test, defense_to_run, dataset_to_run)