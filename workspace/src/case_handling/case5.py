from eval.eval import eval_full_pipeline
import numpy as np

# Dirichlet alphas to test: 0.1, 0.2, 0.3, 0.4, 0.5, 1.0

def run_case_5_generic(alpha_val, seeds_to_test, defense_to_run, dataset_to_run):
    num_seeds = len(seeds_to_test)
    all_results = {}
    
    all_results[alpha_val] = {}
    
    print(f"\n{'#'*60}")
    print(f" STARTING BENCHMARK FOR ALPHA = {alpha_val}")
    print(f"{'#'*60}\n")
    
    for s in seeds_to_test:
        seed_results_dict = eval_full_pipeline(
            dataset_name=dataset_to_run,
            defense_name=defense_to_run,
            alpha=alpha_val,
            attack_category="spa", 
            update_method="fedavg",
            percent_attackers=0.25,           
            num_rounds=200,             
            n_clients=100,
            clients_per_round=25,
            seed=s,
            batch_size=64,
            poison_ratio=0.2
        )

        for defense_name, results_dict in seed_results_dict.items():
            if defense_name not in all_results[alpha_val]:
                all_results[alpha_val][defense_name] = []

            val = results_dict.get('acc', results_dict.get('accuracy', 0.0))
            all_results[alpha_val][defense_name].append(val)
            
    print(f"\n>>> Completed Alpha {alpha_val} across {num_seeds} seeds. <<<")

    print("\n\n==================================================")
    print(f"FINAL RESULTS: {num_seeds}-SEED AVERAGE (SPA Attack)")
    print("==================================================")
    
    print(f"--- ALPHA: {alpha_val} ---")
    print(f"{'Defense':<15} | {'Avg Accuracy':<15} | {'Std Dev':<15}")
    print("-" * 50)
    
    for defense_name, scores in all_results[alpha_val].items():
        avg_acc = np.mean(scores)
        std_dev = np.std(scores)
        print(f"{defense_name.upper():<15} | {avg_acc:.2f}%          | ±{std_dev:.2f}%")
        
    print("==================================================\n")

def case_5_alpha_0_1(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_5_generic(0.1, seeds_to_test, defense_to_run, dataset_to_run)

def case_5_alpha_0_2(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_5_generic(0.2, seeds_to_test, defense_to_run, dataset_to_run)

def case_5_alpha_0_3(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_5_generic(0.3, seeds_to_test, defense_to_run, dataset_to_run)

def case_5_alpha_0_4(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_5_generic(0.4, seeds_to_test, defense_to_run, dataset_to_run)

def case_5_alpha_0_5(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_5_generic(0.5, seeds_to_test, defense_to_run, dataset_to_run)

def case_5_alpha_1_0(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_5_generic(1.0, seeds_to_test, defense_to_run, dataset_to_run)