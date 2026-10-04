from eval.eval import eval_full_pipeline
import numpy as np

# [sign_flip, noise, random, lie, min_max, min_sum, byzmean, tailored_trmean]
# [cba, dba, spa]

def run_case_2_generic(attack_name, seeds_to_test, defense_to_run, dataset_to_run):
    num_seeds = len(seeds_to_test)
    all_results = {}

    alpha_iid = 0.5
    poison_ratio = 0.2

    all_results[attack_name] = {}

    print(f"\n{'#'*60}")
    print(f" STARTING BENCHMARK FOR ATTACK: {attack_name.upper()}")
    print(f"{'#'*60}\n")

    for s in seeds_to_test:
        seed_results_dict = eval_full_pipeline(
            dataset_name=dataset_to_run,
            defense_name=defense_to_run,
            alpha=alpha_iid,
            attack_category=attack_name,  
            update_method="fedavg",
            percent_attackers=0.25,
            num_rounds=200, 
            n_clients=100,
            clients_per_round=25,
            seed=s,
            batch_size=64,
            poison_ratio=poison_ratio
        )
        
        for defense_name, results_dict in seed_results_dict.items():
            if defense_name not in all_results[attack_name]:
                all_results[attack_name][defense_name] = []

            val = results_dict.get('acc', results_dict.get('accuracy', 0.0))
            all_results[attack_name][defense_name].append(val)
            
    print("\n\n==================================================")
    print(f"FINAL RESULTS: {num_seeds}-SEED AVERAGE ({attack_name.upper()} Attack)")
    print("==================================================")
    
    print(f"--- ATTACK: {attack_name.upper()} ---")
    print(f"{'Defense':<15} | {'Avg Accuracy':<15} | {'Std Dev':<15}")
    print("-" * 50)
    
    for defense_name, scores in all_results[attack_name].items():
        avg_acc = np.mean(scores)
        std_dev = np.std(scores)
        print(f"{defense_name.upper():<15} | {avg_acc:.2f}%          | ±{std_dev:.2f}%")
        
    print("==================================================\n")

def case_2_sign_flip(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_2_generic("sign_flip", seeds_to_test, defense_to_run, dataset_to_run)

def case_2_noise(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_2_generic("noise", seeds_to_test, defense_to_run, dataset_to_run)

def case_2_random(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_2_generic("random", seeds_to_test, defense_to_run, dataset_to_run)

def case_2_lie(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_2_generic("lie", seeds_to_test, defense_to_run, dataset_to_run)

def case_2_min_max(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_2_generic("min_max", seeds_to_test, defense_to_run, dataset_to_run)

def case_2_min_sum(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_2_generic("min_sum", seeds_to_test, defense_to_run, dataset_to_run)

def case_2_tailored_trmean(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_2_generic("tailored_trmean", seeds_to_test, defense_to_run, dataset_to_run)

def case_2_byzmean(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_2_generic("byzmean", seeds_to_test, defense_to_run, dataset_to_run)

def case_2_cba(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_2_generic("cba", seeds_to_test, defense_to_run, dataset_to_run)

def case_2_dba(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_2_generic("dba", seeds_to_test, defense_to_run, dataset_to_run)

def case_2_spa(seeds_to_test=[27, 9, 2709], defense_to_run="all", dataset_to_run="cifar10"):
    run_case_2_generic("spa", seeds_to_test, defense_to_run, dataset_to_run)