import time
import os
import argparse
import warnings
import threading
from pathlib import Path
import multiprocessing

from jobs_handling import ResourceMonitor, HistoryManager
from jobs_handling import Job, ResourceAwareScheduler

warnings.filterwarnings("ignore", message="Loky-backed parallel loops")
warnings.simplefilter(action='ignore', category=FutureWarning)

from case_handling.case1 import (
    case_1_alpha_0_1, case_1_alpha_0_2, case_1_alpha_0_3, 
    case_1_alpha_0_4, case_1_alpha_0_5, case_1_alpha_1_0
)
from case_handling.case2 import (
    case_2_sign_flip, case_2_noise, case_2_random, case_2_lie, 
    case_2_min_max, case_2_min_sum, case_2_byzmean, case_2_tailored_trmean,
    case_2_cba, case_2_dba, case_2_spa
)
from case_handling.case3 import (
    case_3_attack_ratio_0_0_5, case_3_attack_ratio_0_1_0, case_3_attack_ratio_0_1_5, 
    case_3_attack_ratio_0_2_0, case_3_attack_ratio_0_2_5, case_3_attack_ratio_0_3_0
)
from case_handling.case4 import (
    case_4_poison_ratio_0_0_5, case_4_poison_ratio_0_1_0, case_4_poison_ratio_0_1_5, case_4_poison_ratio_0_2_0, 
    case_4_poison_ratio_0_2_5, case_4_poison_ratio_0_3_0
)
from case_handling.case5 import (
    case_5_alpha_0_1, case_5_alpha_0_2, case_5_alpha_0_3, 
    case_5_alpha_0_4, case_5_alpha_0_5, case_5_alpha_1_0
)
from case_handling.case6 import (
    case_6_attack_ratio_0_0_5, case_6_attack_ratio_0_1_0, case_6_attack_ratio_0_1_5, 
    case_6_attack_ratio_0_2_0, case_6_attack_ratio_0_2_5, case_6_attack_ratio_0_3_0
)
from case_handling.case7 import (
    case_7_sign_flip, case_7_noise, case_7_random, case_7_lie, 
    case_7_min_max, case_7_min_sum, case_7_byzmean, case_7_tailored_trmean,
    case_7_cba, case_7_dba, case_7_spa
)
from case_handling.no_attack import (
    case_no_attack_iid, case_no_attack_non_iid
)

class RealTimeLoggingScheduler(ResourceAwareScheduler):
    def __init__(self, defense_files_map, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.defense_files_map = defense_files_map
        self.logged_job_ids = set()

    def _reap_finished_jobs(self):
        super()._reap_finished_jobs()
        for f_job in self.finished:
            if f_job.job.job_id not in self.logged_job_ids:
                defense_name = f_job.job.args[2]
                files = self.defense_files_map[defense_name]
                with open(files["timing"], "a") as tf:
                    tf.write(f"{f_job.job.name},{f_job.status},{f_job.runtime:.2f}\n")
                
                if f_job.status == "SUCCESS":
                    with open(files["tracker"], "a") as tr:
                        tr.write(f"{f_job.job.name}\n")
                    print(f" Tracker updated: {f_job.job.name} ({defense_name.upper()})")
                else:
                    print(f" {f_job.job.name} FAILED. Tracker not updated.")
                        
                self.logged_job_ids.add(f_job.job.job_id)

def live_monitor_global(scheduler: ResourceAwareScheduler, stop_event, status_file_path, total_to_run, interval=60):
    while not stop_event.wait(interval):
        completed_count = len(scheduler.finished)
        
        report = "\n" + "="*65 + "\n"
        report += f"[GLOBAL LIVE DASHBOARD] - Updated at: {time.strftime('%H:%M:%S')}\n"
        report += f"Session Progress: {completed_count} / {total_to_run} cases completed\n"
        report += "-" * 65 + "\n"
        
        report += f"Currently running cases ({len(scheduler.running)} active threads):\n"
        if not scheduler.running:
            report += "   (Waiting for jobs to load...)\n"
        
        for rjob in scheduler.running.values():
            elapsed = rjob.runtime / 60
            def_name = rjob.job.args[2].upper()
            report += f"    GPU {rjob.gpu_id:2d} | [{def_name}] {rjob.job.name:<25} | Elapsed: {elapsed:>5.1f} min\n"
        report += "-" * 65 + "\n"

        hw = scheduler.monitor.snapshot()
        report += f"System RAM Used : {hw.ram_used_mb/1024:.1f} / {hw.total_ram_mb/1024:.1f} GB\n"
        report += f"CPU Load        : {hw.cpu_percent:.1f}% / {hw.cpu_count} cores\n"
        
        report += "GPU Status:\n"
        if not hw.gpus:
            report += "  No GPUs detected.\n"
        for gpu in hw.gpus:
            report += f"  [GPU {gpu.gpu_id}] VRAM: {gpu.used_vram_mb:.0f} / {gpu.total_vram_mb:.0f} MB | Util: {gpu.gpu_util_percent}%\n"
            
        report += "="*65 + "\n"

        try:
            with open(status_file_path, "w") as f:
                f.write(report)
        except Exception:
            pass

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Auto-Benchmark Pipeline for SybilGuard")
    parser.add_argument("--seed", type=int, default=2709, help="Random seed (e.g., 2709, 42)")
    parser.add_argument("--force-restart", action="store_true", help="Delete old checkpoints/tracker")
    parser.add_argument("--defense", type=str, default="all", help="all OR comma-separated (e.g., lasa,fltrust,krum)")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use")
    args = parser.parse_args()

    current_dataset = args.dataset.lower().strip()
    if args.defense.lower().strip() == "all":
        defenses_to_run = ["tsfl", "lasa", "fltrust", "foolsgold", "signguard", "feddlad"]
    else:
        defenses_to_run = [d.strip().lower() for d in args.defense.split(",")]

    OUTPUT_DIR = Path(__file__).parents[1] / "output"
    MAIN_CASE_DATA_DIR = OUTPUT_DIR / f"case_data_{current_dataset}"
    RUN_LOGS_DIR = MAIN_CASE_DATA_DIR / "RUN_LOGS" / f"seed_{args.seed}"
    GLOBAL_DASHBOARD_FILE = RUN_LOGS_DIR / "dashboard" / "live_status_GLOBAL.txt"

    # Define Cases
    all_cases = [
        case_no_attack_iid, case_no_attack_non_iid,

        case_1_alpha_0_1, case_1_alpha_0_2, case_1_alpha_0_3, 
        case_1_alpha_0_4, case_1_alpha_0_5, case_1_alpha_1_0,

        case_2_sign_flip, case_2_noise, case_2_random, case_2_lie, 
        case_2_min_max, case_2_min_sum, case_2_byzmean, case_2_tailored_trmean,

        case_2_cba, case_2_dba, case_2_spa,

        case_3_attack_ratio_0_0_5, case_3_attack_ratio_0_1_0, case_3_attack_ratio_0_1_5, 
        case_3_attack_ratio_0_2_0, case_3_attack_ratio_0_2_5, case_3_attack_ratio_0_3_0,

        case_4_poison_ratio_0_0_5, case_4_poison_ratio_0_1_0, case_4_poison_ratio_0_1_5, 
        case_4_poison_ratio_0_2_0, case_4_poison_ratio_0_2_5, case_4_poison_ratio_0_3_0,

        case_5_alpha_0_1, case_5_alpha_0_2, case_5_alpha_0_3, 
        case_5_alpha_0_4, case_5_alpha_0_5, case_5_alpha_1_0,

        case_6_attack_ratio_0_0_5, case_6_attack_ratio_0_1_0, case_6_attack_ratio_0_1_5, 
        case_6_attack_ratio_0_2_0, case_6_attack_ratio_0_2_5, case_6_attack_ratio_0_3_0,

        case_7_sign_flip, case_7_noise, case_7_random, case_7_lie, 
        case_7_min_max, case_7_min_sum, case_7_byzmean, case_7_tailored_trmean,
        
        case_7_cba, case_7_dba, case_7_spa
    ]

    manager = multiprocessing.Manager()
    monitor = ResourceMonitor()
    history_mgr = HistoryManager(history_file=str(RUN_LOGS_DIR / "history_global.json"))

    defense_files_map = {}
    pending_jobs_to_submit = []

    for defense in defenses_to_run:
        TRACKER_DIR = RUN_LOGS_DIR / "tracker"
        DASHBOARD_DIR = RUN_LOGS_DIR / "dashboard"
        BUGS_DIR = RUN_LOGS_DIR / "bugs" / defense
        TRACKER_DIR.mkdir(parents=True, exist_ok=True)
        DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
        BUGS_DIR.mkdir(parents=True, exist_ok=True)

        tracker_file = str(TRACKER_DIR / f"completed_{defense}.txt")
        timing_file = str(DASHBOARD_DIR / f"execution_times_{defense}.csv")
        
        open(tracker_file, 'a').close()
        if not os.path.exists(timing_file):
            with open(timing_file, "w") as tf:
                tf.write("case_name,status,runtime_seconds\n")
                
        defense_files_map[defense] = {"tracker": tracker_file, "timing": timing_file}

        completed_cases = set()
        with open(tracker_file, "r") as f:
            completed_cases = set(line.strip() for line in f if line.strip())
            
        pending_cases = [c for c in all_cases if c.__name__ not in completed_cases]
        print(f" -> [{defense.upper()}] completed: {len(completed_cases)}/54 | pending: {len(pending_cases)}")

        for case_func in pending_cases:
            pending_jobs_to_submit.append((case_func, defense, str(BUGS_DIR)))

    total_pending_jobs = len(pending_jobs_to_submit)
    if total_pending_jobs == 0:
        print("\n ALL DEFENSES ARE FULLY COMPLETED! Exiting...")
        exit(0)

    scheduler = RealTimeLoggingScheduler(
        defense_files_map=defense_files_map,
        monitor=monitor, 
        manager=manager, 
        history_manager=history_mgr, 
        gpu_multiplier=14,
        max_retries=1
    )

    for idx, (case_func, defense, bugs_dir) in enumerate(pending_jobs_to_submit):
        job_name = case_func.__name__
        est_runtime, req_vram, req_ram = history_mgr.get_estimate(job_name, default_vram=4000.0, default_ram=4096.0)
        
        job = Job(
            job_id=idx,
            name=job_name,
            args=(case_func, args.seed, defense, current_dataset, bugs_dir),
            required_vram_mb=req_vram,
            required_ram_mb=req_ram,
            estimated_runtime=est_runtime
        )
        scheduler.submit(job)

    print(f"====================================================")
    print(f"TOTAL JOBS IN QUEUE: {total_pending_jobs}")
    print(f"====================================================\n")

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=live_monitor_global, 
        args=(scheduler, stop_event, str(GLOBAL_DASHBOARD_FILE), total_pending_jobs, 60)
    )
    monitor_thread.daemon = True
    monitor_thread.start()

    total_start = time.time()
    scheduler.run()

    stop_event.set()
    monitor_thread.join()

    print(f"\n========== GLOBAL RUN COMPLETED IN {(time.time() - total_start) / 3600:.2f} HOURS! ==========")