import os
import time
import traceback
import subprocess
import psutil
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Optional
from .resource_monitor import ResourceMonitor
from .history import HistoryManager

@dataclass
class Job:
    job_id: int
    name: str
    args: tuple
    priority: int = 0
    submit_time: float = field(default_factory=time.time)
    retry: int = 0
    required_vram_mb: float = 1500.0 
    required_ram_mb: float = 2048.0
    estimated_runtime: float = 0.0
    next_allowed_time: float = 0.0

@dataclass
class RunningJob:
    job: Job
    process: mp.Process
    result_queue: Any
    gpu_id: int
    pid: int
    start_time: float
    # Live tracking
    peak_real_vram_mb: float = 0.0
    peak_real_ram_mb: float = 0.0

    @property
    def runtime(self):
        return time.time() - self.start_time

@dataclass
class FinishedJob:
    job: Job
    status: str
    runtime: float
    result: Any = None
    gpu_id: Optional[int] = None
    error: Optional[str] = None

def isolated_worker(job: Job, gpu_id: int, result_queue):
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" 
        
    try:
        case_func, seed, defense_name, dataset_name, bugs_dir = job.args
        case_func(seeds_to_test=[seed], defense_to_run=defense_name, dataset_to_run=dataset_name)

        pytorch_vram = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                pytorch_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        except ImportError:
            pass
        
        result_queue.put({"status": "SUCCESS", "error": None, "pytorch_vram_mb": pytorch_vram})
        
    except Exception as e:
        error_trace = traceback.format_exc()
        if 'bugs_dir' in locals():
            log_path = os.path.join(bugs_dir, f"ERROR_{job.name}_{job.job_id}.log")
            with open(log_path, "w") as ef:
                ef.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CRASH IN: {job.name}\n")
                ef.write(error_trace)
                
        result_queue.put({"status": "FAILED", "error": error_trace, "pytorch_vram_mb": 0.0})

class ResourceAwareScheduler:
    def __init__(self, monitor: ResourceMonitor, manager: mp.Manager, 
                 history_manager: HistoryManager, max_retries: int = 3, gpu_multiplier: int = 3):
        self.monitor = monitor
        self.history_manager = history_manager
        self.manager = manager
        self.max_retries = max_retries
        
        hw_init = monitor.snapshot()
        self.num_gpus = max(1, len(hw_init.gpus))
        self.max_concurrent = self.num_gpus * gpu_multiplier 
        
        self.pending: list[Job] = []
        self.running: dict[int, RunningJob] = {}
        self.finished: list[FinishedJob] = []
        
        self.gpu_util_ema = {g.gpu_id: 0.0 for g in hw_init.gpus}
        self.gpu_locked_until = {g.gpu_id: 0.0 for g in hw_init.gpus}

    def submit(self, job: Job):
        self.pending.append(job)

    def run(self):
        while self.pending or self.running:
            self._track_live_processes()
            self._reap_finished_jobs()
            self._schedule_pending_jobs()
            for job in self.pending:
                job.priority += 1 
            time.sleep(2)

    def _track_live_processes(self):
        hw = self.monitor.snapshot()
        for g in hw.gpus:
            old_util = self.gpu_util_ema.get(g.gpu_id, 0.0)
            self.gpu_util_ema[g.gpu_id] = (0.2 * g.gpu_util_percent) + (0.8 * old_util)

        for rjob in self.running.values():
            if rjob.process.is_alive():
                try:
                    p = psutil.Process(rjob.pid)
                    ram_mb = p.memory_info().rss / (1024 * 1024)
                    rjob.peak_real_ram_mb = max(rjob.peak_real_ram_mb, ram_mb)

                    if rjob.gpu_id >= 0 and hasattr(p, 'cpu_affinity'):
                        total_cores = psutil.cpu_count()
                        cores_per_gpu = total_cores // self.num_gpus
                        start_core = rjob.gpu_id * cores_per_gpu
                        p.cpu_affinity(list(range(start_core, start_core + cores_per_gpu)))
                except: pass
                
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True).strip()
            pid_vram = {int(line.split(",")[0]): float(line.split(",")[1]) for line in out.splitlines() if line.strip()}
            for rjob in self.running.values():
                if rjob.pid in pid_vram:
                    rjob.peak_real_vram_mb = max(rjob.peak_real_vram_mb, pid_vram[rjob.pid])
        except: pass

    def _reap_finished_jobs(self):
        done_ids = []
        for jid, rjob in self.running.items():
            if not rjob.process.is_alive():
                if rjob.gpu_id >= 0:
                    self.gpu_locked_until[rjob.gpu_id] = time.time() + 5.0

                exit_code = rjob.process.exitcode
                if not rjob.result_queue.empty():
                    res = rjob.result_queue.get()
                else:
                    if exit_code == -9 or exit_code == 137:
                        res = {"status": "FAILED", "error": "SYSTEM OOM (Killed by Linux OS - Code -9)"}
                    else:
                        res = {"status": "FAILED", "error": f"Process crashed. Exit code: {exit_code}"}
                
                rjob.process.join()
                self._handle_job_completion(jid, rjob, res, done_ids)
                
        for jid in done_ids:
            del self.running[jid]

    def _handle_job_completion(self, jid, rjob, res, done_ids):
        if res["status"] == "FAILED" and rjob.job.retry < self.max_retries:
            rjob.job.retry += 1
            rjob.job.priority += 1000
            delay = 5 * (3 ** (rjob.job.retry - 1))
            rjob.job.next_allowed_time = time.time() + delay
            print(f"[RETRY] {rjob.job.name} failed (GPU {rjob.gpu_id}). Re-queueing (Attempt {rjob.job.retry}/{self.max_retries})...")
            self.submit(rjob.job)
            done_ids.append(jid)
            return
        
        pytorch_vram = res.get("pytorch_vram_mb", 0.0)
        final_peak_vram = max(rjob.peak_real_vram_mb, pytorch_vram)
        final_peak_ram = rjob.peak_real_ram_mb
        
        if res["status"] == "SUCCESS" and final_peak_vram > 0:
            self.history_manager.update(rjob.job.name, rjob.runtime, final_peak_vram, final_peak_ram)
        
        f_job = FinishedJob(
            job=rjob.job, status=res["status"], runtime=rjob.runtime,
            error=res["error"], gpu_id=rjob.gpu_id
        )
        self.finished.append(f_job)
        done_ids.append(jid)
        print(f"[FINISHED] {rjob.job.name} | Peak VRAM: {final_peak_vram:.0f}MB | RAM: {final_peak_ram:.0f}MB | Time: {rjob.runtime:.1f}s")

    def _schedule_pending_jobs(self):
        if not self.pending: return
        hw = self.monitor.snapshot()
        ram_available = hw.free_ram_mb
        
        gpu_available = {g.gpu_id: g.free_vram_mb for g in hw.gpus}
        gpu_totals = {g.gpu_id: g.total_vram_mb for g in hw.gpus}
        active_jobs_per_gpu = {g.gpu_id: 0 for g in hw.gpus}
        
        for rjob in self.running.values():
            if rjob.gpu_id >= 0:
                active_jobs_per_gpu[rjob.gpu_id] += 1
                actual_vram_used = max(rjob.peak_real_vram_mb, rjob.job.required_vram_mb)
                gpu_available[rjob.gpu_id] -= actual_vram_used
            ram_available -= max(rjob.peak_real_ram_mb, rjob.job.required_ram_mb)

        if ram_available < 8192:
            return

        jobs_to_start = []
        now = time.time()
        self.pending.sort(
            key=lambda j: (
                j.priority, 
                -j.estimated_runtime, 
                -j.required_vram_mb
            ), 
            reverse=True
        )
        
        for job in list(self.pending):
            if len(self.running) + len(jobs_to_start) >= self.max_concurrent:
                break
            if time.time() < job.next_allowed_time:
                continue
            if ram_available < job.required_ram_mb:
                continue
                
            assigned_gpu = None
            best_score = -9999
            
            for gid in gpu_available.keys():
                fragmented_vram = gpu_available[gid] * 0.85 
                if now < self.gpu_locked_until.get(gid, 0.0):
                    continue

                if fragmented_vram >= job.required_vram_mb:
                    free_ratio = gpu_available[gid] / max(gpu_totals[gid], 1)
                    idle_ratio = 1.0 - (self.gpu_util_ema[gid] / 100.0)
                    inverse_running = 1.0 / (active_jobs_per_gpu[gid] + 1)

                    score = (job.priority * 0.1) + \
                            (0.4 * free_ratio) + \
                            (0.3 * idle_ratio) + \
                            (0.2 * inverse_running)
                    
                    if score > best_score:
                        best_score = score
                        assigned_gpu = gid

            if assigned_gpu is not None:
                jobs_to_start.append((job, assigned_gpu))
                self.pending.remove(job)
                gpu_available[assigned_gpu] -= job.required_vram_mb
                ram_available -= job.required_ram_mb
                active_jobs_per_gpu[assigned_gpu] += 1
                self.gpu_locked_until[assigned_gpu] = now + 8.0

        for job, gpu_id in jobs_to_start:
            q = self.manager.Queue()
            p = mp.Process(target=isolated_worker, args=(job, gpu_id, q))
            p.start()
            
            self.running[job.job_id] = RunningJob(
                job=job, process=p, result_queue=q, gpu_id=gpu_id, pid=p.pid, start_time=time.time()
            )
            print(f"[LAUNCHING] {job.name} -> GPU {gpu_id} | Req VRAM: {job.required_vram_mb:.0f}MB")