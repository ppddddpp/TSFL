import json
import os

class HistoryManager:
    def __init__(self, history_file: str = "job_history.json"):
        self.history_file = history_file
        self.history = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except Exception: pass
        return {}

    def _save(self):
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=4)
        except Exception: pass

    def update(self, job_name: str, runtime: float, peak_vram_mb: float, peak_ram_mb: float):
        if job_name not in self.history:
            self.history[job_name] = {
                "est_runtime": runtime,
                "peak_vram_mb": peak_vram_mb,
                "peak_ram_mb": peak_ram_mb
            }
        else:
            old_rt = self.history[job_name]["est_runtime"]
            self.history[job_name]["est_runtime"] = (0.2 * runtime) + (0.8 * old_rt)
            self.history[job_name]["peak_vram_mb"] = max(peak_vram_mb, self.history[job_name]["peak_vram_mb"] * 0.99)
            self.history[job_name]["peak_ram_mb"] = max(peak_ram_mb, self.history[job_name]["peak_ram_mb"] * 0.99)
            
        self._save()

    def get_estimate(self, job_name: str, default_vram: float = 3000.0, default_ram: float = 4096.0):
        if job_name in self.history:
            data = self.history[job_name]
            base_vram = data.get("peak_vram_mb", 0.0)
            base_ram = data.get("peak_ram_mb", 0.0)
            
            safe_vram = base_vram + min(base_vram * 0.05, 1024.0) if base_vram > 100 else default_vram
            safe_ram = base_ram + min(base_ram * 0.05, 1024.0) if base_ram > 100 else default_ram
            
            return data.get("est_runtime", 0.0), safe_vram, safe_ram
            
        return 0.0, default_vram, default_ram