import subprocess
import psutil
from dataclasses import dataclass, field
from typing import List

@dataclass
class GPUHardwareInfo:
    gpu_id: int
    free_vram_mb: float
    total_vram_mb: float
    gpu_util_percent: float

    @property
    def used_vram_mb(self) -> float:
        return self.total_vram_mb - self.free_vram_mb

    @property
    def free_ratio(self) -> float:
        return self.free_vram_mb / max(self.total_vram_mb, 1)

    @property
    def used_ratio(self) -> float:
        return self.used_vram_mb / max(self.total_vram_mb, 1)

    def __str__(self):
        return (f"GPU {self.gpu_id} | "
                f"Free: {self.free_vram_mb:.0f}/{self.total_vram_mb:.0f} MB | "
                f"Util: {self.gpu_util_percent:.1f}%")

@dataclass
class HardwareSnapshot:
    free_ram_mb: float
    total_ram_mb: float
    cpu_percent: float
    cpu_count: int
    gpus: List[GPUHardwareInfo] = field(default_factory=list)

    @property
    def gpu_count(self):
        return len(self.gpus)
    
    @property
    def ram_used_mb(self):
        return self.total_ram_mb - self.free_ram_mb

class ResourceMonitor:
    @staticmethod
    def get_ram():
        mem = psutil.virtual_memory()
        return mem.available / 1024 / 1024, mem.total / 1024 / 1024

    @staticmethod
    def get_cpu():
        return psutil.cpu_percent(interval=0.2), psutil.cpu_count()

    @staticmethod
    def get_gpus():
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"
        ]
        try:
            output = subprocess.check_output(cmd, text=True).strip()
        except Exception:
            return []

        gpus = []
        for line in output.splitlines():
            idx, free, total, util = line.split(",")
            gpus.append(GPUHardwareInfo(
                gpu_id=int(idx),
                free_vram_mb=float(free),
                total_vram_mb=float(total),
                gpu_util_percent=float(util)
            ))
        return gpus

    def snapshot(self):
        free_ram, total_ram = self.get_ram()
        cpu_percent, cpu_count = self.get_cpu()
        gpus = self.get_gpus()
        return HardwareSnapshot(
            free_ram_mb=free_ram,
            total_ram_mb=total_ram,
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            gpus=gpus
        )