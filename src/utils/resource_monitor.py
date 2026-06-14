import subprocess
import threading
import time


class ResourceMonitor:
    def __init__(self, interval=5):
        self.interval = interval
        self._samples = []
        self._num_gpus = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._start_time = None

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip().splitlines()
                self._num_gpus = len(out)
                self._samples.append(sum(float(x) for x in out))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._start_time = time.time()
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        elapsed_h = (time.time() - self._start_time) / 3600
        avg_w = sum(self._samples) / len(self._samples) if self._samples else 0
        return {
            "elapsed_h": round(elapsed_h, 4),
            "avg_gpu_w": round(avg_w, 1),
            "gpu_hours": round(elapsed_h * max(self._num_gpus, 1), 4),
        }
