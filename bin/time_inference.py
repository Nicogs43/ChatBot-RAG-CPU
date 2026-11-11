from config import INFERENCE_TIME_DIR, QUESTIONS_PATH, model_dict
import pandas as pd
from chatbot import RagEngine, load_retrieval_resources
import argparse
import gc
import logging
import time
import os
import psutil
import sys
import threading
from pathlib import Path

logging.basicConfig(level=logging.INFO)

MB = 1024 * 1024
process = psutil.Process(os.getpid())


def _to_mb(n_bytes):
    return round(n_bytes / MB, 2)


def _rss_mb():
    return _to_mb(process.memory_info().rss)


def _os_peak_mb():
    """Process-lifetime peak resident memory as tracked by the OS (exact, no sampling).
    Windows: peak working set. Linux/macOS: ru_maxrss. None if unavailable."""
    peak_wset = getattr(process.memory_info(), "peak_wset", None)
    if peak_wset is not None:
        return _to_mb(peak_wset)
    if sys.platform == "win32":  # `resource` is POSIX-only
        return None
    import resource
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS, kilobytes on Linux
    return _to_mb(max_rss if sys.platform == "darwin" else max_rss * 1024)


def _private_mb():
    """Private (non-shared, non-file-backed) memory committed by the process.
    Unlike RSS/working set it is not shrunk when the OS trims or pages out the
    process under memory pressure. Note: mmap'd model weights are file-backed and
    therefore not counted here; RSS is still reported alongside."""
    private = getattr(process.memory_info(), "private", None)  # Windows private bytes
    if private is not None:
        return _to_mb(private)
    try:
        return _to_mb(process.memory_full_info().uss)
    except (psutil.AccessDenied, AttributeError):
        return None


def get_memory_snapshot():
    virtual_memory = psutil.virtual_memory()
    return {
        "proc_rss_mb": _rss_mb(),
        "proc_private_mb": _private_mb(),
        "os_peak_mb": _os_peak_mb(),
        "sys_used_mb": _to_mb(virtual_memory.used),
        "sys_percent": round(virtual_memory.percent, 2),
    }


def diff_snapshots(before, after, prefix=""):
    def delta(key):
        if before[key] is None or after[key] is None:
            return None
        return round(after[key] - before[key], 2)

    return {
        f"{prefix}proc_rss_mb_before": before["proc_rss_mb"],
        f"{prefix}proc_rss_mb_after": after["proc_rss_mb"],
        f"{prefix}proc_rss_mb_delta": delta("proc_rss_mb"),
        f"{prefix}proc_private_mb_before": before["proc_private_mb"],
        f"{prefix}proc_private_mb_after": after["proc_private_mb"],
        f"{prefix}proc_private_mb_delta": delta("proc_private_mb"),
        f"{prefix}sys_percent_before": before["sys_percent"],
        f"{prefix}sys_percent_after": after["sys_percent"],
        f"{prefix}sys_used_mb_before": before["sys_used_mb"],
        f"{prefix}sys_used_mb_after": after["sys_used_mb"],
    }


def invoke_with_peak_rss(engine, question, sample_interval=0.05):
    before = get_memory_snapshot()
    sampled_peak_mb = before["proc_rss_mb"]
    stop_sampling = threading.Event()

    def sample_peak():
        nonlocal sampled_peak_mb
        while not stop_sampling.is_set():
            sampled_peak_mb = max(sampled_peak_mb, _rss_mb())
            stop_sampling.wait(sample_interval)

    sampler_thread = threading.Thread(target=sample_peak, daemon=True)
    sampler_thread.start()
    start_time_inference = time.perf_counter()
    try:
        engine.invoke(question)
    finally:
        end_time_inference = time.perf_counter()
        stop_sampling.set()
        sampler_thread.join()

    after = get_memory_snapshot()

    # The OS peak counter is exact but covers the whole process lifetime. If it rose
    # during this call, the new value is exactly this call's peak; otherwise this
    # call's peak is below the old lifetime peak and the sampled value is the best estimate.
    peak_mb = sampled_peak_mb
    peak_source = "sampled"
    if before["os_peak_mb"] is not None and after["os_peak_mb"] > before["os_peak_mb"]:
        peak_mb = max(peak_mb, after["os_peak_mb"])
        peak_source = "os"

    return {
        "inference_time": end_time_inference - start_time_inference,
        **diff_snapshots(before, after),
        "proc_rss_mb_peak_during_inference": peak_mb,
        "proc_rss_mb_peak_over_before": round(peak_mb - before["proc_rss_mb"], 2),
        "peak_source": peak_source,
    }


def save_init(time_init_dict, suffix, output_dir):
    if not time_init_dict:
        return
    save_time_init = pd.DataFrame.from_dict(time_init_dict, orient="index").reset_index()
    save_time_init = save_time_init.rename(columns={"index": "model"})
    save_time_init.to_csv(output_dir / f"time_initialization_{suffix}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Measure initialization time, inference time and RAM usage for each model in model_dict")
    parser.add_argument("--mode", choices=["first", "all"], default="first",
                        help="'first': one cold call on the first question (*_first_question.csv). "
                             "'all': every question in the dataset, i.e. the inference-steps series (*_new.csv).")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to wait between questions in 'all' mode")
    parser.add_argument("--models", nargs="*", choices=list(model_dict), default=list(model_dict),
                        help="Model keys to measure (default: all)")
    parser.add_argument("--output-dir", type=Path, default=INFERENCE_TIME_DIR,
                        help="Where to write the CSVs (default: data/inference_time/, which holds the thesis results)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "first_question" if args.mode == "first" else "new"

    try:
        data = pd.read_csv(QUESTIONS_PATH)
    except FileNotFoundError:
        raise ValueError("Error loading data")
    if data.empty:
        raise ValueError("Question dataset is empty")
    questions = data["Question"].tolist()
    if args.mode == "first":
        questions = questions[:1]

    vectorstore, reranker = load_retrieval_resources()

    time_init_dict = {}
    interrupted = False
    for model_name in args.models:
        logging.info(f"Testing model: {model_name}")
        engine = None
        time_inference_list = []
        try:
            init_ram_before = get_memory_snapshot()
            start_time_initialization = time.perf_counter()
            engine = RagEngine(model_name, vectorstore, reranker)
            init_time = time.perf_counter() - start_time_initialization
            init_ram_after = get_memory_snapshot()
            time_init_dict[model_name] = {
                "initialization_time": init_time,
                **diff_snapshots(init_ram_before, init_ram_after, prefix="init_"),
            }
            logging.info(f"Initialization time for model {model_name}: {init_time}")

            for i, question in enumerate(questions):
                time_inference_list.append(invoke_with_peak_rss(engine, question))
                if i < len(questions) - 1:
                    time.sleep(args.pause)
        except KeyboardInterrupt:
            logging.info(f"Interrupted during model {model_name}; saving collected results and stopping.")
            interrupted = True
        except Exception:
            logging.exception(f"Model {model_name} failed; continuing with the next model.")
        finally:
            # Drop the engine (it holds the model) and collect, so the next
            # model is measured without the previous one still resident.
            engine = None
            gc.collect()
            logging.info(f"Session ended for model {model_name}.")

        # Only write when there is something to write, so an interrupted/failed run
        # never overwrites previous results with an empty file.
        if time_inference_list:
            pd.DataFrame(time_inference_list).to_csv(args.output_dir / f"{model_name}_time_inference_{suffix}.csv", index=False)
            print(f"Data saved successfully for model: {model_name}")
        save_init(time_init_dict, suffix, args.output_dir)  # save after every model so earlier results survive a crash

        if interrupted:
            break

    del vectorstore
    del reranker
    logging.info("All models tested. Resources cleaned up.")


if __name__ == "__main__":
    main()
