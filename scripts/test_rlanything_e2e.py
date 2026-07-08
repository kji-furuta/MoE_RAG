#!/usr/bin/env python3
"""
RLAnything Orchestrator End-to-End Closed-Loop Test

GPT-2 (124M) + RARdata.json で閉ループ強化学習を2イテレーション実行し、
全コンポーネント（Policy, Reward, Environment, Buffer, Orchestrator）の
統合動作を検証する。
"""

import gc
import json
import os
import time

import torch

gc.collect()
torch.cuda.empty_cache()

from src.training.rl import RLAnythingConfig, RLAnythingOrchestrator

SEP = "=" * 60


def main():
    print(SEP)
    print("RLAnything Orchestrator Closed-Loop Test")
    print(SEP)

    # ── Config ──────────────────────────────────────
    config = RLAnythingConfig(
        model_name="gpt2",
        output_dir="/tmp/rlanything_orchestrator_test",
        use_lora=False,
        use_quantization=False,
        bf16=True,
        grpo_num_generations=2,
        grpo_temperature=1.0,
        grpo_max_new_tokens=32,
        grpo_beta=0.04,
        policy_learning_rate=5e-5,
        policy_per_device_batch_size=2,
        policy_gradient_accumulation_steps=1,
        policy_max_steps=2,
        max_prompt_length=64,
        max_completion_length=32,
        reward_outcome_weight=0.6,
        reward_process_weight=0.4,
        reward_self_consistency_k=2,
        env_success_rate_low=0.2,
        env_success_rate_high=0.8,
        env_max_difficulty=3,
        env_enable_critical_feedback=False,
        num_iterations=2,
        trajectories_per_iteration=2,
        reward_update_interval=1,
        environment_adapt_interval=1,
        early_stopping_patience=5,
        convergence_threshold=0.001,
        gradient_checkpointing=False,
        logging_steps=1,
        save_steps=999,
        report_to="none",
    )
    print("[1/5] Config created: model=%s, iters=%d" % (config.model_name, config.num_iterations))

    # ── Orchestrator ─────────────────────────────────
    print("\n[2/5] Creating Orchestrator...")
    orchestrator = RLAnythingOrchestrator(config)

    # ── Load RARdata tasks ───────────────────────────
    print("\n[3/5] Loading tasks from RARdata.json...")
    with open("/workspace/RARdata.json", "r", encoding="utf-8") as f:
        rar_data = json.load(f)

    tasks_by_difficulty = {
        1: [r["instruction"] for r in rar_data[:5]],
        2: [r["instruction"] for r in rar_data[5:10]],
        3: [r["instruction"] for r in rar_data[10:15]],
    }
    for k, v in tasks_by_difficulty.items():
        print("  Difficulty %d: %d tasks" % (k, len(v)))

    # ── Setup with external model ────────────────────
    print("\n[4/5] Setup (loading gpt2 on cuda:1)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", dtype=torch.bfloat16, device_map="cuda:1"
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    orchestrator.setup(model=model, tokenizer=tokenizer)
    orchestrator.environment.register_all_tasks(tasks_by_difficulty)
    env_stats = orchestrator.environment.get_stats()
    print("  Task pools: %s" % str(env_stats["task_pools"]))
    print("  Setup complete")

    # ── Callbacks ────────────────────────────────────
    iteration_metrics = []

    def on_iter(iteration, metrics):
        iteration_metrics.append(metrics)
        mr = metrics.get("mean_reward", 0)
        sr = metrics.get("success_rate", 0)
        d = metrics.get("difficulty", 1)
        print("  >> Iter %d: reward=%.4f, success=%.4f, difficulty=%d" % (iteration, mr, sr, d))

    orchestrator.set_callbacks(
        on_iteration_complete=on_iter,
        status_callback=lambda msg: print("  Status: %s" % msg),
    )

    # ── Run ──────────────────────────────────────────
    print("\n[5/5] Running closed-loop training...")
    t0 = time.time()
    result = orchestrator.run()
    elapsed = time.time() - t0

    # ── Report ───────────────────────────────────────
    print("\n" + SEP)
    print("RESULTS:")
    print("  Status: %s" % result["status"])
    print("  Iterations: %d/%d" % (result["num_iterations_completed"], result["num_iterations_planned"]))
    print("  Final mean reward: %.4f" % result["final_mean_reward"])
    print("  Final success rate: %.4f" % result["final_success_rate"])
    print("  Best mean reward: %.4f" % result["best_mean_reward"])
    print("  Elapsed: %.1fs" % elapsed)

    buf = result["buffer_stats"]
    print("  Buffer: size=%d, mean_reward=%.4f" % (buf["size"], buf["mean_reward"]))

    env = result["environment_stats"]
    print("  Environment: difficulty=%d, pools=%s" % (env["current_difficulty"], str(env["task_pools"])))

    # Check saved results
    results_dir = os.path.join(config.output_dir, "results")
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        print("  Results files: %s" % str(files))

    # Per-iteration metrics
    print("\nIteration History:")
    for m in iteration_metrics:
        it = m.get("iteration", "?")
        mr = m.get("mean_reward", 0)
        sr = m.get("success_rate", 0)
        d = m.get("difficulty", 1)
        t_sec = m.get("elapsed_seconds", 0)
        print("  iter=%s: reward=%.4f, success=%.4f, difficulty=%d, time=%.1fs" % (it, mr, sr, d, t_sec))

    # ── Cleanup ─────────────────────────────────────
    del orchestrator, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + SEP)
    print("ORCHESTRATOR CLOSED-LOOP TEST COMPLETE")
    print(SEP)

    return result


if __name__ == "__main__":
    main()
