import argparse
import asyncio
import time
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def clean(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_prompt(row: dict) -> str:
    instruction = clean(row.get("instruction") or row.get("prompt") or row.get("question"))
    context = clean(row.get("context") or row.get("input"))

    if context:
        return (
            f"Instruction:\n{instruction}\n\n"
            f"Context:\n{context}"
        )

    return f"{instruction}"


def load_prompts(dataset_name: str, n_prompts: int, seed: int) -> list[str]:
    ds = load_dataset(dataset_name, split="train")
    ds = ds.shuffle(seed=seed)

    if n_prompts < len(ds):
        ds = ds.select(range(n_prompts))

    prompts = []

    for row in ds:
        prompt = build_prompt(row)
        if len(prompt) > 20:
            prompts.append(prompt)

    if not prompts:
        raise RuntimeError("No usable prompts found in the dataset.")

    return prompts


async def call_ollama(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    temperature: float,
    num_predict: int | None,
    num_ctx: int,
    seed: int,
    concurrent_users: int,
    user_id: int,
    request_id: int,
    prompt_index: int,
) -> dict:
    options = {
        "temperature": temperature,
        "num_ctx": num_ctx,
        "seed": seed,
    }


    if num_predict is not None:
        options["num_predict"] = num_predict

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": options,
    }

    started = time.perf_counter()

    try:
        response = await client.post(url, json=payload)
        wall_time_s = time.perf_counter() - started

        response.raise_for_status()
        data = response.json()

        output_tokens = data.get("eval_count", 0)
        generation_duration_s = data.get("eval_duration", 0) / 1e9

        tokens_per_second = (
            output_tokens / generation_duration_s
            if output_tokens and generation_duration_s
            else np.nan
        )

        generation_time_per_token_s = (
            generation_duration_s / output_tokens
            if output_tokens and generation_duration_s
            else np.nan
        )

        wall_time_per_token_s = (
            wall_time_s / output_tokens
            if output_tokens and wall_time_s
            else np.nan
        )

        return {
            "ok": True,
            "error": "",
            "model": model,
            "temperature": temperature,
            "concurrent_users": concurrent_users,
            "user_id": user_id,
            "request_id": request_id,
            "prompt_index": prompt_index,
            "seed": seed,


            "wall_time_s": wall_time_s,
            "ollama_total_duration_s": data.get("total_duration", np.nan) / 1e9,
            "load_duration_s": data.get("load_duration", np.nan) / 1e9,


            "prompt_eval_count": data.get("prompt_eval_count", np.nan),
            "prompt_eval_duration_s": data.get("prompt_eval_duration", np.nan) / 1e9,


            "output_tokens": output_tokens,
            "generation_duration_s": generation_duration_s,
            "generation_time_per_token_s": generation_time_per_token_s,
            "wall_time_per_token_s": wall_time_per_token_s,
            "tokens_per_second": tokens_per_second,


            "response_chars": len(data.get("response", "")),
        }

    except Exception as exc:
        wall_time_s = time.perf_counter() - started

        return {
            "ok": False,
            "error": repr(exc),
            "model": model,
            "temperature": temperature,
            "concurrent_users": concurrent_users,
            "user_id": user_id,
            "request_id": request_id,
            "prompt_index": prompt_index,
            "seed": seed,

            "wall_time_s": wall_time_s,
            "ollama_total_duration_s": np.nan,
            "load_duration_s": np.nan,

            "prompt_eval_count": np.nan,
            "prompt_eval_duration_s": np.nan,

            "output_tokens": np.nan,
            "generation_duration_s": np.nan,
            "generation_time_per_token_s": np.nan,
            "wall_time_per_token_s": np.nan,
            "tokens_per_second": np.nan,

            "response_chars": np.nan,
        }


async def run_condition(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompts: list[str],
    temperature: float,
    concurrent_users: int,
    requests_per_user: int,
    num_predict: int | None,
    num_ctx: int,
    seed: int,
    think_time_s: float,
) -> list[dict]:

    condition_started = time.perf_counter()
    results = []

    for request_id in range(requests_per_user):
        prompt_index = request_id % len(prompts)
        prompt = prompts[prompt_index]

        # I fixed the seed for all users in the same round, this allows to isolate cuncurrency effect
        request_seed = seed + request_id

        round_started = time.perf_counter()

        tasks = [
            call_ollama(
                client=client,
                url=url,
                model=model,
                prompt=prompt,
                temperature=temperature,
                num_predict=num_predict,
                num_ctx=num_ctx,
                seed=request_seed,
                concurrent_users=concurrent_users,
                user_id=user_id,
                request_id=request_id,
                prompt_index=prompt_index,
            )
            for user_id in range(concurrent_users)
        ]

        round_results = await asyncio.gather(*tasks)
        round_elapsed_s = time.perf_counter() - round_started

        for row in round_results:
            row["round_elapsed_s"] = round_elapsed_s
            results.append(row)

        if think_time_s > 0 and request_id < requests_per_user - 1:
            await asyncio.sleep(think_time_s)

    condition_elapsed_s = time.perf_counter() - condition_started

    for row in results:
        row["condition_elapsed_s"] = condition_elapsed_s

    return results


def nan_percentile(series: pd.Series, q: float) -> float:
    values = series.dropna().to_numpy()

    if len(values) == 0:
        return np.nan

    return float(np.nanpercentile(values, q))


def p50(series: pd.Series) -> float:
    return nan_percentile(series, 50)


def p95(series: pd.Series) -> float:
    return nan_percentile(series, 95)


def add_baseline_speedups(summary: pd.DataFrame) -> pd.DataFrame:
    baseline = summary[summary["concurrent_users"] == 1][
        [
            "temperature",
            "p50_wall_time_s",
            "response_throughput_rps",
            "token_throughput_end_to_end_tps",
        ]
    ].rename(
        columns={
            "p50_wall_time_s": "baseline_p50_wall_time_s",
            "response_throughput_rps": "baseline_response_throughput_rps",
            "token_throughput_end_to_end_tps": "baseline_token_throughput_end_to_end_tps",
        }
    )

    merged = summary.merge(baseline, on="temperature", how="left")

    merged["latency_slowdown_vs_1_user"] = (
        merged["p50_wall_time_s"] / merged["baseline_p50_wall_time_s"]
    )

    merged["response_throughput_speedup_vs_1_user"] = (
        merged["response_throughput_rps"] / merged["baseline_response_throughput_rps"]
    )

    merged["token_throughput_speedup_vs_1_user"] = (
        merged["token_throughput_end_to_end_tps"]
        / merged["baseline_token_throughput_end_to_end_tps"]
    )

    return merged.drop(
        columns=[
            "baseline_p50_wall_time_s",
            "baseline_response_throughput_rps",
            "baseline_token_throughput_end_to_end_tps",
        ]
    )


def make_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_df.groupby(["temperature", "concurrent_users"], as_index=False)

    summary = grouped.agg(
        requests=("ok", "size"),
        successful=("ok", "sum"),


        condition_elapsed_s=("condition_elapsed_s", "max"),


        mean_wall_time_s=("wall_time_s", "mean"),
        p50_wall_time_s=("wall_time_s", p50),
        p95_wall_time_s=("wall_time_s", p95),

        mean_ollama_total_duration_s=("ollama_total_duration_s", "mean"),


        mean_prompt_tokens=("prompt_eval_count", "mean"),
        mean_prompt_eval_duration_s=("prompt_eval_duration_s", "mean"),


        total_output_tokens=("output_tokens", "sum"),
        mean_output_tokens=("output_tokens", "mean"),
        p50_output_tokens=("output_tokens", p50),
        p95_output_tokens=("output_tokens", p95),


        mean_generation_duration_s=("generation_duration_s", "mean"),
        p50_generation_duration_s=("generation_duration_s", p50),
        p95_generation_duration_s=("generation_duration_s", p95),


        mean_generation_time_per_token_s=("generation_time_per_token_s", "mean"),
        p50_generation_time_per_token_s=("generation_time_per_token_s", p50),
        p95_generation_time_per_token_s=("generation_time_per_token_s", p95),

        mean_wall_time_per_token_s=("wall_time_per_token_s", "mean"),
        p50_wall_time_per_token_s=("wall_time_per_token_s", p50),
        p95_wall_time_per_token_s=("wall_time_per_token_s", p95),


        mean_tokens_per_second=("tokens_per_second", "mean"),
        p50_tokens_per_second=("tokens_per_second", p50),
        p95_tokens_per_second=("tokens_per_second", p95),

        mean_response_chars=("response_chars", "mean"),
    )

    summary["errors"] = summary["requests"] - summary["successful"]
    summary["error_rate"] = summary["errors"] / summary["requests"]


    summary["response_throughput_rps"] = (
        summary["successful"] / summary["condition_elapsed_s"]
    )

    summary["token_throughput_end_to_end_tps"] = (
        summary["total_output_tokens"] / summary["condition_elapsed_s"]
    )

    summary["per_user_response_throughput_rps"] = (
        summary["response_throughput_rps"] / summary["concurrent_users"]
    )

    summary["per_user_token_throughput_end_to_end_tps"] = (
        summary["token_throughput_end_to_end_tps"] / summary["concurrent_users"]
    )

    summary = summary.sort_values(["temperature", "concurrent_users"])
    summary = add_baseline_speedups(summary)

    return summary.sort_values(["temperature", "concurrent_users"])


#this part is needed only for the plots i wil ladd to the pdf document
def make_plots(summary: pd.DataFrame, outdir: Path) -> None:
    plt.figure()
    for temperature, sub in summary.groupby("temperature"):
        sub = sub.sort_values("concurrent_users")
        plt.plot(
            sub["concurrent_users"],
            sub["p50_wall_time_s"],
            marker="o",
            label=f"temp={temperature}",
        )

    plt.xlabel("Concurrent users")
    plt.ylabel("P50 total response time, seconds")
    plt.title("Total response time vs concurrent users")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "p50_total_response_time_vs_users.png", dpi=160)
    plt.close()

    plt.figure()
    for users, sub in summary.groupby("concurrent_users"):
        sub = sub.sort_values("temperature")
        plt.plot(
            sub["temperature"],
            sub["p50_wall_time_s"],
            marker="o",
            label=f"users={users}",
        )

    plt.xlabel("Temperature")
    plt.ylabel("P50 total response time, seconds")
    plt.title("Total response time vs temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "p50_total_response_time_vs_temperature.png", dpi=160)
    plt.close()

    plt.figure()
    for temperature, sub in summary.groupby("temperature"):
        sub = sub.sort_values("concurrent_users")
        plt.plot(
            sub["concurrent_users"],
            sub["response_throughput_rps"],
            marker="o",
            label=f"temp={temperature}",
        )

    plt.xlabel("Concurrent users")
    plt.ylabel("Responses per second")
    plt.title("Response throughput vs concurrent users")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "response_throughput_vs_users.png", dpi=160)
    plt.close()

    plt.figure()
    for temperature, sub in summary.groupby("temperature"):
        sub = sub.sort_values("concurrent_users")
        plt.plot(
            sub["concurrent_users"],
            sub["token_throughput_end_to_end_tps"],
            marker="o",
            label=f"temp={temperature}",
        )

    plt.xlabel("Concurrent users")
    plt.ylabel("Generated tokens per second, end-to-end")
    plt.title("End-to-end token throughput vs concurrent users")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "token_throughput_end_to_end_vs_users.png", dpi=160)
    plt.close()

    plt.figure()
    for temperature, sub in summary.groupby("temperature"):
        sub = sub.sort_values("concurrent_users")
        plt.plot(
            sub["concurrent_users"],
            sub["response_throughput_speedup_vs_1_user"],
            marker="o",
            label=f"temp={temperature}",
        )

    plt.xlabel("Concurrent users")
    plt.ylabel("Response throughput speedup vs 1 user")
    plt.title("Response throughput scaling")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "response_throughput_speedup_vs_users.png", dpi=160)
    plt.close()

    plt.figure()
    for temperature, sub in summary.groupby("temperature"):
        sub = sub.sort_values("concurrent_users")
        plt.plot(
            sub["concurrent_users"],
            sub["latency_slowdown_vs_1_user"],
            marker="o",
            label=f"temp={temperature}",
        )

    plt.xlabel("Concurrent users")
    plt.ylabel("Latency slowdown vs 1 user")
    plt.title("Latency slowdown as users increase")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "latency_slowdown_vs_users.png", dpi=160)
    plt.close()

    plt.figure()
    for temperature, sub in summary.groupby("temperature"):
        sub = sub.sort_values("concurrent_users")
        plt.plot(
            sub["concurrent_users"],
            sub["p50_generation_time_per_token_s"],
            marker="o",
            label=f"temp={temperature}",
        )

    plt.xlabel("Concurrent users")
    plt.ylabel("P50 generation time per token, seconds")
    plt.title("Generation time per token vs concurrent users")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "p50_generation_time_per_token_vs_users.png", dpi=160)
    plt.close()

    plt.figure()
    for users, sub in summary.groupby("concurrent_users"):
        sub = sub.sort_values("temperature")
        plt.plot(
            sub["temperature"],
            sub["p50_generation_time_per_token_s"],
            marker="o",
            label=f"users={users}",
        )

    plt.xlabel("Temperature")
    plt.ylabel("P50 generation time per token, seconds")
    plt.title("Generation time per token vs temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "p50_generation_time_per_token_vs_temperature.png", dpi=160)
    plt.close()

    plt.figure()
    for users, sub in summary.groupby("concurrent_users"):
        sub = sub.sort_values("temperature")
        plt.plot(
            sub["temperature"],
            sub["p50_output_tokens"],
            marker="o",
            label=f"users={users}",
        )

    plt.xlabel("Temperature")
    plt.ylabel("P50 output tokens")
    plt.title("Output length vs temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "p50_output_tokens_vs_temperature.png", dpi=160)
    plt.close()

    plt.figure()
    for users, sub in summary.groupby("concurrent_users"):
        sub = sub.sort_values("temperature")
        plt.plot(
            sub["temperature"],
            sub["p50_tokens_per_second"],
            marker="o",
            label=f"users={users}",
        )

    plt.xlabel("Temperature")
    plt.ylabel("P50 generation tokens per second")
    plt.title("Per-request generation throughput vs temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "p50_tokens_per_second_vs_temperature.png", dpi=160)
    plt.close()


async def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(
        dataset_name=args.dataset,
        n_prompts=args.n_prompts,
        seed=args.seed,
    )

    url = args.host.rstrip("/") + "/api/generate"
    timeout = httpx.Timeout(args.timeout_s, connect=10.0)

    raw_rows = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        print("Warming up Ollama model...")
        #start ollama and makke a simple request as a warm up, in this way we are sure the OLLAMA server is working correctly
        warmup = await call_ollama(
            client=client,
            url=url,
            model=args.model,
            prompt="Say only: OK",
            temperature=0.0,
            num_predict=8,
            num_ctx=args.num_ctx,
            seed=args.seed,
            concurrent_users=1,
            user_id=0,
            request_id=-1,
            prompt_index=-1,
        )

        if not warmup["ok"]:
            raise RuntimeError(
                "Warmup failed. Check that Ollama is running and that the model "
                f"has been pulled. Error: {warmup['error']}"
            )

        temperatures = parse_float_list(args.temps)
        user_counts = parse_int_list(args.users)

        for temperature in temperatures:
            for concurrent_users in user_counts:
                print()
                print(
                    f"Running condition: "
                    f"temperature={temperature}, "
                    f"concurrent_users={concurrent_users}"
                )

                condition_rows = await run_condition(
                    client=client,
                    url=url,
                    model=args.model,
                    prompts=prompts,
                    temperature=temperature,
                    concurrent_users=concurrent_users,
                    requests_per_user=args.requests_per_user,
                    num_predict=args.num_predict,
                    num_ctx=args.num_ctx,
                    seed=args.seed,
                    think_time_s=args.think_time_s,
                )

                raw_rows.extend(condition_rows)

                raw_df = pd.DataFrame(raw_rows)
                raw_df.to_csv(outdir / "ollama_benchmark_raw.csv", index=False)

                summary = make_summary(raw_df)
                summary.to_csv(outdir / "ollama_benchmark_summary.csv", index=False)

                latest = summary[
                    (summary["temperature"] == temperature)
                    & (summary["concurrent_users"] == concurrent_users)
                ]

                columns_to_print = [
                    "temperature",
                    "concurrent_users",
                    "requests",
                    "successful",
                    "condition_elapsed_s",
                    "p50_wall_time_s",
                    "p95_wall_time_s",
                    "response_throughput_rps",
                    "token_throughput_end_to_end_tps",
                    "latency_slowdown_vs_1_user",
                    "response_throughput_speedup_vs_1_user",
                    "mean_output_tokens",
                    "p50_generation_time_per_token_s",
                ]

                print(latest[columns_to_print].to_string(index=False))

    final_raw = pd.DataFrame(raw_rows)
    final_summary = make_summary(final_raw)

    final_raw.to_csv(outdir / "ollama_benchmark_raw.csv", index=False)
    final_summary.to_csv(outdir / "ollama_benchmark_summary.csv", index=False)

    make_plots(final_summary, outdir)

    print()
    print("Done.")
    print(f"Raw results:     {outdir / 'ollama_benchmark_raw.csv'}")
    print(f"Summary results: {outdir / 'ollama_benchmark_summary.csv'}")
    print()
    print("Main throughput columns:")
    print("- response_throughput_rps")
    print("- token_throughput_end_to_end_tps")
    print("- response_throughput_speedup_vs_1_user")
    print("- token_throughput_speedup_vs_1_user")
    print("- latency_slowdown_vs_1_user")
    print()
    print("Plots:")
    print(f"- {outdir / 'p50_total_response_time_vs_users.png'}")
    print(f"- {outdir / 'p50_total_response_time_vs_temperature.png'}")
    print(f"- {outdir / 'response_throughput_vs_users.png'}")
    print(f"- {outdir / 'token_throughput_end_to_end_vs_users.png'}")
    print(f"- {outdir / 'response_throughput_speedup_vs_users.png'}")
    print(f"- {outdir / 'latency_slowdown_vs_users.png'}")
    print(f"- {outdir / 'p50_generation_time_per_token_vs_users.png'}")
    print(f"- {outdir / 'p50_generation_time_per_token_vs_temperature.png'}")
    print(f"- {outdir / 'p50_output_tokens_vs_temperature.png'}")
    print(f"- {outdir / 'p50_tokens_per_second_vs_temperature.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--dataset", default="databricks/databricks-dolly-15k")

    parser.add_argument("--temps", default="0,0.2,0.7,1.0,1.3")
    parser.add_argument("--users", default="1,2,4,8")

    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--requests-per-user", type=int, default=10)

    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help=(
            "Optional max output token cap. "
            "Leave unset to let output length vary naturally."
        ),
    )

    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--think-time-s", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--outdir", default="ollama_benchmark_results")

    args = parser.parse_args()
    asyncio.run(main(args))
