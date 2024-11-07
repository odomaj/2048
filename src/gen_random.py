from agents.random_agent import RandomAgent
from argparse import ArgumentParser
from multiprocessing import Pool
import numpy as np


def sample(count: int) -> np.ndarray[np.float64]:
    results: np.ndarray[np.float64] = np.zeros(count)
    for i in range(count):
        agent: RandomAgent = RandomAgent()
        agent.run_game()
        results[i] = np.float64(agent.score)
    return results


def sample_sizes(thread_count: int, total_sample_size: int) -> list[int]:
    sample_size = total_sample_size // thread_count
    if sample_size * thread_count != total_sample_size:
        sample_size += 1
    result: list[int] = [sample_size] * (thread_count - 1)
    result.append(total_sample_size - (sample_size * (thread_count - 1)))
    return result


def run_samples(thread_count: int, sample_size: int) -> np.ndarray[np.float64]:
    with Pool(thread_count) as pool:
        results: np.ndarray[np.float64] = np.concatenate(
            pool.map(sample, sample_sizes(thread_count, sample_size))
        )
    return results


def interpret_results(results: np.ndarray[np.float64]) -> str:
    output = (
        f"**********RESULTS**********\n * sample size = {len(results)}\n *"
        f" mean = {results.mean()}\n * variance = {np.var(results)}\n * min ="
        f" {results.min()}\n * 25th percentile ="
        f" {np.percentile(results, 25)}\n * median"
        f" {np.percentile(results, 50)}\n * 75th percentile ="
        f" {np.percentile(results, 75)}\n * max ="
        f" {results.max()}\n***************************"
    )
    return output


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--thread_count",
        "-t",
        default="1",
        help="threads allocated to generate random games",
    )
    arg_parser.add_argument(
        "--sample_size",
        "-s",
        default="1",
        help="number of random games to be run",
    )
    args = arg_parser.parse_args()
    print(
        interpret_results(
            run_samples(int(args.thread_count), int(args.sample_size))
        )
    )
