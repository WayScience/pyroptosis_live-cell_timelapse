"""
This collection of functions runs CellProfiler in parallel and can convert the results into log files
for each process.

Developed from NF1 repository script made by Jenna Tomkinson.
Slightly modified here to fit this project need.
"""

import multiprocessing
import pathlib
import subprocess
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import datetime
from typing import List, Optional, Union


def results_to_log(
    results: List[subprocess.CompletedProcess], log_dir: pathlib.Path, run_name: str
) -> None:
    """
    This function will take the list of subprocess.results from a CellProfiler parallelization run and
    convert into a log file for each process.

    Args:
        results (List[subprocess.CompletedProcess]):
            the outputs from a subprocess.run.
        log_dir (pathlib.Path):
            directory for log files.
        run_name (str):
            a given name for the type of CellProfiler run being done on the plates (example: whole image features).
    """
    for idx, result in enumerate(results):
        command_args = (
            result.args if isinstance(result.args, list) else [str(result.args)]
        )

        plate_name = "unknown_plate"
        if "-i" in command_args:
            image_idx = command_args.index("-i") + 1
            if image_idx < len(command_args):
                plate_name = pathlib.Path(command_args[image_idx]).name

        stdout_string = (
            result.stdout.decode("utf-8", errors="replace")
            if isinstance(result.stdout, (bytes, bytearray))
            else str(result.stdout)
        )
        stderr_string = (
            result.stderr.decode("utf-8", errors="replace")
            if isinstance(result.stderr, (bytes, bytearray))
            else str(result.stderr)
        )

        log_file_path = pathlib.Path(log_dir) / f"{plate_name}_{run_name}_{idx}_run.log"
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(
                f"[{datetime.now().isoformat()}] [Return Code: {result.returncode}]\n"
            )
            log_file.write(f"Command: {' '.join(map(str, command_args))}\n\n")
            log_file.write("--- STDOUT ---\n")
            log_file.write(stdout_string if stdout_string else "<empty>\n")
            log_file.write("\n--- STDERR ---\n")
            log_file.write(stderr_string if stderr_string else "<empty>\n")


def run_cellprofiler_parallel(
    plate_info_dictionary: dict,
    run_name: str,
    plugins_dir: Optional[Union[pathlib.Path, None]] = None,
    log_dir: Optional[pathlib.Path] = None,
    max_workers: Optional[Union[int, None]] = 4,
) -> None:
    """
    This function utilizes multi-processing to run CellProfiler pipelines in parallel.

    Args:
        plate_info_dictionary (dict):
            dictionary with all paths for CellProfiler to run a pipeline.
        run_name (str):
            a given name for the type of CellProfiler run being done on the plates (example: whole image features).
        plugins_dir (pathlib.Path, optional):
            if you are using a CellProfiler plugin module in your pipeline, you must specify a path to the directory.
            This is an optional parameter and defaults to None (no plugin dir provided).
        log_dir (pathlib.Path, optional):
            directory for log files.
        max_workers (int, optional):
            maximum number of worker processes to use for parallel execution.
            This is an optional parameter and defaults to 4.
    Raises:
        FileNotFoundError: if paths to pipeline and images do not exist
    """
    # create a list of commands for each plate with their respective log file
    commands = []

    # make logs directory
    if log_dir is None:
        log_dir = pathlib.Path("./logs")
        pathlib.Path(log_dir).mkdir(exist_ok=True)
    else:
        pathlib.Path(log_dir).mkdir(exist_ok=True)

    # iterate through each plate in the dictionary
    for _, info in plate_info_dictionary.items():
        # set paths for CellProfiler
        path_to_pipeline = info["path_to_pipeline"]
        path_to_images = info["path_to_images"]
        path_to_output = info["path_to_output"]

        # check to make sure paths to pipeline and directory of images are correct before running the pipeline
        if not pathlib.Path(path_to_pipeline).resolve(strict=True):
            raise FileNotFoundError(
                f"The file '{pathlib.Path(path_to_pipeline).name}' does not exist"
            )
        if not pathlib.Path(path_to_images).is_dir():
            raise FileNotFoundError(
                f"Directory '{pathlib.Path(path_to_images).name}' does not exist or is not a directory"
            )
        # make output directory if it is not already created
        pathlib.Path(path_to_output).mkdir(exist_ok=True)

        # creates a command for each plate in the list
        command = [
            "cellprofiler",
            "-c",
            "-r",
            "-p",
            path_to_pipeline,
            "-o",
            path_to_output,
            "-i",
            path_to_images,
        ]

        # if plugins_dir is provided, check to confirm dir exists and add the flag to find the plugins directory with given path
        if plugins_dir:
            if not pathlib.Path(plugins_dir).is_dir():
                raise FileNotFoundError(
                    f"Plugins directory '{pathlib.Path(plugins_dir).name}' does not exist or is not a directory"
                )
            else:
                command.extend(["--plugins-directory", plugins_dir])

        # creates a list of commands
        commands.append(command)

    if max_workers is None:
        # make sure that the number of workers does not exceed the maximum number of workers for the machine
        num_processes = max(1, multiprocessing.cpu_count() - 2)
    else:
        num_processes = max_workers

    # set parallelization executer to the number of commands
    executor = ProcessPoolExecutor(max_workers=num_processes)

    # creates a list of futures that are each CellProfiler process for each plate
    futures: List[Future] = [
        executor.submit(
            subprocess.run,
            args=command,
            capture_output=True,
        )
        for command in commands
    ]

    # the list of CompletedProcesses holds all the information from the CellProfiler run
    results: List[subprocess.CompletedProcess] = [future.result() for future in futures]

    print("All processes have been completed!")

    # convert results to log files
    results_to_log(results=results, log_dir=log_dir, run_name=run_name)

    # for each process, confirm that the process completed successfully
    for result in results:
        print(result.returncode)
        if result.returncode != 0:
            print(
                f"A return code of {result.returncode} was returned for {result}, which means there was an error in the CellProfiler run."
            )

    # to avoid having multiple print statements due to for loop, confirmation that logs are converted is printed here
    print("All results have been converted to log files!")
