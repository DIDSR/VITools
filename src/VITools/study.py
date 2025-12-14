"""Manages and executes virtual imaging studies.

This module provides the `Study` class, which is the primary tool for setting
up, managing, and running a series of virtual imaging trials. It supports
loading study parameters from CSV files, generating diverse scan protocols
from statistical distributions, and executing these simulations either serially
or in parallel on a high-performance computing cluster.

The module also includes a plugin system for discovering available phantom
types and a command-line interface for running studies.
"""
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import shutil
from time import sleep
import ast
from subprocess import run
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import pluggy

from .scanner import Scanner, load_vol
from .phantom import Phantom
from . import hooks

src_dir = Path(__file__).parent.absolute()


def get_available_phantoms() -> dict[str, type[Phantom]]:
    """Discovers and returns available phantom classes via a plugin system.

    This function uses the `pluggy` framework to find and load all installed
    plugins that have registered phantom types through the `register_phantom_types`
    hook specification.

    Returns:
        dict[str, type[Phantom]]: A dictionary mapping the unique string name of
            each discovered phantom to its corresponding class type.
    """
    pm = pluggy.PluginManager(hooks.PROJECT_NAME)
    pm.add_hookspecs(hooks.PhantomSpecs)
    pm.load_setuptools_entrypoints(group=hooks.PROJECT_NAME)

    # The hook returns a list of dictionaries (one from each plugin).
    list_of_results = pm.hook.register_phantom_types()
    discovered_phantom_classes = {}
    for result_dict in list_of_results:
        if result_dict:  # Check if the plugin returned a non-empty dict
            discovered_phantom_classes.update(result_dict)
    return discovered_phantom_classes


def scan_logs_for_errors(directory_path, verbose=True):
    """
    Scans a directory for log files (e.g., task_0.log to task_4999.log),
    prints which log files have a raised error, and copies the error
    message raised in that log file.

    Args:
        directory_path (str): The path to the directory containing the log files.
    """    
    # Regex to match the log file pattern
    log_file_pattern = re.compile(r"task_\d+\.log")
    errors = dict()
    try:
        # Get a list of all files and directories in the specified directory
        with os.scandir(directory_path) as entries:
            for entry in entries:
                # Check if it's a file and if it matches the log file pattern
                if entry.is_file() and log_file_pattern.match(entry.name):
                    log_file_path = entry.path
                    try:
                        with open(log_file_path, 'r') as f:
                            lines = f.readlines()
                        
                        error_lines = []
                        in_traceback = False
                        # Check each line for the start of a traceback
                        for i, line in enumerate(lines):
                            if ("Traceback (most recent call last):") in line or ("Killed" in line):
                                in_traceback = True
                                # Once traceback is found, the rest of the file is the error
                                error_lines = [l.strip() for l in lines[i:] if l.strip()]
                                break

                        if in_traceback and error_lines:
                            # The last line of the traceback is typically the error message
                            error_message = error_lines[-5]
                            errors[entry.name] = error_message
                            if verbose:
                                print(f"--- ERROR FOUND IN: {entry.name} ---")
                                print(f"Error Message: {error_message}\n")

                    except Exception as e:
                        print(f"Could not read file {entry.name}. Reason: {e}")

    except FileNotFoundError:
        print(f"Error: Directory not found at '{directory_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return errors


class Study:
    """Manages and executes a series of imaging simulations.

    This class handles the definition of study parameters from CSV files or
    DataFrames, generates scan parameters from distributions, and runs the
    simulations serially or in parallel. It also tracks and retrieves
    simulation results.

    Attributes:
        metadata (pd.DataFrame): A DataFrame holding the parameters for each
                                 scan in the study.
    """
    def __init__(self, input_csv: pd.DataFrame | str | Path | None = None):
        """Initializes a Study instance.

        Args:
            input_csv (pd.DataFrame | str | Path | None, optional):
                Path to a CSV file or a pandas DataFrame containing study
                metadata. If None, an empty study is initialized.
                Defaults to None.
        """
        self.metadata = pd.DataFrame()
        if input_csv is not None:
            self.load_study(input_csv)

    def __len__(self) -> int:
        """Returns the number of scans defined in the study metadata."""
        return len(self.metadata)

    def __repr__(self) -> str:
        """Returns a string representation of the Study object."""
        repr_str = f'''
Input metadata:\n
{self.metadata}

Results:\n
{self.results}
'''
        return repr_str

    def load_study(self, input_csv: str | Path | pd.DataFrame):
        """Loads study metadata from a CSV file or a pandas DataFrame.

        Args:
            input_csv (str | Path | pd.DataFrame): The path to a CSV file or a
                pandas DataFrame containing the study parameters. The loaded
                data replaces any existing metadata.
        """
        if isinstance(input_csv, pd.DataFrame):
            self.metadata = input_csv
            parent_path = Path(input_csv.output_directory.iloc[0]).parent.absolute()
            study_ids = '-'.join(list(map(lambda o: o.split('case_')[-1], [input_csv.case_id.iloc[0], input_csv.case_id.iloc[-1]])))
            input_csv = parent_path / f'study_plan_{study_ids}.csv'
            print(f'study plan saved to: {input_csv}')
            self.metadata.to_csv(input_csv, index=False)
        elif isinstance(input_csv, str | Path):
            self.metadata = pd.read_csv(input_csv)
        self.csv_fname = input_csv  

    def clear_previous_results(self):
        """Removes output directories associated with each scan in the study.

        This iterates through the 'output_directory' column in the metadata
        and deletes each directory if it exists, effectively cleaning up
        before re-running a study.
        """
        for idx in range(len(self.metadata)):
            output_dir = Path(self.metadata.iloc[idx]['output_directory'])
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    @staticmethod
    def generate_from_distributions(phantoms: list[str],
                                    study_count: int = 1,
                                    output_directory: str | Path = 'results',
                                    views: list[int] = [1000],
                                    scan_coverage: str | list | tuple = 'dynamic',
                                    scanner_model: list[str] = ['Scanner_Default'],
                                    kVp: list[int] = [120],
                                    mA: list[int] = [300],
                                    pitch: list[float] = [0],
                                    recon_kernel: list[str] = ['soft'],
                                    slice_thickness: list[int] = [1],
                                    slice_increment: list[int] = [1],
                                    fov: list[float] = [250],
                                    remove_raw: bool = True,
                                    seed: int | None = None) -> pd.DataFrame:
        """Generates study metadata by sampling parameters from distributions.

        For each of `study_count` cases, parameters like phantom type, scanner,
        kVp, and mA are chosen randomly from the provided lists, allowing for
        the creation of diverse datasets.

        Args:
            phantoms (list[str]): List of phantom names to choose from.
            study_count (int, optional): Number of scan configurations to
                generate. Defaults to 1.
            output_directory (str | Path, optional): Base directory for output.
                Defaults to 'results'.
            views (list[int], optional): List of view counts per rotation.
                Defaults to [1000].
            scan_coverage (str | list | tuple, optional): Scan coverage. Can be
                'dynamic', or a list/tuple of [start_z, end_z].
                Defaults to 'dynamic'.
            scanner_model (list[str], optional): List of scanner model names.
                Defaults to ['Scanner_Default'].
            kVp (list[int], optional): List of kVp values. Defaults to [120].
            mA (list[int], optional): List of mA values. Defaults to [300].
            pitch (list[float], optional): List of pitch values. Defaults to [0].
            recon_kernel (list[str], optional): List of recon kernel names.
                Defaults to ['soft'].
            slice_thickness (list[int], optional): List of slice thicknesses (mm).
                Defaults to [1].
            slice_increment (list[int], optional): List of slice increments (mm).
                Defaults to [1].
            fov (list[float], optional): List of Field of View values (mm).
                Defaults to [250].
            remove_raw (bool, optional): Whether to remove raw projection data
                after reconstruction. Defaults to True.
            seed (int | None, optional): Seed for the random number generator.
                If None, a random seed is used. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the generated study parameters.

        Raises:
            ValueError: If `seed` is invalid.
        """
        output_directory = Path(output_directory)
        assert (scan_coverage == 'dynamic') or isinstance(scan_coverage, (list, tuple))
        if isinstance(scan_coverage, list):
            if len(scan_coverage) < 2:
                scan_coverage = scan_coverage[0].split(' ')
        if isinstance(scan_coverage, list):
            scan_coverage = [float(o) for o in scan_coverage]

        if isinstance(seed, float) or seed is True:
            raise ValueError('seed must be an integer or None.')
        rng = np.random.default_rng(seed)

        params = {
            'case_id': [f'case_{i:04d}' for i in range(study_count)],
            'phantom': rng.choice(list(phantoms), study_count),
            'scanner_model': rng.choice(scanner_model, study_count),
            'kVp': rng.choice(kVp, study_count).astype(float),
            'mA': rng.choice(mA, study_count).astype(float),
            'pitch': rng.choice(pitch, study_count).astype(float),
            'views': rng.choice(views, study_count).astype(float),
            'scan_coverage': [scan_coverage] * study_count,
            'recon_kernel': rng.choice(recon_kernel, study_count),
            'slice_thickness': rng.choice(slice_thickness, study_count),
            'slice_increment': rng.choice(slice_increment, study_count),
            'fov': rng.choice(fov, study_count),
            'global_seed': [seed] * study_count,
            'case_seed': rng.integers(0, 1e6, study_count),
            'output_directory': [output_directory.absolute() / f'case_{i:04d}' for i in range(study_count)],
            'remove_raw': [remove_raw] * study_count
        }
        return pd.DataFrame(params)

    def append(self, phantom: str | pd.DataFrame,
               output_directory: str | Path = 'results',
               scanner_model: str = 'Scanner_Default',
               kVp: int = 120, mA: int = 200, pitch: float = 0,
               views: int = 1000, fov: float = 250,
               scan_coverage: tuple[float, float] | str = 'dynamic',
               recon_kernel: str = 'standard',
               slice_thickness: int | None = 1,
               slice_increment: int | None = None,
               seed: int | None = None,
               remove_raw: bool = True, **kwargs) -> "Study":
        """Appends one or more scans to the study's metadata.

        If `phantom` is a DataFrame, it's concatenated to the existing metadata.
        Otherwise, a new scan configuration is created from the provided
        parameters and added.

        Args:
            phantom (str | pd.DataFrame): Phantom name or a DataFrame of scans.
            output_directory (str | Path, optional): Base output directory.
                Defaults to 'results'.
            scanner_model (str, optional): Scanner model name. Defaults to
                'Scanner_Default'.
            kVp (int, optional): Kilovolt peak. Defaults to 120.
            mA (int, optional): Milliampere value. Defaults to 200.
            pitch (float, optional): Pitch value. Defaults to 0.
            views (int, optional): Number of views. Defaults to 1000.
            fov (float, optional): Field of View (mm). Defaults to 250.
            scan_coverage (tuple[float, float] | str, optional): Scan range
                (start_z, end_z) or 'dynamic'. Defaults to 'dynamic'.
            recon_kernel (str, optional): Recon kernel. Defaults to 'standard'.
            slice_thickness (int | None, optional): Slice thickness (mm).
                Defaults to 1.
            slice_increment (int | None, optional): Slice increment (mm).
                Defaults to `slice_thickness`.
            seed (int | None, optional): Random seed. Defaults to None.
            remove_raw (bool, optional): Remove raw data post-recon.
                Defaults to True.

        Returns:
            Study: The Study instance for method chaining.

        Raises:
            KeyError: If the `phantom` name is not an available type.
        """
        if isinstance(phantom, pd.DataFrame):
            self.metadata = pd.concat([self.metadata, phantom], ignore_index=True)
            self.metadata['case_id'] = [f'case_{o:04d}' for o in range(len(self.metadata))]
            return self

        available_phantoms = get_available_phantoms()
        if phantom not in available_phantoms:
            raise KeyError(f'Phantom "{phantom}" not available. Available phantoms are: {list(available_phantoms.keys())}')

        case_id = int(self.metadata['case_id'].str.split('_').str[-1].max()) + 1 if not self.metadata.empty else 0
        casestr = f'case_{case_id:04d}'

        new_scan = pd.DataFrame({
            'case_id': [casestr],
            'phantom': [phantom],
            'scanner_model': [scanner_model],
            'kVp': [kVp],
            'mA': [mA],
            'pitch': [pitch],
            'views': [int(views)],
            'fov': [fov],
            'scan_coverage': [scan_coverage],
            'recon_kernel': [recon_kernel],
            'slice_thickness': [slice_thickness],
            'slice_increment': [slice_increment or slice_thickness],
            'seed': [seed],
            'remove_raw': [remove_raw],
            'output_directory': [Path(output_directory).absolute() / casestr]
        })
        self.metadata = pd.concat([self.metadata, new_scan], ignore_index=True)
        return self

    def get_scans_completed(self) -> pd.DataFrame:
        """Collects and returns metadata from all completed scans in the study.

        It searches for 'metadata_*.csv' files within the output directories
        and concatenates them into a single DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with results from all completed scans.
                Returns an empty DataFrame if no results are found.
        """
        results_files = []
        for idx in range(len(self.metadata)):
            output_dir = Path(self.metadata.iloc[idx]['output_directory'])
            if output_dir.exists():
                try:
                    results_files.extend(list(output_dir.rglob('metadata_*.csv')))
                except FileNotFoundError:
                    continue

        if not results_files:
            return pd.DataFrame()

        results_dfs = []
        for o in results_files:
            try:
                results_dfs.append(pd.read_csv(o))
            except pd.errors.EmptyDataError:
                continue
        return pd.concat(results_dfs, ignore_index=True)

    def _monitor_progress(self, log_dir, scans_queued=None):
        """Monitors the progress of parallel scans.

        Args:
            log_dir (Path): Directory where logs are stored.
            scans_queued (int, optional): Total number of scans expected to complete.
                If None, defaults to the length of metadata.
        """
        if scans_queued is None:
            scans_queued = len(self.metadata)

        output_df = self.get_scans_completed()
        scans_completed = len(np.unique(output_df.get('case_id', [])))
        errors = {}
        with tqdm(total=scans_queued, initial=scans_completed, desc='Scans completed in parallel') as pbar:
            while scans_completed < scans_queued:
                sleep(1)
                temp_errors = scan_logs_for_errors(log_dir, verbose=False)
                output_df = self.get_scans_completed()

                # Count successful scans
                successful_scans = len(np.unique(output_df.get('case_id', [])))

                # Check for errors
                new_errors_found = False
                if len(temp_errors) > len(errors):
                    errors = temp_errors
                    new_errors_found = True
                    for task_id in errors:
                        print(f"--- ERROR FOUND IN: {task_id} ---")
                        print(f"Error Message: {errors[task_id]}\n")

                # Total completed = successes + failures
                current_total_completed = successful_scans + len(errors)

                if current_total_completed > scans_completed:
                    pbar.update(current_total_completed - scans_completed)
                    scans_completed = current_total_completed

    def run_all(self, parallel: bool = True, overwrite: bool = False, chunk_size: int | None = None) -> "Study":
        """Runs all scans defined in the study.

        Clears previous results if `overwrite` is True. It attempts to run scans
        in parallel using a batch system (e.g., SGE/qsub) if available and
        `parallel` is True, otherwise runs them serially.

        Args:
            parallel (bool, optional): If True, attempts to run scans in
                parallel. Defaults to True.
            overwrite (bool, optional): If True, deletes previous results before
                running. If False, resumes from unfinished scans.
                Defaults to False.
            chunk_size (int | None, optional): If set and running in parallel,
                splits the study into chunks of this size and runs them
                sequentially. Defaults to None.

        Returns:
            Study: The Study instance, after all scans have been processed.
        """
        if overwrite:
            self.clear_previous_results()

        # Check for chunked execution
        if parallel and (chunk_size is not None) and shutil.which("qsub"):
            study_plan = self.metadata
            chunked_study_plans = [study_plan[i:i + chunk_size] for i in range(0, len(study_plan), chunk_size)]

            for chunk_id, study_plan_chunk in enumerate(chunked_study_plans):
                study_chunk = self.__class__(study_plan_chunk)
                print(f'now running chunk: {chunk_id + 1}/{len(chunked_study_plans)}')
                study_chunk.run_all(parallel=True, chunk_size=None, overwrite=False)
            return self

        results = self.results
        patientids = [int(o.split('case_')[1]) for o in self.metadata.case_id if o not in list(results.get('case_id', []))]
        output = Path(self.metadata.iloc[0]['output_directory']).parent
        if parallel and not shutil.which("qsub"):
            print("qsub not found, running in serial mode.")
            parallel = False

        try:
            patientids = [int(os.environ['SLURM_ARRAY_TASK_ID'])]
            print(f'Now running from job {patientids[0]}')
        except KeyError:
            pass

        log_dir = None
        if parallel:
            pyenv = Path(sys.executable).parent / 'activate'
            now = datetime.now()
            log_name = 'VIT-BATCH_' + now.strftime("%m-%d-%Y_%H-%M")
            log_dir = output.absolute() / 'logs' / log_name
            log_dir.mkdir(exist_ok=True, parents=True)
            run(['bash', str(src_dir / 'run_batchmode.sh'),
                 pyenv,
                 str(src_dir / 'batchmode_CT_dataset_pipeline.sge'),
                 f'{self.csv_fname}',
                 log_dir])

            self._monitor_progress(log_dir)
        else:
            for patientid in tqdm(patientids):
                print(f'Now running: case {patientid}')
                results = self.run_study(patientid)
                series = self.metadata[self.metadata.case_id == f'case_{patientid:04d}'].iloc[0]
                output_directory = Path(series.output_directory)
                print(f"saving intermediate results to {output_directory / f'metadata_{patientid}.csv'}")
                results.to_csv(output_directory / f'metadata_{patientid}.csv',
                               index=False)
                if series.remove_raw:
                    shutil.rmtree(output_directory / series.phantom)
                    [os.remove(o) for o in Path('.').rglob('VIT-BATCH*') if
                     o.is_file()]

        return self

    @property
    def results(self) -> pd.DataFrame:
        """A property to access the aggregated results of all completed scans."""
        return self.get_scans_completed()

    def load_phantom(self, patientid: int = 0) -> Phantom:
        """Loads and initializes the phantom for a specific case.

        Args:
            patientid (int, optional): The index of the scan configuration in
                the metadata. Defaults to 0.

        Returns:
            Phantom: An initialized instance of the specified phantom class.
        """
        series = self.metadata[self.metadata.case_id  == f'case_{patientid:04d}'].iloc[0]
        available_phantoms = get_available_phantoms()
        if series.phantom not in available_phantoms:
            raise ValueError(f'phantom {series.phantom} not in `available_phantoms`, please see `get_available_phantoms()`')
        return available_phantoms[series.phantom]()

    def run_study(self, patientid: int = 0) -> pd.DataFrame:
        """Runs a single simulation for a specific case ID.

        This method orchestrates the full pipeline for one scan:
        1. Initializes the phantom.
        2. Sets up the scanner.
        3. Determines the scan range.
        4. Runs the scan and reconstruction.
        5. Writes the output to DICOM files.
        6. Returns a DataFrame with metadata about the generated images.

        Args:
            patientid (int, optional): The index of the scan configuration in
                `self.metadata` to run. Defaults to 0.

        Returns:
            pd.DataFrame: A DataFrame containing metadata for the generated
                DICOM files, including file paths.
        """
        series = self.metadata[self.metadata.case_id  == f'case_{patientid:04d}'].iloc[0]
        print(f'loading phantom: {series.phantom}')
        phantom = self.load_phantom(patientid)
        print('phantom loaded successfully!')
        print(''.join(10*['-']))
        self.scanner = Scanner(phantom, series.scanner_model,
                               output_dir=series.output_directory)

        scan_coverage = series.scan_coverage
        if isinstance(scan_coverage, tuple | list):
            startZ, endZ = scan_coverage
        elif pd.isna(scan_coverage) or scan_coverage == 'dynamic':
            startZ, endZ = self.scanner.recommend_scan_range()
        elif isinstance(scan_coverage, str):
            startZ, endZ = ast.literal_eval(scan_coverage)
        else:
            raise ValueError(f'scan_coverage datatype not recognized, not list, dataframe or str: {scan_coverage}')
        print('scanner loaded successfully!')
        print(''.join(10*['-']))
        self.scanner.run_scan(startZ=startZ, endZ=endZ,
                              views=int(series.views),
                              mA=series.mA, kVp=series.kVp, pitch=series.pitch)
        self.scanner.run_recon(fov=series.fov, kernel=series.recon_kernel,
                               slice_thickness=series.slice_thickness,
                               slice_increment=series.slice_increment)

        output_directory = Path(series.output_directory)
        dicom_path = output_directory / 'dicoms'
        dcm_files = self.scanner.write_to_dicom(dicom_path / f'{phantom.patient_name}.dcm')

        results_data = series.to_dict()
        results_data.update({
            'name': phantom.patient_name,
            'age': getattr(phantom, 'age', 0),
            'image_file_path': ''
        })
        results_df = pd.DataFrame([results_data] * len(dcm_files))
        results_df['image_file_path'] = dcm_files
        return results_df

    def get_images(self, patientid: int = 0) -> np.ndarray:
        """Loads the reconstructed image volume for a specific case.

        Args:
            patientid (int, optional): The index of the case to load.
                Defaults to 0.

        Returns:
            np.ndarray: A 3D NumPy array containing the image volume.
        """
        case_id_str = f'case_{patientid:04d}'
        image_files = self.results[self.results.case_id == case_id_str]['image_file_path']
        return load_vol(image_files.tolist())


def vit_cli(arg_list: list[str] | None = None):
    """Command-line interface for Virtual Imaging Tools simulations.

    Parses arguments to specify an input CSV for study parameters and an
    option to run in parallel. It initializes a `Study` and calls `run_all`.

    Args:
        arg_list (list[str] | None, optional): A list of command-line
            arguments. If None, `sys.argv[1:]` is used. Defaults to None.
    """
    parser = ArgumentParser(
        description='Runs Virtual Imaging Tools (VITools) simulations',
        epilog='Arguments can be given as TOML config files or command line flags.',
        fromfile_prefix_chars='@')
    parser.add_argument('input_csv', nargs='?', type=str,
                        help='Input CSV to define a study.')
    parser.add_argument('--parallel', '-p', action='store_true',
                        help='Run simulations in parallel using a batch system.')
    parser.add_argument('--overwrite', '-o', action='store_true', help="Overwrites previous results")
    parser.add_argument('--chunk_size', '-c', type=int, default=None,
                        help='Number of simulations to run per chunk in parallel mode.')
    args = parser.parse_args(arg_list)

    input_csv = args.input_csv
    if not input_csv and not sys.stdin.isatty():
        input_csv = sys.stdin.read().strip()

    if not input_csv:
        parser.print_help()
        sys.exit(1)

    Study(input_csv).run_all(args.parallel, overwrite=args.overwrite, chunk_size=args.chunk_size)


def clean_study(study_root: str):
    '''removes intermediate files from incomplete scans

    Args:
        study_root (str): root folder containing study, must contain *_study_plan.csv file
    '''
    study_root = Path(study_root)
    if str(study_root).endswith('.csv'):
        study_root = study_root.parent

    try:
        study_plan = next(study_root.glob('*_study_plan.csv'))
    except StopIteration:
        print(f"Error: No study plan matching '*_study_plan.csv' found in {study_root}")
        return

    study = Study(study_plan)
    results = study.results

    if 'case_id' in results.columns:
        complete_studies = set(sorted(np.unique(results.case_id)))
        print(f"scans completed: {len(complete_studies)}")
    else:
        complete_studies = set()
        print("scans completed: 0")

    incomplete_studies = [
        cid for cid in study.metadata.case_id if cid not in complete_studies
    ]

    print(f"Removing intermediate files from {len(incomplete_studies)} incomplete studies")

    # Assuming output_directory is populated in metadata
    if len(study.metadata) > 0:
        base_dir = Path(study.metadata.output_directory.iloc[0]).parent

        for case_id in tqdm(incomplete_studies):
             # We use case_id as the folder name
             target_dir = base_dir / case_id
             shutil.rmtree(target_dir, ignore_errors=True)
    else:
        print("Study metadata is empty.")


def vit_clean_cli(arg_list: list[str] | None = None):
    """Command-line interface for cleaning VITools simulations.

    Args:
        arg_list (list[str] | None, optional): A list of command-line
            arguments. If None, `sys.argv[1:]` is used. Defaults to None.
    """
    parser = ArgumentParser(
        "Study Cleaning Tool",
        description="removes bulky intermediate files from incomplete sims"
        )
    parser.add_argument(
        'study_root',
        help='root folder containing the study and *_study_plan.csv file'
        )
    args = parser.parse_args(arg_list)
    clean_study(args.study_root)


if __name__ == '__main__':
    vit_cli()
