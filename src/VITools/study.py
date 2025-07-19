'''
This module provides tools for setting up, managing, and running virtual imaging studies.

It defines a `Study` class to organize simulation parameters and results, 
a function to discover available phantom types via a plugin system, 
and a command-line interface for executing studies.
'''

import os
import sys
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


def get_available_phantoms():
    '''
    This module provides tools for setting up, managing, and running virtual imaging studies.

    It defines a `Study` class to organize simulation parameters and results, 
    a function to discover available phantom types via a plugin system, 
    and a command-line interface for executing studies.
    '''
    pm = pluggy.PluginManager(hooks.PROJECT_NAME)
    pm.add_hookspecs(hooks.PhantomSpecs)
    pm.load_setuptools_entrypoints(group=hooks.PROJECT_NAME)

    # --- Call the hook to get all registered phantom types ---
    # The hook returns a list of lists (one list per plugin implementation that returned something)
    list_of_results = pm.hook.register_phantom_types()
    # Flatten the list of lists and filter out None or empty lists from plugins
    discovered_phantom_classes = {}
    for result_list in list_of_results:
        if result_list:  # Check if the plugin returned a non-empty list
            discovered_phantom_classes.update(result_list)
    return discovered_phantom_classes


class Study:
    '''
    Manages and executes a series of imaging simulations (a "study").

    This class handles the definition of study parameters, loading study configurations
    from CSV files or DataFrames, generating scan parameters from distributions,
    and running the simulations either serially or in parallel (if a queuing system
    like qsub or SLURM is available). It also tracks and retrieves simulation results.

    Attributes:
        metadata (pd.DataFrame): A DataFrame holding the parameters for each scan 
                                 in the study.
    '''
    def __init__(self, input_csv: pd.DataFrame | str | None = None):
        '''
        Initializes a Study instance.

        Args:
            input_csv (pd.DataFrame | str | None, optional): 
                Path to a CSV file or a pandas DataFrame containing study metadata.
                If None, an empty study is initialized. Defaults to None.
        '''
        self.metadata = pd.DataFrame()
        if input_csv is not None:
            self.load_study(input_csv)

    def __len__(self):
        return len(self.metadata)

    def __repr__(self):
        repr = f'''
Input metadata:\n
{self.metadata}

Results:\n
{self.results}
'''
        return repr

    def load_study(self, input_csv: str | Path | pd.DataFrame):
        '''
        Loads study metadata from a CSV file or a pandas DataFrame.

        Args:
            input_csv (str | Path | pd.DataFrame):
                The path to a CSV file or a pandas DataFrame containing the study parameters.
                The loaded data replaces any existing metadata in the `self.metadata` attribute.
        '''
        if isinstance(input_csv, pd.DataFrame):
            self.metadata = input_csv
        elif isinstance(input_csv, str | Path):
            self.metadata = pd.read_csv(input_csv)

    def clear_previous_results(self):
        '''
        Removes output directories associated with each scan in the current study metadata.

        This is typically used to clean up before re-running a study. It iterates through
        the 'output_directory' column in the metadata and deletes each directory if it exists.
        '''
        for idx in range(len(self.metadata)):
            output_dir = Path(self.metadata.iloc[idx]['output_directory'])
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    @staticmethod
    def generate_from_distributions(phantoms: list[str],
                                    study_count: int = 1,
                                    output_directory: str | Path = 'results',
                                    views: list[int] = [1000],
                                    scan_coverage='dynamic',
                                    scanner_model: list[str] = ['Scanner_Default'],
                                    kVp: list[int] = [120],
                                    mA: list[int] = [300],
                                    pitch: list[float] = [0],
                                    recon_kernel=['soft'],
                                    slice_thickness: list[int] = [1],
                                    slice_increment: list[int] = [1],
                                    fov: list[float] = [250],
                                    remove_raw: bool = True,
                                    seed: int | None = None):
        '''Generates study metadata by sampling parameters from distributions.

        For each of `study_count` cases, parameters like phantom type, scanner, kVp, mA,
        etc., are chosen randomly from the provided lists. This method allows for the
        creation of diverse datasets for simulation.

        Args:
            phantoms (list[str]): List of phantom names to choose from.
            study_count (int, optional): Number of scan configurations to generate.
            output_directory (str | Path, optional): Base directory for output. Individual case
                directories will be created under this path.
            views (list[int], optional): List of view counts for projection data.
            scan_coverage (str | list | tuple, optional): Scan coverage specification. Can be
                'dynamic' (to auto-determine) or a list/tuple of [start_z, end_z].
            scanner_model (list[str], optional): List of scanner model names.
            kVp (list[int], optional): List of kilovolt peak values.
            mA (list[int], optional): List of milliampere values.
            pitch (list[float], optional): List of pitch values.
            recon_kernel (list[str], optional): List of recon kernel names.
            slice_thickness (list[int], optional): List slice thicknesses in mm.
            slice_increment (list[int], optional): List slice increments in mm.
            fov (list[float], optional): List of Field of View values in mm.
            remove_raw (bool, optional): Whether to remove raw projection data after
                reconstruction.
            seed (int | None, optional): seed for the random number generator. If None or False,
                a random seed is used. If an integer, that seed is used.

        Returns:
            pd.DataFrame: A DataFrame containing the generated study parameters, one row per scan.

        Raises:
            ValueError: If `seed` is a float or True.
        '''
        output_directory = Path(output_directory)
        assert (scan_coverage == 'dynamic') or isinstance(scan_coverage,
                                                         list | tuple)
        if isinstance(scan_coverage, list):
            if len(scan_coverage) < 2:
                scan_coverage = scan_coverage[0].split(' ')
        if isinstance(scan_coverage, list):
            scan_coverage = list(map(int, scan_coverage))
            for o in scan_coverage:
                assert isinstance(o, int | float)

        kVp_list = kVp if isinstance(kVp, list | tuple) else [kVp]
        mA_list = mA if isinstance(mA, list | tuple) else [mA]
        pitch_list = pitch if isinstance(pitch, list | tuple) else [pitch]
        view_list = views if isinstance(views, list | tuple) else [views]
        slice_thickness_list = slice_thickness if\
            isinstance(slice_thickness, list | tuple) else [slice_thickness]
        slice_increment_list = slice_increment if\
            isinstance(slice_increment, list | tuple) else [slice_increment]
        fov_list = fov if isinstance(fov, list | tuple) else [fov]
        kernel_list = recon_kernel if isinstance(recon_kernel, list | tuple)\
            else [recon_kernel]

        if isinstance(seed, float):
            raise ValueError('seed cannot be float, set to False or integer')
        elif not seed:  # check if seed is bool and False
            random = np.random.default_rng()
            global_seed = random.integers(0, 1e6)
            random = np.random.default_rng(global_seed)
        elif seed is True:  # check if seed is bool and True
            raise ValueError('seed cannot be True, set to False or integer')
        elif isinstance(seed, int):  # if not True or False, check if int:
            global_seed = seed
            random = np.random.default_rng(seed)
        else:
            raise ValueError('seed must be False or integer')

        params = {
            'case_id': [],
            'phantom': [],
            'scanner_model': [],
            'kVp': [],
            'mA': [],
            'pitch': [],
            'views': [],
            'scan_coverage': [],
            'recon_kernel': [],
            'slice_thickness': [],
            'slice_increment': [],
            'fov': [],
            'global_seed': [],
            'case_seed': [],
            'output_directory': [],
            'remove_raw': []
        }

        for i in range(study_count):
            casestr = f'case_{i:04d}'
            params['case_id'].append(casestr)
            params['phantom'].append(random.choice(list(phantoms)))
            params['scanner_model'].append(random.choice(scanner_model))
            params['kVp'].append(float(random.choice(kVp_list)))
            params['mA'].append(float(random.choice(mA_list)))
            params['pitch'].append(float(random.choice(pitch_list)))
            params['views'].append(float(random.choice(view_list)))
            params['scan_coverage'].append(scan_coverage)
            params['recon_kernel'].append(random.choice(kernel_list))
            params['slice_thickness'].append(random.choice(slice_thickness_list))
            params['slice_increment'].append(random.choice(slice_increment_list))
            params['fov'].append(random.choice(fov_list))
            params['global_seed'].append(global_seed)
            params['case_seed'].append(random.integers(0, 1e6))
            params['output_directory'].append(output_directory.absolute() / casestr)
            params['remove_raw'].append(remove_raw)
        return pd.DataFrame(params)

    def append(self, phantom: str | pd.DataFrame,
               output_directory: str | Path = 'results',
               scanner_model: str = 'Scanner_Default',
               kVp: int = 120, mA: int = 200, pitch: float = 0,
               views: int = 1000, fov: float = 250,
               scan_coverage: tuple[float] | str = 'dynamic',
               recon_kernel: str = 'standard',
               slice_thickness: int | None = 1,
               slice_increment: int | None = None,
               seed: int | None = None,
               remove_raw: bool = True, **kwargs):
        '''Appends one or more scans to the study's metadata.

        If `phantom` is a DataFrame, it's concatenated to the existing metadata.
        Otherwise, a new scan configuration is created using the provided parameters
        and added to the metadata.

        Args:
            phantom (str | pd.DataFrame): The name of the phantom to use (must be an available phantom type)
                or a pandas DataFrame containing scan parameters to append.
            output_directory (str | Path, optional): Base directory for output.
                Defaults to 'results'.
            scanner_model (str, optional): Scanner model name.
                Defaults to 'Scanner_Default'.
            kVp (int, optional): Kilovolt peak value.
                Defaults to 120.
            mA (int, optional): Milliampere value. Defaults to 200.
            pitch (float, optional): pitch value. Defaults to 0.
            views (int, optional): Number of views for projection. Defaults to 1000.
            fov (float, optional): Field of View in mm. Defaults to 250.
            scan_coverage (tuple[float,...] | str, optional): Scan coverage. Can be 'dynamic' or a tuple (start_z, end_z).
                Defaults to 'dynamic'.
            recon_kernel (str, optional): Reconstruction kernel name. Defaults to 'standard'.
            slice_thickness (int | None, optional): Slice thickness in mm. Defaults to 1.
            slice_increment (int | None, optional): Slice increment in mm.
                Defaults to `slice_thickness` if None.
            seed (int | None, optional): seed for the random number generator.
                If None or False, a random seed is used. If an integer,
                that seed is used. Defaults to None.
            remove_raw (bool, optional): Whether to remove raw data after
                reconstruction. Defaults to True.

        Returns:
            Study: The Study instance itself, allowing for method chaining.

        Raises:
            KeyError: If the specified `phantom` name is not an available phantom type.
        '''
        if isinstance(phantom, pd.DataFrame):
            self.metadata = pd.concat([self.metadata, phantom], ignore_index=True)
            self.metadata['case_id'] = list(map(lambda o: f'case_{o:04d}',
                                           range(len(self.metadata))))
            return self
        available_phantoms = get_available_phantoms()
        if phantom not in available_phantoms:
            raise KeyError(f'phantom {phantom} not available. Available phantoms are {available_phantoms}')

        series = pd.DataFrame(
            {'phantom': [phantom],
             'scanner_model': [scanner_model],
             'kVp': [kVp],
             'mA': [mA],
             'views': [int(views)],
             'scan_coverage': [scan_coverage],
             'pitch': [pitch],
             'recon_kernel': [recon_kernel],
             'slice_thickness': [slice_thickness],
             'slice_increment': [slice_increment],
             'fov': [fov],
             'case_seed': [seed],
             'remove_raw': [remove_raw]}
                )
        case_id = int(self.metadata['case_id'].max().split('case_')[-1]) + 1 if\
            'case_id' in self.metadata else 0
        casestr = f'case_{case_id:04d}'
        series['case_id'] = casestr
        series['output_directory'] = [Path(output_directory).absolute() / casestr]
        self.metadata = pd.concat([self.metadata, series], ignore_index=True)
        return self

    def get_scans_completed(self):
        '''
        Collects and returns metadata from all completed scans in the study.

        It searches for 'metadata_*.csv' files within the output directories specified
        in the study's metadata. These files are assumed to contain results
        from individual scan simulations.

        Returns:
            pd.DataFrame | list:
                A pandas DataFrame concatenating all found metadata CSV files.
                Returns an empty list if no result files are found.
        '''
        scans_completed = 0
        results_files = []
        for idx in range(len(self.metadata)):
            output_dir = Path(self.metadata.iloc[idx]['output_directory'])
            try:
                results_files.extend(list(output_dir.rglob('metadata_*.csv')))
            except FileNotFoundError:
                continue
            if len(results_files) > 0:
                scans_completed += 1
        if len(results_files) < 1:
            return pd.DataFrame()
        return pd.concat([pd.read_csv(o) for o in results_files],
                         ignore_index=True)

    def run_all(self, parallel=True) -> pd.DataFrame:
        '''
        Runs all scans defined in the study.

        Clears previous results, then attempts to run scans in parallel using qsub
        if `parallel` is True and qsub is available. Otherwise, runs scans serially.
        It monitors the progress of parallel jobs and waits for their completion.

        Args:
            parallel (bool, optional):
                If True, attempts to run scans in parallel using batch system.
                If False or if the batch system is not found, runs serially.
                Defaults to True.

        Returns:
            Study: The Study instance itself, after all scans have processed.
        '''
        self.clear_previous_results()
        patientids = list(range(len(self.metadata)))
        if parallel and not shutil.which("qsub"):
            print("qsub not found, running in serial mode.")
            parallel = False
        else:
            output = Path(self.metadata.iloc[0]['output_directory']).parent
            output.mkdir(exist_ok=True, parents=True)
            csv_fname = output / 'sim_input.csv'
            csv_fname = csv_fname.absolute()
            self.metadata.to_csv(csv_fname)

        try:
            patientids = [int(os.environ['SLURM_ARRAY_TASK_ID']) - 1]
            print(f'Now running from job {patientids[0] + 1}')
        except KeyError:
            pass

        if parallel:
            run(['bash', str(src_dir / 'run_batchmode.sh'),
                 str(src_dir / 'batchmode_CT_dataset_pipeline.sge'),
                 f'{csv_fname}'])
        else:
            for patientid in tqdm(patientids):
                results = self.run_study(patientid)
                series = self.metadata.iloc[patientid]
                output_directory = Path(series.output_directory)
                results.to_csv(output_directory / f'metadata_{patientid}.csv',
                               index=False)
                if series.remove_raw:
                    shutil.rmtree(output_directory / series.phantom)
                    [os.remove(o) for o in Path('.').rglob('VIT-BATCH*') if
                     o.is_file()]

        output_df = self.get_scans_completed()
        scans_queued = len(patientids)
        output_df = self.get_scans_completed()
        scans_completed = len(np.unique(output_df.get('case_id', [])))
        with tqdm(total=scans_queued, desc='Scans completed in parallel') as pbar:
            while scans_completed < scans_queued:
                sleep(1)
                output_df = self.get_scans_completed()
                if len(np.unique(output_df.get('case_id', []))) > scans_completed:
                    pbar.update(
                        len(np.unique(output_df.get('case_id', []))) - scans_completed
                        )
                    scans_completed = len(np.unique(output_df.get('case_id', [])))
        return self

    @property
    def results(self):
        return self.get_scans_completed()

    def load_phantom(self,  patientid: int = 0) -> Phantom:
        series = self.metadata.iloc[patientid]
        available_phantoms = get_available_phantoms()
        return available_phantoms[series.phantom]()

    def run_study(self, patientid: int = 0):
        '''
        Runs a single simulation study for a specific patient/case ID.

        This method orchestrates the simulation for one entry in the
        `self.metadata` DataFrame. It involves:
        1. Initializing the specified phantom.
        2. Setting up the virtual scanner.
        3. Determining the scan range (z-axis coverage).
        4. Running the scan (projection data generation).
        5. Running the reconstruction.
        6. Writing the reconstructed images to DICOM files.
        7. Saving metadata about the scan.
        8. Optionally, removing raw projection data.

        Args:
            patientid (int, optional):
                The index of the scan configuration in `self.metadata` to run.
                Defaults to 0 (the first scan).

        Returns:
            Study: The Study instance itself.

        Raises:
            KeyError: If the phantom specified in the metadata is not found.
            IndexError: If `patientid` is out of bounds for `self.metadata`.
        '''
        series = self.metadata.iloc[patientid]
        phantom = self.load_phantom(patientid)
        patient_name = phantom.patient_name
        age = phantom.age if hasattr(phantom, 'age') else 0
        self.scanner = Scanner(phantom, series.scanner_model,
                               output_dir=series.output_directory)
        scan_coverage = series.scan_coverage
        if isinstance(scan_coverage, float):
            if np.isnan(scan_coverage):
                scan_coverage = 'dynamic'
        if isinstance(scan_coverage, str):
            if scan_coverage == 'dynamic':
                startZ, endZ = self.scanner.recommend_scan_range()
            else:
                scan_coverage = ast.literal_eval(scan_coverage)
                startZ, endZ = scan_coverage
        elif isinstance(scan_coverage, tuple | list):
            startZ, endZ = scan_coverage
        self.scanner.run_scan(startZ=startZ, endZ=endZ,
                              views=int(series.views),
                              mA=series.mA, kVp=series.kVp, pitch=series.pitch)
        self.scanner.run_recon(fov=series.fov, kernel=series.recon_kernel,
                               slice_thickness=series.slice_thickness,
                               slice_increment=series.slice_increment)

        output_directory = series.output_directory or self.scanner.output_dir
        output_directory = Path(output_directory)
        dicom_path = output_directory / 'dicoms'
        dcm_files = self.scanner.write_to_dicom(dicom_path /
                                                f'{patient_name}.dcm')
        nslices = len(dcm_files)
        results = pd.DataFrame(
            {'case_id': nslices*[series.case_id],
             'name': nslices*[patient_name],
             'age': nslices*[age],
             'kVp': nslices*[series.kVp],
             'mA': nslices*[series.mA],
             'pitch': nslices*[series.pitch],
             'views': nslices*[series.views],
             'scanner_model': nslices*[series.scanner_model],
             'recon_kernel': nslices*[series.recon_kernel],
             'slice_thickness': nslices*[series.slice_thickness],
             'slice_increment': nslices*[series.slice_increment],
             'fov': nslices*[series.fov],
             'case_seed': nslices*[series.case_seed],
             'image_file_path': dcm_files}
             )
        return results

    def get_images(self, patientid: int = 0):
        return load_vol(self.results[self.results.case_id ==
                                     f'case_{patientid:04d}']['image_file_path'])


def vit_cli(arg_list: list[str] | None = None):
    '''
    Command-line interface for Virtual Imaging Tools (VITools) simulations.

    Parses command-line arguments to specify an input CSV file for study
    parameters and an option to run simulations in parallel. If no input CSV
    is provided via arguments, it attempts to read from stdin.

    The input CSV can define a study to be run. This function initializes a
    `Study` object with this CSV and then calls its `run_all` method.

    Args:
        arg_list (list[str] | None, optional):
            A list of command-line arguments to parse. If None, `sys.argv[1:]`
            is used.
            Defaults to None.
    '''
    parser = ArgumentParser(
        description='Runs Virtual Imaging Tools (VITools) simulations',
        epilog='''
        arguments can be given as toml config files or command line
        flags, each overriding defaults
        ''',
        fromfile_prefix_chars='@')
    parser.add_argument('input_csv', nargs='?', type=str,
                        help='''
                          input csv to recreate prior dataset,
                          see `recruit --help` for more details
                        ''')
    parser.add_argument('--parallel', '-p', type=bool,
                        default=False,
                        help='run simulations in parallel')
    args = parser.parse_args(arg_list)
    if args.input_csv:
        input_csv = args.input_csv
    elif not sys.stdin.isatty():
        input_csv = sys.stdin.read().strip()
    else:
        parser.print_help()

    Study(input_csv).run_all(args.parallel)


if __name__ == '__main__':
    vit_cli()
