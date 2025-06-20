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
from shutil import rmtree
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
        the 'OutputDirectory' column in the metadata and deletes each directory if it exists.
        '''
        for idx in range(len(self.metadata)):
            output_dir = Path(self.metadata.iloc[idx]['OutputDirectory'])
            if output_dir.exists():
                rmtree(output_dir)

    @staticmethod
    def generate_from_distributions(Phantoms: list[str],
                                    StudyCount: int = 1,
                                    OutputDirectory: str | Path = 'results',
                                    Views: list[int] = [1000],
                                    ScanCoverage='dynamic',
                                    ScannerModel: list[str] = ['Scanner_Default'],
                                    kVp: list[int] = [120],
                                    mA: list[int] = [300],
                                    Pitch: list[float] = [0],
                                    ReconKernel=['soft'],
                                    SliceThickness: list[int] = [1],
                                    SliceIncrement: list[int] = [1],
                                    FOV: list[float] = [250],
                                    RemoveRawData: bool = True,
                                    Seed: int | None = None):
        '''Generates study metadata by sampling parameters from distributions.

        For each of `StudyCount` cases, parameters like phantom type, scanner, kVp, mA,
        etc., are chosen randomly from the provided lists. This method allows for the
        creation of diverse datasets for simulation.

        Args:
            Phantoms (list[str]): List of phantom names to choose from.
            StudyCount (int, optional): Number of scan configurations to generate.
            OutputDirectory (str | Path, optional): Base directory for output. Individual case
                directories will be created under this path.
            Views (list[int], optional): List of view counts for projection data.
            ScanCoverage (str | list | tuple, optional): Scan coverage specification. Can be
                'dynamic' (to auto-determine) or a list/tuple of [start_z, end_z].
            ScannerModel (list[str], optional): List of scanner model names.
            kVp (list[int], optional): List of kilovolt peak values.
            mA (list[int], optional): List of milliampere values.
            Pitch (list[float], optional): List of pitch values.
            ReconKernel (list[str], optional): List of recon kernel names.
            SliceThickness (list[int], optional): List slice thicknesses in mm.
            SliceIncrement (list[int], optional): List slice increments in mm.
            FOV (list[float], optional): List of Field of View values in mm.
            RemoveRawData (bool, optional): Whether to remove raw projection data after
                reconstruction.
            Seed (int | None, optional): Seed for the random number generator. If None or False,
                a random seed is used. If an integer, that seed is used.

        Returns:
            pd.DataFrame: A DataFrame containing the generated study parameters, one row per scan.

        Raises:
            ValueError: If `Seed` is a float or True.
        '''
        OutputDirectory = Path(OutputDirectory)
        assert (ScanCoverage == 'dynamic') or isinstance(ScanCoverage,
                                                         list | tuple)
        if isinstance(ScanCoverage, list):
            if len(ScanCoverage) < 2:
                ScanCoverage = ScanCoverage[0].split(' ')
        if isinstance(ScanCoverage, list):
            ScanCoverage = list(map(int, ScanCoverage))
            for o in ScanCoverage:
                assert isinstance(o, int | float)

        kVp_list = kVp if isinstance(kVp, list | tuple) else [kVp]
        mA_list = mA if isinstance(mA, list | tuple) else [mA]
        pitch_list = Pitch if isinstance(Pitch, list | tuple) else [Pitch]
        view_list = Views if isinstance(Views, list | tuple) else [Views]
        slice_thickness_list = SliceThickness if\
            isinstance(SliceThickness, list | tuple) else [SliceThickness]
        slice_increment_list = SliceIncrement if\
            isinstance(SliceIncrement, list | tuple) else [SliceIncrement]
        FOV_list = FOV if isinstance(FOV, list | tuple) else [FOV]
        kernel_list = ReconKernel if isinstance(ReconKernel, list | tuple)\
            else [ReconKernel]

        if isinstance(Seed, float):
            raise ValueError('Seed cannot be float, set to False or integer')
        elif not Seed:  # check if Seed is bool and False
            random = np.random.default_rng()
            global_seed = random.integers(0, 1e6)
            random = np.random.default_rng(global_seed)
        elif Seed is True:  # check if Seed is bool and True
            raise ValueError('Seed cannot be True, set to False or integer')
        elif isinstance(Seed, int):  # if not True or False, check if int:
            global_seed = Seed
            random = np.random.default_rng(Seed)
        else:
            raise ValueError('Seed must be False or integer')

        params = {
            'CaseID': [],
            'Phantom': [],
            'ScannerModel': [],
            'kVp': [],
            'mA': [],
            'Pitch': [],
            'Views': [],
            'ScanCoverage': [],
            'ReconKernel': [],
            'SliceThickness': [],
            'SliceIncrement': [],
            'FOV': [],
            'GlobalSeed': [],
            'CaseSeed': [],
            'OutputDirectory': [],
            'RemoveRawData': []
        }

        for i in range(StudyCount):
            casestr = f'case_{i:04d}'
            params['CaseID'].append(casestr)
            params['Phantom'].append(random.choice(list(Phantoms)))
            params['ScannerModel'].append(random.choice(ScannerModel))
            params['kVp'].append(float(random.choice(kVp_list)))
            params['mA'].append(float(random.choice(mA_list)))
            params['Pitch'].append(float(random.choice(pitch_list)))
            params['Views'].append(float(random.choice(view_list)))
            params['ScanCoverage'].append(ScanCoverage)
            params['ReconKernel'].append(random.choice(kernel_list))
            params['SliceThickness'].append(random.choice(slice_thickness_list))
            params['SliceIncrement'].append(random.choice(slice_increment_list))
            params['FOV'].append(random.choice(FOV_list))
            params['GlobalSeed'].append(global_seed)
            params['CaseSeed'].append(random.integers(0, 1e6))
            params['OutputDirectory'].append(OutputDirectory.absolute() / casestr)
            params['RemoveRawData'].append(RemoveRawData)
        return pd.DataFrame(params)

    def append(self, Phantom: str | pd.DataFrame,
               OutputDirectory: str | Path = 'results',
               ScannerModel: str = 'Scanner_Default',
               kVp: int = 120, mA: int = 200, Pitch: float = 0,
               Views: int = 1000, FOV: float = 250,
               ScanCoverage: tuple[float] | str = 'dynamic',
               ReconKernel: str = 'standard',
               SliceThickness: int | None = 1,
               SliceIncrement: int | None = None,
               Seed: int | None = None,
               RemoveRawData: bool = True, **kwargs):
        '''Appends one or more scans to the study's metadata.

        If `Phantom` is a DataFrame, it's concatenated to the existing metadata.
        Otherwise, a new scan configuration is created using the provided parameters
        and added to the metadata.

        Args:
            Phantom (str | pd.DataFrame): The name of the phantom to use (must be an available phantom type)
                or a pandas DataFrame containing scan parameters to append.
            OutputDirectory (str | Path, optional): Base directory for output.
                Defaults to 'results'.
            ScannerModel (str, optional): Scanner model name.
                Defaults to 'Scanner_Default'.
            kVp (int, optional): Kilovolt peak value.
                Defaults to 120.
            mA (int, optional): Milliampere value. Defaults to 200.
            Pitch (float, optional): Pitch value. Defaults to 0.
            Views (int, optional): Number of views for projection. Defaults to 1000.
            FOV (float, optional): Field of View in mm. Defaults to 250.
            ScanCoverage (tuple[float,...] | str, optional): Scan coverage. Can be 'dynamic' or a tuple (start_z, end_z).
                Defaults to 'dynamic'.
            ReconKernel (str, optional): Reconstruction kernel name. Defaults to 'standard'.
            SliceThickness (int | None, optional): Slice thickness in mm. Defaults to 1.
            SliceIncrement (int | None, optional): Slice increment in mm.
                Defaults to `SliceThickness` if None.
            Seed (int | None, optional): Seed for the random number generator.
                If None or False, a random seed is used. If an integer,
                that seed is used. Defaults to None.
            RemoveRawData (bool, optional): Whether to remove raw data after
                reconstruction. Defaults to True.

        Returns:
            Study: The Study instance itself, allowing for method chaining.

        Raises:
            KeyError: If the specified `Phantom` name is not an available phantom type.
        '''
        if isinstance(Phantom, pd.DataFrame):
            self.metadata = pd.concat([self.metadata, Phantom], ignore_index=True)
            self.metadata['CaseID'] = list(map(lambda o: f'case_{o:04d}',
                                           range(len(self.metadata))))
            return self
        available_phantoms = get_available_phantoms()
        if Phantom not in available_phantoms:
            raise KeyError(f'phantom {Phantom} not available. Available phantoms are {available_phantoms}')

        series = pd.DataFrame(
            {'Phantom': [Phantom],
             'ScannerModel': [ScannerModel],
             'kVp': [kVp],
             'mA': [mA],
             'Views': [int(Views)],
             'ScanCoverage': [ScanCoverage],
             'Pitch': [Pitch],
             'ReconKernel': [ReconKernel],
             'SliceThickness': [SliceThickness],
             'SliceIncrement': [SliceIncrement],
             'FOV': [FOV],
             'CaseSeed': [Seed],
             'RemoveRawData': [RemoveRawData]}
                )
        caseid = int(self.metadata['CaseID'].max().split('case_')[-1]) + 1 if\
            'CaseID' in self.metadata else 0
        casestr = f'case_{caseid:04d}'
        series['CaseID'] = casestr
        series['OutputDirectory'] = [Path(OutputDirectory).absolute() / casestr]
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
            output_dir = Path(self.metadata.iloc[idx]['OutputDirectory'])
            try:
                results_files.extend(list(output_dir.rglob('metadata_*.csv')))
            except FileNotFoundError:
                continue
            if len(results_files) > 0:
                scans_completed += 1
        if len(results_files) < 1:
            return []
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
        returncode = True
        if parallel:
            try:
                out = run(["qsub", "--help"])
                returncode = out.returncode
            except FileNotFoundError:
                returncode = True
        if returncode:
            parallel = False
            print('qsub not found, running in serial mode')
        else:
            output = Path(self.metadata.iloc[0]['OutputDirectory']).parent
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
                OutputDirectory = Path(series.OutputDirectory)
                results.to_csv(OutputDirectory / f'metadata_{patientid}.csv',
                               index=False)
                if series.RemoveRawData:
                    rmtree(OutputDirectory / series.Phantom)
                    [os.remove(o) for o in Path('.').rglob('VIT-BATCH*') if
                     o.is_file()]

        output_df = self.get_scans_completed()
        scans_queued = len(patientids)
        scans_completed = len(self.get_scans_completed())
        with tqdm(total=scans_queued,
                  desc='Scans completed in parallel') as pbar:
            while scans_completed < scans_queued:
                sleep(1)
                if len(self.get_scans_completed()) > scans_completed:
                    pbar.update(
                        len(self.get_scans_completed()) - scans_completed
                        )
                    output_df = self.get_scans_completed()
                    scans_completed = len(output_df)
        return self

    @property
    def results(self):
        return self.get_scans_completed()

    def load_phantom(self,  patientid: int = 0) -> Phantom:
        series = self.metadata.iloc[patientid]
        available_phantoms = get_available_phantoms()
        return available_phantoms[series.Phantom]()

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
        self.scanner = Scanner(phantom, series.ScannerModel,
                               output_dir=series.OutputDirectory)
        ScanCoverage = series.ScanCoverage
        if isinstance(ScanCoverage, float):
            if np.isnan(ScanCoverage):
                ScanCoverage = 'dynamic'
        if isinstance(ScanCoverage, str):
            if ScanCoverage == 'dynamic':
                startZ, endZ = self.scanner.recommend_scan_range()
            else:
                ScanCoverage = ast.literal_eval(ScanCoverage)
                startZ, endZ = ScanCoverage
        elif isinstance(ScanCoverage, tuple | list):
            startZ, endZ = ScanCoverage
        self.scanner.run_scan(startZ=startZ, endZ=endZ,
                              views=int(series.Views),
                              mA=series.mA, kVp=series.kVp, pitch=series.Pitch)
        self.scanner.run_recon(fov=series.FOV, kernel=series.ReconKernel,
                               sliceThickness=series.SliceThickness,
                               sliceIncrement=series.SliceIncrement)

        OutputDirectory = series.OutputDirectory or self.scanner.output_dir
        OutputDirectory = Path(OutputDirectory)
        dicom_path = OutputDirectory / 'dicoms'
        dcm_files = self.scanner.write_to_dicom(dicom_path /
                                                f'{patient_name}.dcm')
        nslices = len(dcm_files)
        results = pd.DataFrame(
            {'CaseID': nslices*[series.CaseID],
             'Name': nslices*[patient_name],
             'Age': nslices*[age],
             'kVp': nslices*[series.kVp],
             'mA': nslices*[series.mA],
             'Pitch': nslices*[series.Pitch],
             'Views': nslices*[series.Views],
             'ScannerModel': nslices*[series.ScannerModel],
             'ReconKernel': nslices*[series.ReconKernel],
             'SliceThickness': nslices*[series.SliceThickness],
             'SliceIncrement': nslices*[series.SliceIncrement],
             'FOV': nslices*[series.FOV],
             'CaseSeed': nslices*[series.CaseSeed],
             'ImageFilePath': dcm_files}
             )
        return results

    def get_images(self, patientid: int = 0):
        return load_vol(self.results[self.results.CaseID ==
                                     f'case_{patientid:04d}']['ImageFilePath'])


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
