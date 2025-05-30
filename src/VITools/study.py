'''
The script defines a class `study` with a method `run_series`. This method appears to perform several tasks
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

from .scanner import Scanner

from VITools import get_available_phantoms

src_dir = Path(__file__).parent.absolute()


class Study:
    '''
    defines the study to be simulated and organizes metadata. Initialized by a `Scanner`
    '''
    def __init__(self, input_csv: pd.DataFrame | str | None = None):
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
        if isinstance(input_csv, pd.DataFrame):
            self.metadata = input_csv
        elif isinstance(input_csv, str | Path):
            self.metadata = pd.read_csv(input_csv)

    def clear_previous_results(self):
        for idx in range(len(self.metadata)):
            output_dir = Path(self.metadata.iloc[idx]['OutputDirectory'])
            if output_dir.exists():
                rmtree(output_dir)

    def generate_from_distributions(Phantoms: list[str],
                                    StudyCount: int,
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
        '''
        generate metadata from distributions,

        each parameter specified as a list or dataframe
        '''
        OutputDirectory = Path(OutputDirectory)
        assert (ScanCoverage == 'dynamic') or isinstance(ScanCoverage, list | tuple)
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
        slice_thickness_list = SliceThickness if isinstance(SliceThickness, list | tuple) else [SliceThickness]
        slice_increment_list = SliceIncrement if isinstance(SliceIncrement, list | tuple) else [SliceIncrement]
        FOV_list = FOV if isinstance(FOV, list | tuple) else [FOV]
        kernel_list = ReconKernel if isinstance(ReconKernel, list | tuple) else [ReconKernel]

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
            params['OutputDirectory'].append(OutputDirectory / casestr)
            params['RemoveRawData'].append(RemoveRawData)
        return pd.DataFrame(params)

    def append(self, Phantom,
               OutputDirectory: str | Path = 'results',
               ScannerModel: str = 'Scanner_Default',
               kVp: int = 120, mA: int = 200, Pitch: float = 0,
               Views: int = 1000, FOV: float = 250,
               ScanCoverage: tuple[float] | str = 'dynamic',
               ReconKernel: str = 'standard',
               SliceThickness: int | None = 1,
               SliceIncrement: int | None = None,
               RemoveRawData: bool = True, **kwargs):
        '''add scan or scans to study'''
        if isinstance(Phantom, pd.DataFrame):
            self.metadata = pd.concat([self.metadata, Phantom], ignore_index=True)
            self.metadata['CaseID'] = list(map(lambda o: f'case_{o:04d}',
                                           range(len(self.metadata))))
            return self

        if Phantom not in get_available_phantoms():
            raise KeyError(f'phantom {Phantom} not available. Available phantoms are {get_available_phantoms}')

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
             'RemoveRawData': [RemoveRawData]}
                )
        caseid = int(self.metadata['CaseID'].max().split('case_')[-1]) + 1 if\
            'CaseID' in self.metadata else 0
        casestr = f'case_{caseid:04d}'
        series['CaseID'] = casestr
        series['OutputDirectory'] = [Path(OutputDirectory) / casestr]
        self.metadata = pd.concat([self.metadata, series], ignore_index=True)
        return self

    def get_scans_completed(self):
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
        self.clear_previous_results()
        patientids = list(range(len(self.metadata)))
        if parallel:
            out = run(["qsub", "--help"])
            returncode = out.returncode
            if returncode:
                parallel = False
                print('qsub not found, running in serial mode')
            else:
                output = Path(self.metadata.iloc[0]['OutputDirectory']).parent  # move inside loop
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
                self.run_study(patientid)
        output_df = self.get_scans_completed()

        scans_queued = len(patientids)
        scans_completed = len(self.get_scans_completed())
        with tqdm(total=scans_queued, desc='Scans completed in parallel') as pbar:
            while scans_completed < scans_queued:
                sleep(1)
                if len(self.get_scans_completed()) > scans_completed:
                    pbar.update(len(self.get_scans_completed()) - scans_completed)
                    output_df = self.get_scans_completed()
                    scans_completed = len(output_df)
        return self

    @property
    def results(self):
        return self.get_scans_completed()

    def run_study(self, patientid: int = 0):
        series = self.metadata.iloc[patientid]
        available_phantoms = get_available_phantoms()
        phantom = available_phantoms[series.Phantom]()
        patient_name = phantom.patient_name
        age = phantom.age if hasattr(phantom, 'age') else 0
        scanner = Scanner(phantom, series.ScannerModel,
                          output_dir=series.OutputDirectory)
        ScanCoverage = series.ScanCoverage
        if isinstance(ScanCoverage, float):
            if np.isnan(ScanCoverage):
                ScanCoverage = 'dynamic'
        if isinstance(ScanCoverage, str):
            if ScanCoverage == 'dynamic':
                startZ, endZ = scanner.recommend_scan_range()
            else:
                ScanCoverage = ast.literal_eval(ScanCoverage)
                startZ, endZ = ScanCoverage
        elif isinstance(ScanCoverage, tuple | list):
            startZ, endZ = ScanCoverage
        scanner.run_scan(startZ=startZ, endZ=endZ, views=int(series.Views),
                         mA=series.mA, kVp=series.kVp, pitch=series.Pitch)
        scanner.run_recon(fov=series.FOV, kernel=series.ReconKernel,
                          sliceThickness=series.SliceThickness,
                          sliceIncrement=series.SliceIncrement)

        OutputDirectory = series.OutputDirectory or self.scanner.output_dir
        OutputDirectory = Path(OutputDirectory)
        dicom_path = OutputDirectory / 'dicoms'
        dcm_files = scanner.write_to_dicom(dicom_path / f'{patient_name}.dcm')
        nslices = len(dcm_files)
        results = pd.DataFrame({'Name': nslices*[patient_name],
                                'Age': nslices*[age],
                                'kVp': nslices*[series.kVp],
                                'mA': nslices*[series.mA],
                                'Pitch': nslices*[series.Pitch],
                                'Views': nslices*[series.Views],
                                'ReconKernel': nslices*[series.ReconKernel],
                                'SliceThickness': nslices*[series.SliceThickness],
                                'SliceIncrement': nslices*[series.SliceIncrement],
                                'FOV': nslices*[series.FOV],
                                'ImageFilePath': dcm_files})
        results.to_csv(OutputDirectory / f'metadata_{patientid}.csv',
                       index=False)
        if series.RemoveRawData:
            rmtree(OutputDirectory / patient_name)
            [os.remove(o) for o in Path('.').rglob('VIT-BATCH*') if o.is_file()]
        return self


def vit_cli(arg_list: list[str] | None = None):
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
