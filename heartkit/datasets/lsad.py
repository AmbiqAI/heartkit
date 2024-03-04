import functools
import logging
import os
import random
from collections.abc import Iterable
from enum import IntEnum
from multiprocessing import Pool

import h5py
import numpy as np
import numpy.typing as npt
import physiokit as pk
import sklearn.model_selection
import tensorflow as tf
from tqdm import tqdm

from ..tasks import HKDiagnostic, HKRhythm
from ..utils import download_file
from .dataset import HKDataset
from .defines import PatientGenerator, SampleGenerator

logger = logging.getLogger(__name__)


class LsadScpCode(IntEnum):
    """LSAD SCP codes"""

    # AV Blocks
    AVB = 233917008  # atrioventricular block
    AVB11 = 270492004  # 1 degree atrioventricular block
    AVB2 = 195042002  # 2 degree atrioventricular block
    AVB21 = 54016002  # 2 degree atrioventricular block(Type one)
    AVB22 = 28189009  # 2 degree atrioventricular block(Type two)
    AVB3 = 27885002  # 3 degree atrioventricular block
    AVB221 = 426183003  # Mobitz type II (2nd degree AV block)

    # Axis shifts/rotations
    ALS = 39732003  # Axis left shift
    ARS = 47665007  # Axis right shift
    CR = 251198002  # colockwise rotation
    CCR = 251199005  # countercolockwise rotation

    # Premature and escape beats
    SPB = 63593006  # Supraventricular premature beat
    APB = 284470004  # atrial premature beat
    AEB = 251187003  # Atrial escape beat
    ABI = 251173003  # atrial bigeminy
    JPT = 251164006  # junctional premature beat
    JEB = 426995002  # junctional escape beat
    VPB = 17338001  # ventricular premature beat
    VPB2 = 427172004  # Premature ventricular contraction
    VPE = 195060002  # ventricular preexcitation
    VB = 11157007  # ventricular bigeminy
    VEB = 75532003  # ventricular escape beat
    VFW = 13640000  # ventricular fusion wave
    VET = 251180001  # ventricular escape trigeminy

    # Conduction disturbances
    BBB = 6374002  # BBB
    LBBB = 164909002  # LBBB
    RBBB = 59118001  # RBBB
    ILBBB = 251120003  # Incomplete LBBB
    CRBBB = 713427006  # Complete RBBB
    CLBBB = 733534002  # Complete LBBB
    IRBBB = 713426002  # Incomplete RBBB

    # Hypertrophy
    LVH = 164873001  # left ventricle hypertrophy
    LVH2 = 55827005  # Left ventricular hypertrophy
    RAH = 446358003  # right atrial hypertrophy
    RVH = 89792004  # right ventricle hypertrophy
    LAH = 446813000  # left atrial hypertrophy

    # Myocardial infarction
    MI = 164865005  # MI
    AMI = 57054005  # Acute MI
    AAMI = 54329005  # Acute MI of anterior wall

    # Fiducial abnormalities
    PWE = 251205003  # Prolonged Pwave
    PWT = 251223006  # Tall P wave
    PWC = 164912004  # P wave Change
    PRID = 49578007  # Short PR interval
    PRIE = 164947007  # PR interval extension
    ARW = 365413008  # Abnormal R wave
    QTIE = 111975006  # QT interval extension
    QTID = 77867006  # Short QT interval
    STDD = 429622005  # ST depression
    STE = 164930006  # ST extension
    STTC = 428750005  # ST-T Change
    STTU = 164931005  # ST tilt up
    # 55930002 # ST change
    TWC = 164934002  # T wave Change
    TWO = 59931005  # T wave inverted
    AQW = 164917005  # Abnormal Q wave
    UW = 164937009  # U wave abnormal
    LVQRSAL = 251146004  # lower voltage QRS in all lead
    LVQRSCL = 251148003  # lower voltage QRS in chest lead
    LVQRSLL = 251147008  # lower voltage QRS in limb lead

    SB = 426177001  # Sinus Bradycardia
    SBRAD = 426627000  # Bradycardia

    SR = 426783006  # Sinus Rhythm

    # Fibrillation and flutter
    AFIB = 164889003  # Atrial Fibrillation
    AF = 164890007  # Atrial Flutter
    VFIB = 164896001  # Ventricular Fibrillation
    VF = 111288001  # Ventricular flutter

    AA = 17366009  # Atrial arrhythmia
    SA = 427393009  # Sinus Irregularity / arrhythmia

    # Tachycardia
    ST = 427084000  # Sinus Tachycardia
    SVT = 426761007  # Supraventricular Tachycardia
    AT = 713422000  # Atrial Tachycardia
    JT = 426648003  # Juntional Tacchycardia

    PVT = 425856008  # Paroxysmal ventricular tachycardia
    AVNRT = 233896004  # Atrioventricular Node Reentrant Tachycardia
    AVRT = 233897008  # Atrioventricular Reentrant Tachycardia
    AVNRT2 = 251166008  #  AV Node reentrant tachycardia
    SAAWR = 195101003  # Sinus Atrium to Atrial Wandering Rhythm
    WAVN = 195101003  # Wandering in the atrioventricalualr node
    WPW = 74390002  # Wolff-Parkinson-White
    # 233892002  # Ectopic atrial tachycardia

    # Fascicular blocks
    LAFB = 445118002  # Left anterior fascicular block
    LPFB = 445211001  # Left posterior fascicular block

    # 65778007  # Sinoatrial block
    # 5609005 # sinus arrest

    # Others
    ERV = 428417006  # Early repolarization of the ventricles
    FQRS = 164942001  # fQRS Wave
    IDC = 698252002  # Intraventricular conduction delay
    # 418818005 Brugada syndrome
    # 775926597 not found
    # 61277005  # Accelerated idioventricular rhythm
    # 10370003  # Rhythm from artificial pacemaker
    # 61721007  # Counterclockwise vectorcardiographic loop
    # 50799005  # Atrioventricular dissociation
    # 106068003  # Atrial rhythm
    # 29320008  # Ectopic rhythm
    # 426664006  # Accelerated junctional rhythm
    # 81898007  # Ventrical escape rhythm
    # -2108975294  # ???


##
# These map LSAD specific labels to common labels
##

LsadDiagnosticMap = {
    # NORM
    LsadScpCode.SR: HKDiagnostic.NORM,
    # STTC
    LsadScpCode.STTC: HKDiagnostic.STTC,
    LsadScpCode.QTIE: HKDiagnostic.STTC,
    # MI
    LsadScpCode.MI: HKDiagnostic.MI,
    # HYP
    LsadScpCode.RVH: HKDiagnostic.HYP,
    LsadScpCode.RAH: HKDiagnostic.HYP,
    LsadScpCode.LVH: HKDiagnostic.HYP,
    # CD
    LsadScpCode.LBBB: HKDiagnostic.CD,
    LsadScpCode.RBBB: HKDiagnostic.CD,
    LsadScpCode.AVB: HKDiagnostic.CD,
    LsadScpCode.AVB11: HKDiagnostic.CD,
    LsadScpCode.AVB2: HKDiagnostic.CD,
    LsadScpCode.AVB21: HKDiagnostic.CD,
    LsadScpCode.AVB22: HKDiagnostic.CD,
    LsadScpCode.AVB3: HKDiagnostic.CD,
    LsadScpCode.IDC: HKDiagnostic.CD,
    LsadScpCode.WPW: HKDiagnostic.CD,
}

LsadRhythmMap = {
    LsadScpCode.SR: HKRhythm.sr,
    LsadScpCode.SB: HKRhythm.sbrad,
    LsadScpCode.SBRAD: HKRhythm.sbrad,
    LsadScpCode.ST: HKRhythm.stach,
    LsadScpCode.AA: HKRhythm.sarrh,
    LsadScpCode.SA: HKRhythm.sarrh,
    LsadScpCode.AVNRT: HKRhythm.svt,
    LsadScpCode.AVNRT2: HKRhythm.svt,
    LsadScpCode.AVRT: HKRhythm.svt,
    LsadScpCode.SVT: HKRhythm.svt,
    LsadScpCode.WPW: HKRhythm.svt,
    LsadScpCode.AT: HKRhythm.svt,
    LsadScpCode.JT: HKRhythm.svt,
    LsadScpCode.PVT: HKRhythm.vtach,
    LsadScpCode.AFIB: HKRhythm.afib,
    LsadScpCode.AF: HKRhythm.aflut,
    LsadScpCode.VFIB: HKRhythm.vfib,
    LsadScpCode.VF: HKRhythm.vflut,
    LsadScpCode.ABI: HKRhythm.bigu,
    LsadScpCode.VB: HKRhythm.bigu,
    LsadScpCode.VET: HKRhythm.trigu,
}


LsadLeadsMap = {
    "i": 0,
    "ii": 1,
    "iii": 2,
    "avr": 3,
    "avl": 4,
    "avf": 5,
    "v1": 6,
    "v2": 7,
    "v3": 8,
    "v4": 9,
    "v5": 10,
    "v6": 11,
}


class LsadDataset(HKDataset):
    """LSAD dataset"""

    def __init__(
        self,
        ds_path: os.PathLike,
        task: str,
        frame_size: int,
        target_rate: int,
        spec: tuple[tf.TensorSpec, tf.TensorSpec],
        class_map: dict[int, int] | None = None,
        leads: list[int] | None = None,
    ) -> None:
        super().__init__(
            ds_path=ds_path / "lsad",
            task=task,
            frame_size=frame_size,
            target_rate=target_rate,
            spec=spec,
            class_map=class_map,
        )
        self.leads = leads or list(range(12))

    @property
    def sampling_rate(self) -> int:
        """Sampling rate in Hz"""
        return 500

    @property
    def mean(self) -> float:
        """Dataset mean"""
        return 0

    @property
    def std(self) -> float:
        """Dataset st dev"""
        return 1

    @functools.cached_property
    def patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs

        Returns:
            npt.NDArray: patient IDs
        """

        pts = np.array([int(p.stem) for p in self.ds_path.glob("*.h5")])
        pts.sort()
        return pts

    def get_train_patient_ids(self) -> npt.NDArray:
        """Get dataset training patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        idx = int(len(self.patient_ids) * 0.80)
        return self.patient_ids[:idx]

    def get_test_patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs reserved for testing only

        Returns:
            npt.NDArray: patient IDs
        """
        idx = int(len(self.patient_ids) * 0.80)
        return self.patient_ids[idx:]

    def _pt_key(self, patient_id: int):
        """Get patient key"""
        return f"{patient_id:05d}"

    def task_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Task-level data generator.

        Args:
            patient_generator (PatientGenerator): Patient data generator
            samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample data generator
        """
        if self.task == "rhythm":
            return self.rhythm_data_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )

        if self.task == "denoise":
            return self.denoising_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )

        if self.task == "diagnostic":
            return self.diagnostic_data_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )

        raise NotImplementedError()

    def uniform_patient_generator(
        self,
        patient_ids: npt.NDArray,
        repeat: bool = True,
        shuffle: bool = True,
    ) -> PatientGenerator:
        """Yield data for each patient in the array.

        Args:
            patient_ids (pt.ArrayLike): Array of patient ids
            repeat (bool, optional): Whether to repeat generator. Defaults to True.
            shuffle (bool, optional): Whether to shuffle patient ids.. Defaults to True.

        Returns:
            PatientGenerator: Patient generator

        Yields:
            Iterator[PatientGenerator]
        """
        patient_ids = np.copy(patient_ids)
        while True:
            if shuffle:
                np.random.shuffle(patient_ids)
            for patient_id in patient_ids:
                pt_key = self._pt_key(patient_id)
                with h5py.File(self.ds_path / f"{pt_key}.h5", mode="r") as h5:
                    yield patient_id, h5
                # END WITH
            # END FOR
            if not repeat:
                break
            # END IF
        # END WHILE

    def random_patient_generator(
        self,
        patient_ids: list[int],
        patient_weights: list[int] | None = None,
    ) -> PatientGenerator:
        """Samples patient data from the provided patient distribution.

        Args:
            patient_ids (list[int]): Patient ids
            patient_weights (list[int] | None, optional): Probabilities associated with each patient. Defaults to None.

        Returns:
            PatientGenerator: Patient generator

        Yields:
            Iterator[PatientGenerator]
        """
        while True:
            for patient_id in np.random.choice(patient_ids, size=1024, p=patient_weights):
                pt_key = self._pt_key(patient_id)
                with h5py.File(self.ds_path / f"{pt_key}.h5", mode="r") as h5:
                    yield patient_id, h5
                # END WITH
            # END FOR
        # END WHILE

    def signal_generator(self, patient_generator: PatientGenerator, samples_per_patient: int = 1) -> SampleGenerator:
        """Generate random frames using patient generator.

        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
                    Patient data may contain only signals, since labels are not used.
            samples_per_patient (int): Samples per patient.

        Returns:
            SampleGenerator: Generator of input data of shape (frame_size, 1)
        """
        input_size = int(np.round((self.sampling_rate / self.target_rate) * self.frame_size))
        for _, segment in patient_generator:
            data = segment["data"][:]
            for _ in range(samples_per_patient):
                lead = random.choice(self.leads)
                start = np.random.randint(0, data.shape[1] - input_size)
                x = data[lead, start : start + input_size].squeeze()
                x = np.nan_to_num(x).astype(np.float32)
                if self.sampling_rate != self.target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                # END IF
                yield x
            # END FOR
        # END FOR

    def denoising_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Generate frames and denoised frames."""
        gen = self.signal_generator(patient_generator, samples_per_patient)
        for x in gen:
            y = x.copy()
            yield x, y

    def rhythm_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Generate frames w/ rhythm labels (e.g. afib) using patient generator.

        Args:
            patient_generator (PatientGenerator): Patient Generator
            samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample generator

        Yields:
            Iterator[SampleGenerator]
        """
        return self._label_data_generator(
            patient_generator=patient_generator,
            local_map=LsadRhythmMap,
            samples_per_patient=samples_per_patient,
        )

    def diagnostic_data_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Generate frames w/ diagnostic labels using patient generator.

        Args:
            patient_generator (PatientGenerator): Patient Generator
            samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample generator

        Yields:
            Iterator[SampleGenerator]
        """
        return self._label_data_generator(
            patient_generator=patient_generator,
            local_map=LsadDiagnosticMap,
            samples_per_patient=samples_per_patient,
        )

    def _label_data_generator(
        self,
        patient_generator: PatientGenerator,
        local_map: dict[int, int],
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Generate frames w/ labels using patient generator.

        Args:
            patient_generator (PatientGenerator): Patient Generator
            local_map (dict[int, int]): Local label map
            samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.

        Returns:
            SampleGenerator: Sample generator

        Yields:
            Iterator[SampleGenerator]
        """

        # Target labels and mapping
        tgt_labels = list(set(self.class_map.values()))

        # Convert dataset labels -> HK labels -> class map labels (-1 indicates not in class map)
        tgt_map = {k: self.class_map.get(v, -1) for (k, v) in local_map.items()}
        num_classes = len(tgt_labels)

        # If samples_per_patient is a list, then it must be the same length as nclasses
        if isinstance(samples_per_patient, Iterable):
            samples_per_tgt = samples_per_patient
        else:
            num_per_tgt = int(max(1, samples_per_patient / num_classes))
            samples_per_tgt = num_classes * [num_per_tgt]

        input_size = int(np.round((self.sampling_rate / self.target_rate) * self.frame_size))

        for _, seg in patient_generator:
            # pt_info = {k:v for k,v in seg.attrs.items()}
            # 1. Grab patient scp label (fixed for all samples)
            slabels = seg["slabels"][:]

            # 2. Map scp labels (skip patient if not in class map == -1)
            pt_lbls = []
            pt_lbl_weights = []
            for i in range(slabels.shape[0]):
                label = tgt_map.get(int(slabels[i, 0]), -1)
                if label == -1:
                    continue
                # END IF
                if label not in pt_lbls:
                    pt_lbls.append(label)
                    pt_lbl_weights.append(1 + slabels[i, 1])
                else:
                    i = pt_lbls.index(label)
                    pt_lbl_weights[i] += slabels[i, 1]
                # END IF
            # END FOR

            if len(pt_lbls) == 0:
                continue
            # END IF

            # Its possible to have multiple labels, we assign based on weights
            y = random.choices(pt_lbls, pt_lbl_weights, k=1)[0]

            # 3. Generate samples based on samples_per_tgt
            label_index = tgt_labels.index(y)
            num_samples = samples_per_tgt[label_index]
            data = seg["data"][:]
            # print(f'{pt} creating {num_samples} samples')
            for _ in range(num_samples):
                # select random lead and start index
                lead = random.choice(self.leads)
                # lead = self.leads
                start = np.random.randint(0, data.shape[1] - input_size)
                # Extract frame
                x = np.nan_to_num(data[lead, start : start + input_size], posinf=0, neginf=0).astype(np.float32)
                # Resample if needed
                if self.sampling_rate != self.target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                yield x, y
            # END FOR
            # print(f'{pt} created {num_samples} samples')
        # END FOR

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download dataset

        This will download preprocessed HDF5 files from S3.

        Args:
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """
        self.download_raw_dataset(num_workers=num_workers, force=force)

    def download_raw_dataset(self, num_workers: int | None = None, force: bool = False):
        """Downloads full dataset zipfile and converts into individial patient HDF5 files.

        Args:
            force (bool, optional): Whether to force re-download if destination exists. Defaults to False.
            num_workers (int, optional): # parallel workers. Defaults to os.cpu_count().
        """
        logger.info("Downloading LSAD dataset")
        ds_url = (
            "https://www.physionet.org/static/published-projects/ecg-arrhythmia/"
            "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip"
        )
        ds_zip_path = self.ds_path / "lsad.zip"
        os.makedirs(self.ds_path, exist_ok=True)
        if os.path.exists(ds_zip_path) and not force:
            logger.warning(
                f"Zip file already exists. Please delete or set `force` flag to redownload. PATH={ds_zip_path}"
            )
        else:
            download_file(ds_url, ds_zip_path, progress=True)

        # 2. Extract and convert patient ECG data to H5 files
        logger.info("Processing LSAD patient data")
        self._convert_dataset_zip_to_hdf5(
            zip_path=ds_zip_path,
            force=force,
            num_workers=num_workers,
        )
        logger.info("Finished LSAD patient data")

    def _convert_dataset_zip_to_hdf5(
        self,
        zip_path: os.PathLike,
        patient_ids: npt.NDArray | None = None,
        force: bool = False,
        num_workers: int | None = None,
    ):
        """Convert zipped Icentia dataset into individial patient HDF5 files.

        Args:
            zip_path (PathLike): Zipfile path
            patient_ids (npt.NDArray | None, optional): List of patient IDs to extract. Defaults to all.
            force (bool, optional): Whether to force re-download if destination exists. Defaults to False.
            num_workers (int, optional): # parallel workers. Defaults to os.cpu_count().
        """
        import re  # pylint: disable=import-outside-toplevel
        import tempfile  # pylint: disable=import-outside-toplevel
        import zipfile  # pylint: disable=import-outside-toplevel

        import wfdb  # pylint: disable=import-outside-toplevel

        if not patient_ids:
            patient_ids = self.patient_ids

        zp = zipfile.ZipFile(zip_path, mode="r")  # pylint: disable=consider-using-with

        # zp_root = "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
        # scp_df = pd.read_csv(io.BytesIO(zp.read(os.path.join(zp_root, "scp_statements.csv"))))
        # with open(self.ds_path / "ConditionNames_SNOMED-CT.csv", "wb") as fp:
        #     fp.write(zp.read(os.path.join(zp_root, "scp_statements.csv")))

        zp_rec_names = list(
            filter(
                lambda f: f.endswith(".mat"),
                (f.filename for f in zp.filelist),
            )
        )
        for zp_rec_name in tqdm(zp_rec_names, desc="Converting"):
            try:
                # Extract patient ID by remove JS prefix and .mat suffix
                pt_id = os.path.basename(zp_rec_name).removeprefix("JS").removesuffix(".mat")
                pt_path = self.ds_path / f"{pt_id}.h5"
                with tempfile.TemporaryDirectory() as tmpdir:
                    rec_fpath = os.path.join(tmpdir, f"JS{pt_id}")

                    with open(f"{rec_fpath}.mat", "wb") as fp:
                        fp.write(zp.read(zp_rec_name))
                    with open(f"{rec_fpath}.hea", "wb") as fp:
                        fp.write(zp.read(zp_rec_name.replace(".mat", ".hea")))

                    rec = wfdb.rdrecord(rec_fpath, physical=True)
                    data = rec.p_signal.astype(np.float32).T
                    # rec.comments is a list of strings formatted as "key: value"
                    pt_info = {x.split(":")[0].strip(): x.split(":")[1].strip() for x in rec.comments}
                    del rec
                # END WITH

                if "Dx" not in pt_info:
                    print(f"Skipping {zp_rec_name} - no Dx")
                    continue

                scp_codes = np.array([int(x) for x in re.findall(r"\d+", pt_info["Dx"])])
                slabels = np.array([[x, 100] for x in scp_codes], dtype=np.int32)

                # Save patient data to HDF5
                with h5py.File(pt_path, mode="w") as h5:
                    h5.create_dataset(
                        name="/data",
                        data=data,
                        compression="gzip",
                        compression_opts=6,
                    )

                    h5.create_dataset(name="/slabels", data=slabels)
                    # Add patient info as attributes
                    for k, v in pt_info.items():
                        h5.attrs[k] = v
                    # END FOR
                # END WITH

            except Exception:  # pylint: disable=broad-except
                logger.warning(f"Failed processing {zp_rec_name}")
                continue

    def filter_patients_for_task(self, patient_ids: npt.NDArray) -> npt.NDArray:
        """Filter patients based on task.
        Useful to remove patients w/o labels for task to speed up data loading.

        Args:
            patient_ids (npt.NDArray): Patient ids

        Returns:
            npt.NDArray: Filtered patient ids
        """
        if self.task in ("rhythm", "diagnotic"):
            label_mask = self._get_patient_labels(patient_ids)
            neg_mask = label_mask == -1
            num_neg = neg_mask.sum()
            if num_neg > 0:
                logger.warning(f"Removed {num_neg} of {patient_ids.size} patients w/ no target class")
            return patient_ids[~neg_mask]
        return patient_ids

    def split_train_test_patients(self, patient_ids: npt.NDArray, test_size: float) -> list[list[int]]:
        """Perform train/test split on patients for given task.
        NOTE: We only perform inter-patient splits and not intra-patient.

        Args:
            patient_ids (npt.NDArray): Patient Ids
            test_size (float): Test size

        Returns:
            list[list[int]]: Train and test sets of patient ids
        """
        stratify = None

        # Use stratified split for rhythm task
        if self.task in ("rhythm", "diagnotic"):
            stratify = self._get_patient_labels(patient_ids)
            neg_mask = stratify == -1
            stratify = stratify[~neg_mask]
            patient_ids = patient_ids[~neg_mask]
            num_neg = neg_mask.sum()
            if num_neg > 0:
                logger.warning(f"Removed {num_neg} patients w/ no target class")

        return sklearn.model_selection.train_test_split(
            patient_ids,
            test_size=test_size,
            shuffle=True,
            stratify=stratify,
        )

    def _get_patient_labels(self, patient_ids: npt.NDArray) -> npt.NDArray:
        """Get rhythm class for each patient

        Args:
            patient_ids (npt.NDArray): Patient ids

        Returns:
            npt.NDArray: Patient ids

        """
        ids = patient_ids.tolist()
        with Pool() as pool:
            pt_rhythms = list(pool.imap(self._get_patient_label, ids))
        return np.array(pt_rhythms)

    def _get_patient_label(self, patient_id: int) -> int:
        """Get label class for patient

        Args:
            patient_id (int): Patient id

        Returns:
            int: Target rhythm class
        """
        tgt_map = {k: self.class_map.get(v, -1) for (k, v) in LsadRhythmMap.items()}
        pt_key = self._pt_key(patient_id)
        with h5py.File(self.ds_path / f"{pt_key}.h5", mode="r") as h5:
            pt_rhythms: npt.NDArray[np.int64] = np.array(h5["slabels"][:, 0])
        if pt_rhythms.size == 0:
            return -1
        pt_classes: list[int] = [tgt_map[r] for r in pt_rhythms if tgt_map.get(r, -1) != -1]
        if len(pt_classes) == 0:
            return -1
        return random.choice(pt_classes)
