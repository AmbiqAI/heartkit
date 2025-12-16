import contextlib
import functools
import os
import zipfile
import random
from collections.abc import Iterable
from enum import IntEnum
from typing import Generator

import h5py
import numpy as np
import numpy.typing as npt
import physiokit as pk
import sklearn.model_selection
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import helia_edge as helia

from .dataset import HKDataset
from .defines import PatientGenerator, PatientData

logger = helia.utils.setup_logger(__name__)


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
    AVD = 50799005  # Atrioventricular dissociation
    # 106068003  # Atrial rhythm
    # 29320008  # Ectopic rhythm
    # 426664006  # Accelerated junctional rhythm
    # 81898007  # Ventrical escape rhythm
    # -2108975294  # ???


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
        leads: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.leads = leads or list(LsadLeadsMap.values())
        self._cached_data = {}

    @property
    def name(self) -> str:
        """Dataset name"""
        return "lsad"

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
        pts = np.array([int(p.stem) for p in self.path.glob("*.h5")])
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

    def label_key(self, label_type: str = "scp") -> str:
        """Get label key

        Args:
            label_type (str, optional): Label type. Defaults to "scp".

        Returns:
            str: Label key
        """
        if label_type == "scp":
            return "slabels"
        raise ValueError(f"Invalid label type: {label_type}")

    @contextlib.contextmanager
    def patient_data(self, patient_id: int) -> Generator[PatientData, None, None]:
        """Get patient data

        Args:
            patient_id (int): Patient ID

        Returns:
            Generator[PatientData, None, None]: Patient data
        """
        pt_key = self._pt_key(patient_id)
        pt_path = self.path / f"{pt_key}.h5"
        if self.cacheable:
            if pt_key not in self._cached_data:
                pt_data = {}
                with h5py.File(pt_path, mode="r") as h5:
                    pt_data["data"] = h5["data"][:]
                    pt_data[self.label_key("scp")] = h5[self.label_key("scp")][:]
                self._cached_data[pt_key] = pt_data
            # END IF
            yield self._cached_data[pt_key]
        else:
            with h5py.File(pt_path, mode="r") as h5:
                yield h5
            # END WITH
        # END IF

    def signal_generator(
        self,
        patient_generator: PatientGenerator,
        frame_size: int,
        samples_per_patient: int = 1,
        target_rate: int | None = None,
    ) -> Generator[npt.NDArray, None, None]:
        """Generate random frames.

        Args:
            patient_generator (PatientGenerator): Generator that yields patient data.
            frame_size (int): Frame size
            samples_per_patient (int, optional): Samples per patient. Defaults to 1.
            target_rate (int | None, optional): Target rate. Defaults to None.

        Returns:
            Generator[npt.NDArray, None, None]: Generator of input data
        """
        if target_rate is None:
            target_rate = self.sampling_rate

        input_size = int(np.ceil((self.sampling_rate / target_rate) * frame_size))
        for pt in patient_generator:
            with self.patient_data(pt) as h5:
                data = h5["data"][:]
            # END WITH
            for _ in range(samples_per_patient):
                lead = random.choice(self.leads)
                start = np.random.randint(0, data.shape[1] - input_size)
                x = data[lead, start : start + input_size].squeeze()
                x = np.nan_to_num(x).astype(np.float32)
                if self.sampling_rate != target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, target_rate, axis=0)
                    x = x[:frame_size]
                # END IF
                yield x
            # END FOR
        # END FOR

    def signal_label_generator(
        self,
        patient_generator: PatientGenerator,
        frame_size: int,
        samples_per_patient: int = 1,
        target_rate: int | None = None,
        label_map: dict[int, int] | None = None,
        label_type: str = "scp",
        label_format: str | None = None,
    ) -> Generator[tuple[npt.NDArray, int | npt.NDArray], None, None]:
        """Generate frames w/ labels using patient generator.

        Args:
            patient_generator (PatientGenerator): Patient Generator
            frame_size (int): Frame size
            samples_per_patient (int, optional): Samples per patient. Defaults to 1.
            target_rate (int, optional): Target rate. Defaults to None.
            label_map (dict[int, int], optional): Label map. Defaults to None.
            label_type (str, optional): Class type. Defaults to "scp".
            label_format (str, optional): Label format. Defaults to None.

        Returns:
            Generator[tuple[npt.NDArray, int|npt.NDArray], None, None]: Generator of input data and labels

        Yields:
            tuple[npt.NDArray, int]: Input data and label
        """
        if target_rate is None:
            target_rate = self.sampling_rate
        # END IF

        # Target labels and mapping
        tgt_labels = sorted(list(set((lbl for lbl in label_map.values() if lbl != -1))))
        label_key = self.label_key(label_type)
        num_classes = len(tgt_labels)

        # If samples_per_patient is a list, then it must be the same length as nclasses
        if isinstance(samples_per_patient, Iterable):
            samples_per_tgt = samples_per_patient
        else:
            num_per_tgt = int(max(1, samples_per_patient / num_classes))
            samples_per_tgt = num_classes * [num_per_tgt]

        input_size = int(np.ceil((self.sampling_rate / target_rate) * frame_size))

        for pt in patient_generator:
            # 1. Grab patient scp label (fixed for all samples)
            with self.patient_data(pt) as h5:
                data = h5["data"][:]
                slabels = h5[label_key][:]
            # END WITH

            # 2. Map scp labels (skip patient if not in class map == -1)
            pt_lbls = []
            pt_lbl_weights = []
            for i in range(slabels.shape[0]):
                label = label_map.get(int(slabels[i, 0]), -1)
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
            pt_lbls = np.array(pt_lbls, dtype=np.int32)

            if pt_lbls.size == 0:
                continue
            # END IF

            if label_format == "multi_hot":
                y = np.zeros(num_classes, dtype=np.int32)
                y[pt_lbls] = 1
                num_samples = sum((samples_per_tgt[tgt_labels.index(i)] for i in pt_lbls))
            elif label_format == "one_hot":
                y = np.zeros(num_classes, dtype=np.int32)
                pt_lbl = random.choices(pt_lbls, pt_lbl_weights, k=1)[0]
                y[pt_lbl] = 1
                num_samples = samples_per_tgt[tgt_labels.index(pt_lbl)]
            elif label_format is None:
                # Its possible to have multiple labels, we assign based on weights
                y = random.choices(pt_lbls, pt_lbl_weights, k=1)[0]
                num_samples = samples_per_tgt[tgt_labels.index(y)]
            else:
                raise ValueError(f"Invalid label_format: {label_format}")

            # 3. Generate samples based on samples_per_tgt
            for _ in range(num_samples):
                # select random lead and start index
                lead = random.choice(self.leads)
                start = np.random.randint(0, data.shape[1] - input_size)
                x = np.nan_to_num(data[lead, start : start + input_size], posinf=0, neginf=0).astype(np.float32)
                # Resample if needed
                if self.sampling_rate != target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, target_rate, axis=0)
                    x = x[:frame_size]  # truncate to frame size
                x = np.reshape(x, (frame_size, 1))
                yield x, y
            # END FOR
        # END FOR

    def split_train_test_patients(
        self,
        patient_ids: npt.NDArray,
        test_size: float,
        label_map: dict[int, int] | None = None,
        label_type: str | None = None,
        label_threshold: int | None = 2,
    ) -> list[list[int]]:
        """Perform train/test split on patients for given task.
        NOTE: We only perform inter-patient splits and not intra-patient.

        Args:
            patient_ids (npt.NDArray): Patient Ids
            test_size (float): Test size
            label_map (dict[int, int], optional): Label map. Defaults to None.
            label_type (str, optional): Label type. Defaults to None.
            label_threshold (int, optional): Label threshold. Defaults to 2.

        Returns:
            list[list[int]]: Train and test sets of patient ids
        """
        stratify = None
        if label_map is not None and label_type is not None:
            patients_labels = self.get_patients_labels(patient_ids, label_map=label_map, label_type=label_type)
            # Select random label for stratification or -1 if no labels
            stratify = np.array([random.choice(x) if len(x) > 0 else -1 for x in patients_labels])

            # Remove patients w/ label counts below threshold
            for i, label in enumerate(sorted(set(label_map.values()))):
                class_counts = np.sum(stratify == label)
                if label_threshold is not None and class_counts < label_threshold:
                    stratify[stratify == label] = -1
                    logger.warning(f"Removed class {label} w/ only {class_counts} samples")
                # END IF
            # END FOR

            # Remove patients w/o labels
            neg_mask = stratify == -1
            stratify = stratify[~neg_mask]
            patient_ids = patient_ids[~neg_mask]

            num_neg = neg_mask.sum()
            if num_neg > 0:
                logger.debug(f"Removed {num_neg} patients w/ no target class")
            # END IF
        # END IF

        # Get occurence of each class along with class index
        if stratify is not None:
            class_counts = np.zeros(len(label_map), dtype=np.int32)
            logger.debug(f"[{self.name}] Stratify class counts:")
            for i, label in enumerate(sorted(set(label_map.values()))):
                class_counts = np.sum(stratify == label)
                logger.debug(f"Class {label}: {class_counts}")
            # END FOR
        # END IF

        return sklearn.model_selection.train_test_split(
            patient_ids,
            test_size=test_size,
            shuffle=True,
            stratify=stratify,
        )

    def filter_patients_for_labels(
        self,
        patient_ids: npt.NDArray,
        label_map: dict[int, int] | None = None,
        label_type: str | None = None,
    ) -> npt.NDArray:
        """Filter patients based on labels.
        Useful to remove patients w/o labels for task to speed up data loading.

        Args:
            patient_ids (npt.NDArray): Patient ids
            patient_ids (npt.NDArray): Patient ids
            label_map (dict[int, int], optional): Label map. Defaults to None.
            label_type (str, optional): Label type. Defaults to None.

        Returns:
            npt.NDArray: Filtered patient ids
        """
        if label_map is None or label_type is None:
            return patient_ids

        patients_labels = self.get_patients_labels(patient_ids, label_map, label_type)
        # Find any patient with empty list
        label_mask = np.array([len(x) > 0 for x in patients_labels])
        neg_mask = ~label_mask
        num_neg = neg_mask.sum()
        if num_neg > 0:
            logger.debug(f"Removed {num_neg} of {patient_ids.size} patients w/ no target class")
        return patient_ids[~neg_mask]

    def get_patients_labels(
        self,
        patient_ids: npt.NDArray,
        label_map: dict[int, int],
        label_type: str = "scp",
    ) -> list[list[int]]:
        """Get class labels for each patient

        Args:
            patient_ids (npt.NDArray): Patient ids
            label_map (dict[int, int]): Label map
            label_type (str, optional): Label type. Defaults to "scp".

        Returns:
            list[list[int]]: List of class labels per patient

        """
        ids = patient_ids.tolist()
        func = functools.partial(self.get_patient_labels, label_map=label_map, label_type=label_type)
        pts_labels = process_map(func, ids, desc=f"Sorting {self.name} labels")
        return pts_labels

    def get_patient_labels(self, patient_id: int, label_map: dict[int, int], label_type: str = "scp") -> list[int]:
        """Get class labels for patient

        Args:
            patient_id (int): Patient id
            label_map (dict[int, int]): Label map
            label_type (str, optional): Label type. Defaults to "scp".

        Returns:
            list[int]: List of class labels

        """
        with self.patient_data(patient_id) as h5:
            labels = h5[self.label_key(label_type)][:, 0]
        labels = np.unique(labels)
        labels: list[int] = [label_map[r] for r in labels if label_map.get(r, -1) != -1]
        return labels

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download dataset

        This will download preprocessed HDF5 files from S3.

        Args:
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """
        os.makedirs(self.path, exist_ok=True)
        zip_path = self.path / f"{self.name}.zip"

        did_download = helia.utils.download_s3_file(
            key=f"{self.name}/{self.name}.zip",
            dst=zip_path,
            bucket="ambiq-ai-datasets",
            checksum="size",
        )
        if did_download:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.path)

    def download_raw_dataset(self, num_workers: int | None = None, force: bool = False):
        """Downloads full dataset zipfile and converts into individial patient HDF5 files.

        Args:
            force (bool, optional): Whether to force re-download if destination exists. Defaults to False.
            num_workers (int, optional): # parallel workers. Defaults to os.cpu_count().
        """
        logger.debug("Downloading LSAD dataset")
        ds_url = (
            "https://www.physionet.org/static/published-projects/ecg-arrhythmia/"
            "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip"
        )
        ds_zip_path = self.path / "lsad.zip"
        os.makedirs(self.path, exist_ok=True)
        if os.path.exists(ds_zip_path) and not force:
            logger.warning(
                f"Zip file already exists. Please delete or set `force` flag to redownload. PATH={ds_zip_path}"
            )
        else:
            helia.utils.download_file(ds_url, ds_zip_path, progress=True)

        # 2. Extract and convert patient ECG data to H5 files
        logger.debug("Processing LSAD patient data")
        self._convert_dataset_zip_to_hdf5(
            zip_path=ds_zip_path,
            force=force,
            num_workers=num_workers,
        )
        logger.debug("Finished LSAD patient data")

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
                pt_path = self.path / f"{pt_id}.h5"
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
                    logger.debug(f"Skipping {zp_rec_name} - no Dx")
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
        # END FOR

    def close(self):
        """Close dataset"""
        self._cached_data.clear()
