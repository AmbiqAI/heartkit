import contextlib
import functools
import logging
import os
import random
from collections.abc import Iterable
from enum import IntEnum
from multiprocessing import Pool
from typing import Generator

import h5py
import numpy as np
import numpy.typing as npt
import physiokit as pk
import sklearn.model_selection
from tqdm import tqdm

from ..utils import download_file
from .dataset import HKDataset
from .defines import PatientGenerator
from .utils import download_s3_objects

logger = logging.getLogger(__name__)


class PtbxlScpCode(IntEnum):
    """PTBXL SCP codes"""

    # Diagnostic codes
    NDT = 0  # STTC, STTC
    NST_ = 1  # STTC, NST_
    DIG = 2  # STTC, STTC
    LNGQT = 3  # STTC, STTC
    NORM = 4  # NORM, NORM
    IMI = 5  # MI, IMI
    ASMI = 6  # MI, AMI
    LVH = 7  # HYP, LVH
    LAFB = 8  # CD, LAFB_LPFB
    ISC_ = 9  # STTC, ISC_
    IRBBB = 10  # CD, IRBBB
    AVB1 = 11  # CD, _AVB
    IVCD = 12  # CD, IVCD
    ISCAL = 13  # STTC, ISCA
    CRBBB = 14  # CD, CRBBB
    CLBBB = 15  # CD, CLBBB
    ILMI = 16  # MI, IMI
    LAO_LAE = 17  # HYP, AO_LAE
    AMI = 18  # MI, AMI
    ALMI = 19  # MI, AMI
    ISCIN = 20  # STTC, ISCI
    INJAS = 21  # MI, AMI
    LMI = 22  # MI, LMI
    ISCIL = 23  # STTC, ISCI
    LPFB = 24  # CD, LAFB_LPFB
    ISCAS = 25  # STTC, ISCA
    INJAL = 26  # MI, AMI
    ISCLA = 27  # STTC, ISCA
    RVH = 28  # HYP, RVH
    ANEUR = 29  # STTC, STTC
    RAO_RAE = 30  # HYP, RAO_RAE
    EL = 31  # STTC, STTC
    WPW = 32  # CD, WPW
    ILBBB = 33  # CD, ILBBB
    IPLMI = 34  # MI, IMI
    ISCAN = 35  # STTC, ISCA
    IPMI = 36  # MI, IMI
    SEHYP = 37  # HYP, SEHYP
    INJIN = 38  # MI, IMI
    INJLA = 39  # MI, AMI
    PMI = 40  # MI, PMI
    AVB3 = 41  # CD, _AVB
    INJIL = 42  # MI, IMI
    AVB2 = 43  # CD, _AVB
    # Form codes
    ABQRS = 44
    PVC = 45
    STD_ = 46
    VCLVH = 47
    QWAVE = 48
    LOWT = 49
    NT_ = 50
    PAC = 51
    LPR = 52
    INVT = 53
    LVOLT = 54
    HVOLT = 55
    TAB_ = 56
    STE_ = 57
    PRC_S = 58
    # Rhythm codes
    SR = 59
    AFIB = 60
    STACH = 61
    SARRH = 62
    SBRAD = 63
    PACE = 64
    SVARR = 65
    BIGU = 66
    AFLT = 67
    SVTAC = 68
    PSVT = 69
    TRIGU = 70


##
# These map PTBXL specific labels to common labels
##

PtbxlScpRawMap = {
    "NDT": PtbxlScpCode.NDT,
    "NST_": PtbxlScpCode.NST_,
    "DIG": PtbxlScpCode.DIG,
    "LNGQT": PtbxlScpCode.LNGQT,
    "NORM": PtbxlScpCode.NORM,
    "IMI": PtbxlScpCode.IMI,
    "ASMI": PtbxlScpCode.ASMI,
    "LVH": PtbxlScpCode.LVH,
    "LAFB": PtbxlScpCode.LAFB,
    "ISC_": PtbxlScpCode.ISC_,
    "IRBBB": PtbxlScpCode.IRBBB,
    "1AVB": PtbxlScpCode.AVB1,
    "IVCD": PtbxlScpCode.IVCD,
    "ISCAL": PtbxlScpCode.ISCAL,
    "CRBBB": PtbxlScpCode.CRBBB,
    "CLBBB": PtbxlScpCode.CLBBB,
    "ILMI": PtbxlScpCode.ILMI,
    "LAO/LAE": PtbxlScpCode.LAO_LAE,
    "AMI": PtbxlScpCode.AMI,
    "ALMI": PtbxlScpCode.ALMI,
    "ISCIN": PtbxlScpCode.ISCIN,
    "INJAS": PtbxlScpCode.INJAS,
    "LMI": PtbxlScpCode.LMI,
    "ISCIL": PtbxlScpCode.ISCIL,
    "LPFB": PtbxlScpCode.LPFB,
    "ISCAS": PtbxlScpCode.ISCAS,
    "INJAL": PtbxlScpCode.INJAL,
    "ISCLA": PtbxlScpCode.ISCLA,
    "RVH": PtbxlScpCode.RVH,
    "ANEUR": PtbxlScpCode.ANEUR,
    "RAO/RAE": PtbxlScpCode.RAO_RAE,
    "EL": PtbxlScpCode.EL,
    "WPW": PtbxlScpCode.WPW,
    "ILBBB": PtbxlScpCode.ILBBB,
    "IPLMI": PtbxlScpCode.IPLMI,
    "ISCAN": PtbxlScpCode.ISCAN,
    "IPMI": PtbxlScpCode.IPMI,
    "SEHYP": PtbxlScpCode.SEHYP,
    "INJIN": PtbxlScpCode.INJIN,
    "INJLA": PtbxlScpCode.INJLA,
    "PMI": PtbxlScpCode.PMI,
    "3AVB": PtbxlScpCode.AVB3,
    "INJIL": PtbxlScpCode.INJIL,
    "2AVB": PtbxlScpCode.AVB2,
    "ABQRS": PtbxlScpCode.ABQRS,
    "PVC": PtbxlScpCode.PVC,
    "STD_": PtbxlScpCode.STD_,
    "VCLVH": PtbxlScpCode.VCLVH,
    "QWAVE": PtbxlScpCode.QWAVE,
    "LOWT": PtbxlScpCode.LOWT,
    "NT_": PtbxlScpCode.NT_,
    "PAC": PtbxlScpCode.PAC,
    "LPR": PtbxlScpCode.LPR,
    "INVT": PtbxlScpCode.INVT,
    "LVOLT": PtbxlScpCode.LVOLT,
    "HVOLT": PtbxlScpCode.HVOLT,
    "TAB_": PtbxlScpCode.TAB_,
    "STE_": PtbxlScpCode.STE_,
    "PRC(S)": PtbxlScpCode.PRC_S,
    "SR": PtbxlScpCode.SR,
    "AFIB": PtbxlScpCode.AFIB,
    "STACH": PtbxlScpCode.STACH,
    "SARRH": PtbxlScpCode.SARRH,
    "SBRAD": PtbxlScpCode.SBRAD,
    "PACE": PtbxlScpCode.PACE,
    "SVARR": PtbxlScpCode.SVARR,
    "BIGU": PtbxlScpCode.BIGU,
    "AFLT": PtbxlScpCode.AFLT,
    "SVTAC": PtbxlScpCode.SVTAC,
    "PSVT": PtbxlScpCode.PSVT,
    "TRIGU": PtbxlScpCode.TRIGU,
}


PtbxlLeadsMap = {
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


class PtbxlDataset(HKDataset):
    """PTBXL dataset"""

    def __init__(
        self,
        ds_path: os.PathLike,
        leads: list[int] | None = None,
    ) -> None:
        super().__init__(
            ds_path=ds_path,
        )
        self.leads = leads or list(range(12))
        self._data_cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        """Dataset name"""
        return "ptbxl"

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

    @property
    def patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        pts = np.arange(1, 21838)
        rm_pts = np.array(
            [
                137,
                139,
                140,
                141,
                142,
                143,
                145,
                456,
                458,
                459,
                461,
                462,
                2506,
                2511,
                3795,
                3798,
                3800,
                3832,
                5817,
                7777,
                7779,
                7782,
                9821,
                9825,
                9888,
                11810,
                11814,
                11817,
                11838,
                13791,
                13793,
                13796,
                13797,
                13799,
                15742,
                18150,
            ],
            dtype=np.int32,
        )
        return pts[~np.in1d(pts, rm_pts)]

    def get_train_patient_ids(self) -> npt.NDArray:
        """Get dataset training patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        return self.patient_ids[:18500]

    def get_test_patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs reserved for testing only

        Returns:
            npt.NDArray: patient IDs
        """
        return self.patient_ids[18500:]

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
        if label_type == "beat":
            return "blabels"
        raise ValueError(f"Invalid label type: {label_type}")

    @contextlib.contextmanager
    def patient_data(self, patient_id: int) -> Generator[h5py.Group, None, None]:
        """Get patient data

        Args:
            patient_id (int): Patient ID

        Returns:
            Generator[h5py.Group, None, None]: Patient data
        """
        with h5py.File(self.ds_path / f"{self._pt_key(patient_id)}.h5", mode="r") as h5:
            yield h5

    def signal_generator(
        self,
        patient_generator: PatientGenerator,
        frame_size: int,
        samples_per_patient: int = 1,
        target_rate: int | None = None,
    ) -> Generator[npt.NDArray, None, None]:
        """Generate random frames.

        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
                    Patient data may contain only signals, since labels are not used.
            samples_per_patient (int): Samples per patient.

        Returns:
            Generator[npt.NDArray, None, None]: Generator of input data of shape (frame_size, 1)
        """
        if target_rate is None:
            target_rate = self.sampling_rate

        input_size = int(np.round((self.sampling_rate / target_rate) * frame_size))

        for pt in patient_generator:
            with self.patient_data(pt) as h5:
                data: h5py.Dataset = h5["data"][:]
            # END WITH
            for _ in range(samples_per_patient):
                lead = random.choice(self.leads)
                start = np.random.randint(0, data.shape[1] - input_size)
                x = data[lead, start : start + input_size].squeeze()
                x = np.nan_to_num(x).astype(np.float32)
                if self.sampling_rate != target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, target_rate, axis=0)
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
    ) -> Generator[tuple[npt.NDArray, int], None, None]:
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
            Generator[tuple[npt.NDArray, int], None, None]: Generator of input data and labels

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
        # END IF

        input_size = int(np.round((self.sampling_rate / target_rate) * frame_size))

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
                # y = np.expand_dims(y, axis=0)
                num_samples = sum((samples_per_tgt[tgt_labels.index(i)] for i in pt_lbls))
            elif label_format == "one_hot":
                raise NotImplementedError()
            elif label_format is None:
                # Its possible to have multiple labels, we assign based on weights
                y = random.choices(pt_lbls, pt_lbl_weights, k=1)[0]
                num_samples = samples_per_tgt[tgt_labels.index(y)]
            else:
                raise ValueError(f"Invalid label_format: {label_format}")

            # 3. Generate samples based on samples_per_tgt

            # print(f'{pt} creating {num_samples} samples')
            for _ in range(num_samples):
                # select random lead and start index
                lead = random.choice(self.leads)
                # lead = self.leads
                start = np.random.randint(0, data.shape[1] - input_size)
                # Extract frame
                x = np.nan_to_num(data[lead, start : start + input_size], posinf=0, neginf=0).astype(np.float32)
                # Resample if needed
                if self.sampling_rate != target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, target_rate, axis=0)
                yield x, y
            # END FOR
        # END FOR

    def split_train_test_patients(
        self,
        patient_ids: npt.NDArray,
        test_size: float,
        label_map: dict[int, int] | None = None,
        label_type: str | None = None,
    ) -> list[list[int]]:
        """Perform train/test split on patients for given task.
        NOTE: We only perform inter-patient splits and not intra-patient.

        Args:
            patient_ids (npt.NDArray): Patient Ids
            test_size (float): Test size
            label_map (dict[int, int], optional): Label map. Defaults to None.
            label_type (str, optional): Label type. Defaults to None.

        Returns:
            list[list[int]]: Train and test sets of patient ids
        """
        stratify = None
        if label_map is not None and label_type is not None:
            patients_labels = self.get_patients_labels(patient_ids, label_map=label_map, label_type=label_type)
            # Select random label for stratification or -1 if no labels
            stratify = np.array([random.choice(x) if len(x) > 0 else -1 for x in patients_labels])
            # Remove patients w/o labels
            neg_mask = stratify == -1
            stratify = stratify[~neg_mask]
            patient_ids = patient_ids[~neg_mask]
            num_neg = neg_mask.sum()
            if num_neg > 0:
                logger.warning(f"Removed {num_neg} patients w/ no target class")
            # END IF
        # END IF

        return sklearn.model_selection.train_test_split(
            patient_ids,
            test_size=test_size,
            shuffle=True,
            stratify=stratify,
        )

    def filter_patients_for_labels(
        self, patient_ids: npt.NDArray, label_map: dict[int, int] | None = None, label_type: str | None = None
    ) -> npt.NDArray:
        """Filter patients based on labels.
        Useful to remove patients w/o labels for task to speed up data loading.

        Args:
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
        neg_mask = label_mask == -1
        num_neg = neg_mask.sum()
        if num_neg > 0:
            logger.warning(f"Removed {num_neg} of {patient_ids.size} patients w/ no target class")
        return patient_ids[~neg_mask]

    def get_patients_labels(
        self, patient_ids: npt.NDArray, label_map: dict[int, int], label_type: str = "scp"
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
        with Pool() as pool:
            pts_labels = list(pool.imap(func, ids))
        return pts_labels

    def get_patient_scp_codes(self, patient_id: int) -> list[int]:
        """Get SCP codes for patient

        Args:
            patient_id (int): Patient id

        Returns:
            list[int]: List of SCP codes

        """
        with self.patient_data(patient_id) as h5:
            codes = h5[self.label_key("scp")][:, 0]
        return np.unique(codes).tolist()

    def get_patient_labels(self, patient_id: int, label_map: dict[int, int], label_type: str = "scp") -> list[int]:
        """Get class labels for patient

        Args:
            patient_id (int): Patient id

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
        download_s3_objects(
            bucket="ambiq-ai-datasets",
            prefix=self.ds_path.stem,
            dst=self.ds_path.parent,
            checksum="size",
            progress=True,
            num_workers=num_workers,
        )

    def download_raw_dataset(self, num_workers: int | None = None, force: bool = False):
        """Downloads full dataset zipfile and converts into individial patient HDF5 files.

        Args:
            force (bool, optional): Whether to force re-download if destination exists. Defaults to False.
            num_workers (int, optional): # parallel workers. Defaults to os.cpu_count().
        """
        logger.info("Downloading PTB-XL dataset")
        ds_url = (
            "https://www.physionet.org/static/published-projects/ptb-xl/"
            "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2.zip"
        )
        ds_zip_path = self.ds_path / "ptbxl.zip"
        os.makedirs(self.ds_path, exist_ok=True)
        if os.path.exists(ds_zip_path) and not force:
            logger.warning(
                f"Zip file already exists. Please delete or set `force` flag to redownload. PATH={ds_zip_path}"
            )
        else:
            download_file(ds_url, ds_zip_path, progress=True)

        # 2. Extract and convert patient ECG data to H5 files
        logger.info("Processing PTB-XL patient data")
        self._convert_dataset_zip_to_hdf5(
            zip_path=ds_zip_path,
            force=force,
            num_workers=num_workers,
        )
        logger.info("Finished PTB-XL patient data")

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

        import ast  # pylint: disable=import-outside-toplevel
        import io  # pylint: disable=import-outside-toplevel
        import re  # pylint: disable=import-outside-toplevel
        import tempfile  # pylint: disable=import-outside-toplevel
        import zipfile  # pylint: disable=import-outside-toplevel

        import pandas as pd  # pylint: disable=import-outside-toplevel
        import wfdb  # pylint: disable=import-outside-toplevel

        if not patient_ids:
            patient_ids = self.patient_ids

        zp = zipfile.ZipFile(zip_path, mode="r")  # pylint: disable=consider-using-with

        zp_root = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2"

        # scp_df = pd.read_csv(io.BytesIO(zp.read(os.path.join(zp_root, "scp_statements.csv"))))
        with open(self.ds_path / "scp_statements.csv", "wb") as fp:
            fp.write(zp.read(os.path.join(zp_root, "scp_statements.csv")))

        db_df = pd.read_csv(io.BytesIO(zp.read(os.path.join(zp_root, "ptbxl_database.csv"))))

        # # Get unique diagnostic classes
        # diag_class = list(set(scp_df.diagnostic_class.to_list()))
        # list(filter(lambda v: isinstance(v, str) or not math.isnan(v), diag_class))
        for patient in tqdm(patient_ids, desc="Converting"):
            # logger.info(f"Processing patient {patient}")
            pt_id = self._pt_key(patient)
            pt_path = self.ds_path / f"{pt_id}.h5"

            pt_info = db_df[db_df.ecg_id == patient]
            if len(pt_info) == 0:
                logger.warning(f"Patient {patient} not found in database. Skipping.")
                continue
            pt_info = pt_info.iloc[0].to_dict()

            # Get r-peaks and scp codes
            rpeaks = np.array([int(x) for x in re.findall(r"\d+", pt_info["r_peaks"])])
            scp_codes = ast.literal_eval(pt_info["scp_codes"])

            # Get scp labels
            slabels = []
            for k, v in scp_codes.items():
                if k not in PtbxlScpRawMap:
                    logger.warning(f"Unknown SCP code {k} for patient {patient}")
                slabels.append([PtbxlScpRawMap[k], v])
            slabels = np.array(slabels, dtype=np.float32)

            # Get beat labels
            blabels = np.array([[i, 1] for i in rpeaks])

            # Grab ECG data
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(zp_root, "records500", f"{1000*(patient//1000):05d}", f"{pt_id}_hr")
                rec_fpath = os.path.join(tmpdir, f"{pt_id}_hr")
                with open(f"{rec_fpath}.hea", "wb") as fp:
                    fp.write(zp.read(f"{zip_path}.hea"))
                with open(f"{rec_fpath}.dat", "wb") as fp:
                    fp.write(zp.read(f"{zip_path}.dat"))
                rec = wfdb.rdrecord(rec_fpath)
                data = rec.p_signal.astype(np.float32).T
            # END WITH

            # Save patient data to HDF5
            with h5py.File(pt_path, mode="w") as h5:
                h5.create_dataset(
                    name="/data",
                    data=data,
                    compression="gzip",
                    compression_opts=6,
                )
                h5.create_dataset(name="/blabels", data=blabels)
                h5.create_dataset(name="/slabels", data=slabels)
                # Add patient info as attributes
                for k, v in pt_info.items():
                    h5.attrs[k] = v
                # END FOR
            # END WITH
