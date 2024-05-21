import contextlib
import functools
import logging
import os
import random
import tempfile
import zipfile
from enum import IntEnum
from multiprocessing import Pool
from typing import Generator

import h5py
import numpy as np
import numpy.typing as npt
import physiokit as pk
import sklearn.model_selection
import sklearn.preprocessing
from tqdm import tqdm

from ..utils import download_file
from .dataset import HKDataset
from .defines import PatientGenerator
from .utils import download_s3_objects

logger = logging.getLogger(__name__)


class IcentiaRhythm(IntEnum):
    """Icentia rhythm labels"""

    noise = 0
    normal = 1
    afib = 2
    aflut = 3
    end = 4


class IcentiaBeat(IntEnum):
    """Incentia beat labels"""

    undefined = 0
    normal = 1
    pac = 2
    aberrated = 3
    pvc = 4


IcentiaLeadsMap = {
    "i": 0,  # Modified lead I
}


class IcentiaDataset(HKDataset):
    """Icentia dataset"""

    def __init__(
        self,
        ds_path: os.PathLike,
        leads: list[int] | None = None,
    ) -> None:
        super().__init__(ds_path=ds_path)
        self.leads = leads or list(IcentiaLeadsMap.values())

    @property
    def name(self) -> str:
        """Dataset name"""
        return "icentia11k"

    @property
    def sampling_rate(self) -> int:
        """Sampling rate in Hz"""
        return 250

    @property
    def mean(self) -> float:
        """Dataset mean"""
        return 0.0018

    @property
    def std(self) -> float:
        """Dataset st dev"""
        return 1.3711

    @property
    def patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        return np.arange(11_000)

    def get_train_patient_ids(self) -> npt.NDArray:
        """Get training patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        return self.patient_ids[:10_000]

    def get_test_patient_ids(self) -> npt.NDArray:
        """Get patient IDs reserved for testing only

        Returns:
            npt.NDArray: patient IDs
        """
        return self.patient_ids[10_000:]

    def _pt_key(self, patient_id: int):
        return f"p{patient_id:05d}"

    def label_key(self, label_type: str = "rhythm") -> str:
        """Get label key

        Args:
            label_type (str, optional): Label type. Defaults to "rhythm".

        Returns:
            str: Label key
        """
        if label_type == "rhythm":
            return "rlabels"
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
            yield h5[self._pt_key(patient_id)]

    def signal_generator(
        self,
        patient_generator: PatientGenerator,
        frame_size: int,
        samples_per_patient: int = 1,
        target_rate: int | None = None,
    ) -> Generator[npt.NDArray, None, None]:
        """Generate random frames.

        Args:
            patient_generator (PatientGenerator): Patient generator
            frame_size (int): Frame size
            samples_per_patient (int, optional): Samples per patient. Defaults to 1.
            target_rate (int | None, optional): Target rate. Defaults to None.

        Returns:
            SampleGenerator: Generator of input data of shape (frame_size, 1)
        """
        if target_rate is None:
            target_rate = self.sampling_rate

        input_size = int(np.round((self.sampling_rate / target_rate) * frame_size))
        for pt in patient_generator:
            with self.patient_data(pt) as segments:
                for _ in range(samples_per_patient):
                    segment = segments[np.random.choice(list(segments.keys()))]
                    segment_size = segment["data"].shape[0]
                    frame_start = np.random.randint(segment_size - input_size)
                    frame_end = frame_start + input_size
                    x = segment["data"][frame_start:frame_end].squeeze()
                    x = np.nan_to_num(x).astype(np.float32)
                    if self.sampling_rate != target_rate:
                        x = pk.signal.resample_signal(x, self.sampling_rate, target_rate, axis=0)
                    # END IF
                    yield x
                # END FOR
            # END WITH
        # END FOR

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

    def split_train_test_patients(
        self,
        patient_ids: npt.NDArray,
        test_size: float,
        label_map: dict[int, int] | None = None,
        label_type: str | None = None,
    ) -> list[list[int]]:
        """Perform train/test split on patients for given task.

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
            # Use stratified split for rhythm task
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
        self, patient_ids: npt.NDArray, label_map: dict[int, int], label_type: str = "rhythm"
    ) -> list[list[int]]:
        """Get class labels for each patient

        Args:
            patient_ids (npt.NDArray): Patient ids
            label_map (dict[int, int]): Label map
            label_type (str, optional): Label type. Defaults to "rhythm".

        Returns:
            list[list[int]]: List of class labels per patient

        """
        ids = patient_ids.tolist()
        func = functools.partial(self.get_patient_labels, label_map=label_map, label_type=label_type)
        with Pool() as pool:
            pts_labels = list(pool.imap(func, ids))
        return pts_labels

    def get_patient_labels(self, patient_id: int, label_map: dict[int, int], label_type: str = "rhythm") -> list[int]:
        """Get class labels for patient

        Args:
            patient_id (int): Patient id
            label_map (dict[int, int]): Label map
            label_type (str, optional): Label type. Defaults to "rhythm".

        Returns:
            list[int]: List of class labels

        """
        label_key = self.label_key(label_type)
        labels = set()
        with self.patient_data(patient_id) as h5:
            for _, segment in h5.items():
                slabels = segment[label_key][:]
                if not slabels.shape[0]:
                    continue
                slabels = slabels[:, 1]
                slabels = np.unique(slabels)
                slabels: list[int] = [label_map[l] for l in slabels if label_map.get(l, -1) != -1]
                labels.update(labels, slabels)
            # END FOR
        # END WITH
        return list(labels)

    def download_raw_dataset(self, num_workers: int | None = None, force: bool = False):
        """Downloads full Icentia dataset zipfile and converts into individial patient HDF5 files.
        NOTE: This is a very long process (e.g. 24 hrs). Please use `icentia11k.download_dataset` instead.

        Args:
            force (bool, optional): Whether to force re-download if destination exists. Defaults to False.
            num_workers (int, optional): # parallel workers. Defaults to os.cpu_count().
        """
        logger.info("Downloading icentia11k dataset")
        ds_url = (
            "https://physionet.org/static/published-projects/icentia11k-continuous-ecg/"
            "icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0.zip"
        )
        ds_zip_path = self.ds_path / "icentia11k.zip"
        os.makedirs(self.ds_path, exist_ok=True)
        if os.path.exists(ds_zip_path) and not force:
            logger.warning(
                f"Zip file already exists. Please delete or set `force` flag to redownload. PATH={ds_zip_path}"
            )
        else:
            download_file(ds_url, ds_zip_path, progress=True)

        # 2. Extract and convert patient ECG data to H5 files
        logger.info("Generating icentia11k patient data")
        self._convert_dataset_zip_to_hdf5(
            zip_path=ds_zip_path,
            force=force,
            num_workers=num_workers,
        )
        logger.info("Finished icentia11k patient data")

    def _convert_dataset_pt_zip_to_hdf5(self, patient: int, zip_path: os.PathLike, force: bool = False):
        """Extract patient data from Icentia zipfile. Pulls out ECG data along with all labels.

        Args:
            patient (int): Patient id
            zip_path (PathLike): Zipfile path
            force (bool, optional): Whether to override destination if it exists. Defaults to False.
        """
        import re  # pylint: disable=import-outside-toplevel

        import wfdb  # pylint: disable=import-outside-toplevel

        # These map Wfdb labels to icentia labels
        WfdbRhythmMap = {
            "": IcentiaRhythm.noise.value,
            "(N": IcentiaRhythm.normal.value,
            "(AFIB": IcentiaRhythm.afib.value,
            "(AFL": IcentiaRhythm.aflut.value,
            ")": IcentiaRhythm.end.value,
        }
        WfdbBeatMap = {
            "Q": IcentiaBeat.undefined.value,
            "N": IcentiaBeat.normal.value,
            "S": IcentiaBeat.pac.value,
            "a": IcentiaBeat.aberrated.value,
            "V": IcentiaBeat.pvc.value,
        }

        logger.info(f"Processing patient {patient}")
        pt_id = self._pt_key(patient)
        pt_path = self.ds_path / f"{pt_id}.h5"
        if not force and os.path.exists(pt_path):
            logger.debug(f"Skipping patient {pt_id}")
            return
        zp = zipfile.ZipFile(zip_path, mode="r")  # pylint: disable=consider-using-with
        h5 = h5py.File(pt_path, mode="w")

        # Find all patient .dat file indices
        zp_rec_names = filter(
            lambda f: re.match(f"{pt_id}_[A-z0-9]+.dat", os.path.basename(f)),
            (f.filename for f in zp.filelist),
        )
        for zp_rec_name in zp_rec_names:
            try:
                zp_hdr_name = zp_rec_name.replace(".dat", ".hea")
                zp_atr_name = zp_rec_name.replace(".dat", ".atr")

                with tempfile.TemporaryDirectory() as tmpdir:
                    rec_fpath = os.path.join(tmpdir, os.path.basename(zp_rec_name))
                    atr_fpath = rec_fpath.replace(".dat", ".atr")
                    hdr_fpath = rec_fpath.replace(".dat", ".hea")
                    with open(hdr_fpath, "wb") as fp:
                        fp.write(zp.read(zp_hdr_name))
                    with open(rec_fpath, "wb") as fp:
                        fp.write(zp.read(zp_rec_name))
                    with open(atr_fpath, "wb") as fp:
                        fp.write(zp.read(zp_atr_name))
                    rec = wfdb.rdrecord(os.path.splitext(rec_fpath)[0], physical=True)
                    atr = wfdb.rdann(os.path.splitext(atr_fpath)[0], extension="atr")
                pt_seg_path = f"/{os.path.splitext(os.path.basename(zp_rec_name))[0].replace('_', '/')}"
                data = rec.p_signal.astype(np.float16)
                blabels = np.array(
                    [[atr.sample[i], WfdbBeatMap.get(s)] for i, s in enumerate(atr.symbol) if s in WfdbBeatMap],
                    dtype=np.int32,
                )
                rlabels = np.array(
                    [
                        [atr.sample[i], WfdbRhythmMap.get(atr.aux_note[i], 0)]
                        for i, s in enumerate(atr.symbol)
                        if s == "+"
                    ],
                    dtype=np.int32,
                )
                h5.create_dataset(
                    name=f"{pt_seg_path}/data",
                    data=data,
                    compression="gzip",
                    compression_opts=3,
                )
                h5.create_dataset(name=f"{pt_seg_path}/blabels", data=blabels)
                h5.create_dataset(name=f"{pt_seg_path}/rlabels", data=rlabels)
            except Exception as err:  # pylint: disable=broad-except
                logger.warning(f"Failed processing {zp_rec_name}", err)
                continue
        h5.close()

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
        if not patient_ids:
            patient_ids = self.patient_ids
        f = functools.partial(self._convert_dataset_pt_zip_to_hdf5, zip_path=zip_path, force=force)
        with Pool(processes=num_workers) as pool:
            _ = list(tqdm(pool.imap(f, patient_ids), total=len(patient_ids)))
