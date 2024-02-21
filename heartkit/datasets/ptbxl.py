import logging
import os
import random
from collections.abc import Iterable
from enum import IntEnum

import h5py
import numpy as np
import numpy.typing as npt
import physiokit as pk
import tensorflow as tf
from tqdm import tqdm

from ..tasks import HeartRhythm
from ..utils import download_file
from .dataset import HKDataset
from .defines import PatientGenerator, SampleGenerator

logger = logging.getLogger(__name__)


class PtbxlRhythm(IntEnum):
    """PTBXL rhythm labels"""

    SR = 0  # Sinus rhythm (normal)
    AFIB = 1  # Atrial fibrillation (irregular rhythm, no p-waves, rapid ventricular response)
    STACH = 2  # Sinus tachycardia (fast normal rhythm)
    SARRH = 3  # Sinus arrhythmia (RR variation > 0.12s, normal rhythm)
    SBRAD = 4  # Sinus bradycardia (slow normal rhythm)
    PACE = 5  # Paced rhythm (artificial pacemaker rhythm)
    SVARR = 6  # Supraventricular arrhythmia (includes atrial flutter, atrial tachycardia, and atrial fibrillation)
    BIGU = 7  # Bigeminy (every other beat is PVC)
    AFLT = 8  # Atrial flutter (regular atrial rhythm at 250-350 bpm, sawtooth pattern)
    SVTAC = 9  # Supraventricular tachycardia (fast atrial rhythm, 150-250 bpm)
    PSVT = 10  # Paroxysmal supraventricular tachycardia (sudden onset of SVT)
    TRIGU = 11  # Trigeminy (every third beat is PVC)


##
# These map PTBXL specific labels to common labels
##
HeartRhythmMap = {
    PtbxlRhythm.SR: HeartRhythm.normal,
    PtbxlRhythm.AFIB: HeartRhythm.afib,
    PtbxlRhythm.AFLT: HeartRhythm.aflut,
    PtbxlRhythm.STACH: HeartRhythm.stach,
    PtbxlRhythm.SBRAD: HeartRhythm.sbrad,
    PtbxlRhythm.SARRH: HeartRhythm.sarrh,
    PtbxlRhythm.SVARR: HeartRhythm.svarr,
    PtbxlRhythm.SVTAC: HeartRhythm.svt,
    PtbxlRhythm.PSVT: HeartRhythm.svt,
    PtbxlRhythm.BIGU: HeartRhythm.bigu,
    PtbxlRhythm.TRIGU: HeartRhythm.trigu,
    PtbxlRhythm.PACE: HeartRhythm.pace,
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
        task: str,
        frame_size: int,
        target_rate: int,
        spec: tuple[tf.TensorSpec, tf.TensorSpec],
        class_map: dict[int, int] | None = None,
    ) -> None:
        super().__init__(
            ds_path=ds_path / "ptbxl",
            task=task,
            frame_size=frame_size,
            target_rate=target_rate,
            spec=spec,
            class_map=class_map,
        )

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
        if self.task == "arrhythmia":
            return self.rhythm_data_generator(
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
                lead = np.random.randint(0, data.shape[0])
                start = np.random.randint(0, data.shape[1] - input_size)
                x = data[lead, start : start + input_size].squeeze()
                x = np.nan_to_num(x).astype(np.float32)
                if self.sampling_rate != self.target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                # END IF
                yield x
            # END FOR
        # END FOR

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
        # Target labels and mapping
        tgt_labels = list(set(self.class_map.values()))

        # Convert dataset labels -> HK labels -> class map labels (-1 indicates not in class map)
        tgt_map = {k: self.class_map.get(v, -1) for (k, v) in HeartRhythmMap.items()}
        num_classes = len(tgt_labels)

        # If samples_per_patient is a list, then it must be the same length as nclasses
        if isinstance(samples_per_patient, Iterable):
            samples_per_tgt = samples_per_patient
        else:
            num_per_tgt = int(max(1, samples_per_patient / num_classes))
            samples_per_tgt = num_per_tgt * [num_classes]

        input_size = int(np.round((self.sampling_rate / self.target_rate) * self.frame_size))

        for _, seg in patient_generator:
            # pt_info = {k:v for k,v in seg.attrs.items()}
            # 1. Grab patient rhythm label (fixed for all samples)
            rlabels = seg["rlabels"][:]

            # 2. Map rhythm labels (skip patient if not in class map == -1)
            pt_lbls = []
            pt_lbl_weights = []
            for i in range(rlabels.shape[0]):
                label = tgt_map.get(int(rlabels[i, 0]), -1)
                if label == -1:
                    continue
                # END IF
                if label not in pt_lbls:
                    pt_lbls.append(label)
                    pt_lbl_weights.append(1 + rlabels[i, 1])
                else:
                    i = pt_lbls.index(label)
                    pt_lbl_weights[i] += rlabels[i, 1]
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
            for _ in range(num_samples):
                # select random lead and start index
                lead = np.random.randint(0, data.shape[0])
                start = np.random.randint(0, data.shape[1] - input_size)
                # Extract frame
                x = data[lead, start : start + input_size]
                # Resample if needed
                if self.sampling_rate != self.target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, self.target_rate, axis=0)
                yield x, y
            # END FOR
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
        import tempfile  # pylint: disable=import-outside-toplevel
        import zipfile  # pylint: disable=import-outside-toplevel

        import pandas as pd  # pylint: disable=import-outside-toplevel
        import wfdb  # pylint: disable=import-outside-toplevel

        if not patient_ids:
            patient_ids = self.patient_ids

        zp = zipfile.ZipFile(zip_path, mode="r")  # pylint: disable=consider-using-with

        rhythm_scp_keys = [r.name for r in PtbxlRhythm]
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
            rpeaks = np.genfromtxt(io.StringIO(pt_info["r_peaks"].replace("\n", " ")), autostrip=True)
            rpeaks = rpeaks[~np.isnan(rpeaks)].astype(np.int32)
            scp_codes = ast.literal_eval(pt_info["scp_codes"])

            # Get rhythm labels
            rlabels = []
            for k, v in scp_codes.items():
                if k in rhythm_scp_keys:
                    rlabels.append([PtbxlRhythm[k].value, v])
            rlabels = np.array(rlabels, dtype=np.float32)

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
                h5.create_dataset(name="/rlabels", data=rlabels)
                # Add patient info as attributes
                for k, v in pt_info.items():
                    h5.attrs[k] = v
                # END FOR
            # END WITH
