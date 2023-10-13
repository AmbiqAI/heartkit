import functools
import logging
import os
import tempfile
import zipfile
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import physiokit as pk
from tqdm import tqdm

from ..defines import HeartTask
from ..utils import download_file
from .dataset import HeartKitDataset
from .defines import PatientGenerator, SampleGenerator

logger = logging.getLogger(__name__)

LudbSymbolMap = {
    "o": 0,  # Other
    "p": 1,  # P Wave
    "N": 2,  # QRS complex
    "t": 3,  # T Wave
}
LudbLeadsMap = {
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

FID_LEAD_IDX = 0
FID_LBL_IDX = 1
FID_LOC_IDX = 2
SEG_LEAD_IDX = 0
SEG_LBL_IDX = 1
SEG_BEG_IDX = 2
SEG_END_IDX = 3


class LudbDataset(HeartKitDataset):
    """LUDB dataset"""

    def __init__(
        self,
        ds_path: os.PathLike,
        task: HeartTask = HeartTask.arrhythmia,
        frame_size: int = 1250,
        target_rate: int = 250,
    ) -> None:
        super().__init__(ds_path / "ludb", task, frame_size, target_rate)

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
        return np.arange(1, 201)

    def get_train_patient_ids(self) -> npt.NDArray:
        """Get dataset training patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        return self.patient_ids[:180]

    def get_test_patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs reserved for testing only

        Returns:
            npt.NDArray: patient IDs
        """
        return self.patient_ids[180:]

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
        if self.task == HeartTask.segmentation:
            return self.segmentation_generator(
                patient_generator=patient_generator,
                samples_per_patient=samples_per_patient,
            )
        raise NotImplementedError()

    def segmentation_generator(
        self,
        patient_generator: PatientGenerator,
        samples_per_patient: int | list[int] = 1,
    ) -> SampleGenerator:
        """Generate frames and segment labels.

        Args:
            patient_generator (PatientGenerator): Patient Generator
            samples_per_patient (int | list[int], optional): # samples per patient. Defaults to 1.
        Returns:
            SampleGenerator: Sample generator
        Yields:
            Iterator[SampleGenerator]
        """
        for _, pt in patient_generator:
            # NOTE: [:] will load all data into RAM- ideal for small dataset
            data = pt["data"][:]
            segs = pt["segmentations"][:]
            fids = pt["fiducials"][:]

            if self.sampling_rate != self.target_rate:
                ratio = self.target_rate / self.sampling_rate
                data = pk.signal.resample_signal(data, self.sampling_rate, self.target_rate, axis=0)
                segs[:, (SEG_BEG_IDX, SEG_END_IDX)] = segs[:, (SEG_BEG_IDX, SEG_END_IDX)] * ratio
                fids[:, FID_LOC_IDX] = fids[:, FID_LOC_IDX] * ratio
            # END IF

            # Create segmentation mask
            labels = np.zeros_like(data)
            for seg_idx in range(segs.shape[0]):
                seg = segs[seg_idx]
                labels[seg[SEG_BEG_IDX] : seg[SEG_END_IDX], seg[SEG_LEAD_IDX]] = seg[SEG_LBL_IDX]
            # END FOR

            start_offset = max(0, segs[0][SEG_BEG_IDX] - 100)
            stop_offset = max(0, data.shape[0] - segs[-1][SEG_END_IDX] + 100)
            for _ in range(samples_per_patient):
                # Randomly pick an ECG lead
                lead_idx = np.random.randint(data.shape[1])
                # Randomly select frame within the segment
                frame_start = np.random.randint(start_offset, data.shape[0] - self.frame_size - stop_offset)
                frame_end = frame_start + self.frame_size
                x = data[frame_start:frame_end, lead_idx].astype(np.float32)
                y = labels[frame_start:frame_end, lead_idx].astype(np.int32)
                yield x, y
            # END FOR

            # start_offset = max(segs[0][SEG_BEG_IDX], int(0.55 * self.frame_size))
            # stop_offset = int(data.shape[0] - 0.55 * self.frame_size)
            # # Identify R peak locations and randomly shuffle
            # rfids = fids[
            #     (fids[:, FID_LBL_IDX] == 2)
            #     & (start_offset < fids[:, FID_LOC_IDX])
            #     & (fids[:, FID_LOC_IDX] < stop_offset)
            # ]
            # if rfids.shape[0] <= 2:
            #     continue
            # np.random.shuffle(rfids)
            # for i in range(min(samples_per_patient, rfids.shape[0])):
            #     lead_idx = rfids[i, FID_LEAD_IDX]
            #     frame_start = max(rfids[i, FID_LOC_IDX] - int(random.uniform(0.45, 0.55) * self.frame_size), 0)
            #     frame_end = frame_start + self.frame_size
            #     if frame_end - frame_start < self.frame_size:
            #         continue
            #     x = data[frame_start:frame_end, lead_idx].astype(np.float32)
            #     y = labels[frame_start:frame_end, lead_idx].astype(np.int32)
            #     yield x, y
            # # END FOR

        # END FOR

    def signal_generator(self, patient_generator: PatientGenerator, samples_per_patient: int = 1) -> SampleGenerator:
        """
        Generate frames using patient generator.
        from the segments in patient data by placing a frame in a random location within one of the segments.
        Args:
            patient_generator (PatientGenerator): Generator that yields a tuple of patient id and patient data.
                    Patient data may contain only signals, since labels are not used.
            samples_per_patient (int): Samples per patient.
        Returns:
            SampleGenerator: Generator of input data of shape (frame_size, 1)
        """
        for _, pt in patient_generator:
            data = pt["data"][:]
            if self.sampling_rate != self.target_rate:
                data = pk.signal.resample_signal(data, self.sampling_rate, self.target_rate, axis=0)
            # END IF
            for _ in range(samples_per_patient):
                lead_idx = np.random.randint(data.shape[1])
                if data.shape[0] > self.frame_size:
                    frame_start = np.random.randint(data.shape[0] - self.frame_size)
                else:
                    frame_start = 0
                frame_end = frame_start + self.frame_size
                x = data[frame_start:frame_end, lead_idx].astype(np.float32).reshape((self.frame_size,))
                yield x
            # END FOR
        # END FOR

    def get_patient_data_segments(self, patient: int) -> tuple[npt.NDArray, npt.NDArray]:
        """Get patient's entire data and segments
        Args:
            patient (int): Patient ID (1-based)

        Returns:
            tuple[npt.NDArray, npt.NDArray]: (data, segment labels)
        """
        pt_key = f"p{patient:05d}"
        with h5py.File(self.ds_path / f"{pt_key}.h5", mode="r") as pt:
            data: npt.NDArray = pt["data"][:]
            segs: npt.NDArray = pt["segmentations"][:]
        labels = np.zeros_like(data)
        for seg_idx in range(segs.shape[0]):  # pylint: disable=no-member
            seg = segs[seg_idx]
            labels[seg[SEG_BEG_IDX] : seg[SEG_END_IDX] + 0, seg[SEG_LEAD_IDX]] = seg[SEG_LBL_IDX]
        # END FOR
        return data, labels

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
                pt_key = f"p{patient_id:05d}"
                with h5py.File(self.ds_path / f"{pt_key}.h5", mode="r") as h5:
                    yield patient_id, h5
            # END FOR
            if not repeat:
                break
        # END WHILE

    def convert_pt_wfdb_to_hdf5(
        self, patient: int, src_path: os.PathLike, dst_path: os.PathLike, force: bool = False
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Convert LUDB patient data from WFDB to more consumable HDF5 format.

        Args:
            patient (int): Patient id (1-based)
            src_path (str): Source path to WFDB folder
            dst_path (str): Destination path to store HDF5 file

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray]: data, segments, and fiducials
        """
        import wfdb  # pylint: disable=import-outside-toplevel

        pt_id = f"p{patient:05d}"
        pt_src_path = str(src_path / f"{patient}")
        rec = wfdb.rdrecord(pt_src_path)
        data = np.zeros_like(rec.p_signal)
        segs = []
        fids = []
        for i, lead in enumerate(rec.sig_name):
            lead_id = LudbLeadsMap.get(lead)
            ann = wfdb.rdann(pt_src_path, extension=lead)
            seg_start = seg_stop = sym_id = None
            data[:, lead_id] = rec.p_signal[:, i]
            for j, symbol in enumerate(ann.symbol):
                # Start of segment
                if symbol == "(":
                    seg_start = ann.sample[j]
                    seg_stop = None
                # Fiducial / segment type
                elif symbol in LudbSymbolMap:
                    sym_id = LudbSymbolMap.get(symbol)
                    if seg_start is None:
                        seg_start = ann.sample[j]
                    fids.append([lead_id, sym_id, ann.sample[j]])
                # End of segment (start and symbol are never 0 but can be None)
                elif symbol == ")" and seg_start and sym_id:
                    seg_stop = ann.sample[j]
                    segs.append([lead_id, sym_id, seg_start, seg_stop])
                else:
                    seg_start = seg_stop = None
                    sym_id = None
            # END FOR
        # END FOR
        segs = np.array(segs)
        fids = np.array(fids)

        if dst_path:
            os.makedirs(dst_path, exist_ok=True)
            pt_dst_path = dst_path / f"{pt_id}.h5"
            with h5py.File(pt_dst_path, "w") as h5:
                h5.create_dataset("data", data=data, compression="gzip")
                h5.create_dataset("segmentations", data=segs, compression="gzip")
                h5.create_dataset("fiducials", data=fids, compression="gzip")
            # END WITH
        # END IF

        return data, segs, fids

    def convert_dataset_zip_to_hdf5(
        self,
        zip_path: os.PathLike,
        patient_ids: npt.NDArray | None = None,
        force: bool = False,
        num_workers: int | None = None,
    ):
        """Convert dataset into individial patient HDF5 files.

        Args:
            zip_path (PathLike): Zip path
            patient_ids (npt.NDArray | None, optional): List of patient IDs to extract. Defaults to all.
            force (bool, optional): Whether to force re-download if destination exists. Defaults to False.
            num_workers (int, optional): # parallel workers. Defaults to os.cpu_count().
        """
        if not patient_ids:
            patient_ids = self.patient_ids

        subdir = "lobachevsky-university-electrocardiography-database-1.0.1"
        with Pool(processes=num_workers) as pool, tempfile.TemporaryDirectory() as tmpdir, zipfile.ZipFile(
            zip_path, mode="r"
        ) as zp:
            ludb_dir = Path(tmpdir, "ludb")
            zp.extractall(ludb_dir)
            f = functools.partial(
                self.convert_pt_wfdb_to_hdf5,
                src_path=ludb_dir / subdir / "data",
                dst_path=self.ds_path,
                force=force,
            )
            _ = list(tqdm(pool.imap(f, patient_ids), total=len(patient_ids)))
        # END WITH

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download LUDB dataset

        Args:
            num_workers (int | None, optional): # parallel workers. Defaults to None.
            force (bool, optional): Force redownload. Defaults to False.
        """

        logger.info("Downloading LUDB dataset")
        ds_url = (
            "https://physionet.org/static/published-projects/ludb/"
            "lobachevsky-university-electrocardiography-database-1.0.1.zip"
        )
        ds_zip_path = self.ds_path / "ludb.zip"
        os.makedirs(self.ds_path, exist_ok=True)
        if os.path.exists(ds_zip_path) and not force:
            logger.warning(
                f"Zip file already exists. Please delete or set `force` flag to redownload. PATH={ds_zip_path}"
            )
        else:
            download_file(ds_url, ds_zip_path, progress=True)

        # 2. Extract and convert patient ECG data to H5 files
        logger.info("Generating LUDB patient data")
        self.convert_dataset_zip_to_hdf5(zip_path=ds_zip_path, force=force, num_workers=num_workers)
        logger.info("Finished LUDB patient data")
