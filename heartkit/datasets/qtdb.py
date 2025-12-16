import contextlib
import functools
import os
import random
import tempfile
import zipfile
from typing import Generator

import h5py
import numpy as np
import numpy.typing as npt
import physiokit as pk
from tqdm.contrib.concurrent import process_map
import helia_edge as helia

from .dataset import HKDataset
from .defines import PatientGenerator, PatientData

logger = helia.utils.setup_logger(__name__)

QtdbSymbolMap = {
    "o": 0,  # Other
    "p": 1,  # P Wave
    "N": 2,  # QRS complex
    "t": 3,  # T Wave
}

FID_LEAD_IDX = 0
FID_LBL_IDX = 1
FID_LOC_IDX = 2
SEG_LEAD_IDX = 0
SEG_LBL_IDX = 1
SEG_BEG_IDX = 2
SEG_END_IDX = 3


class QtdbDataset(HKDataset):
    """QT dataset"""

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._cached_data = {}

    @property
    def name(self) -> str:
        """Dataset name"""
        return "qtdb"

    @property
    def sampling_rate(self) -> int:
        """Sampling rate in Hz"""
        return 250

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

        # fmt: off
        return np.array(
            [
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                100,
                102,
                103,
                104,
                106,
                107,
                110,
                111,
                112,
                114,
                116,
                117,
                121,
                122,
                123,
                124,
                126,
                129,
                133,
                136,
                166,
                170,
                203,
                210,
                211,
                213,
                221,
                223,
                230,
                231,
                232,
                233,
                301,
                302,
                303,
                306,
                307,
                308,
                310,
                405,
                406,
                409,
                411,
                509,
                603,
                604,
                606,
                607,
                609,
                612,
                704,
                803,
                808,
                811,
                820,
                821,
                840,
                847,
                853,
                871,
                872,
                873,
                883,
                891,
                14046,
                14157,
                14172,
                15814,
                16265,
                16272,
                16273,
                16420,
                16483,
                16539,
                16773,
                16786,
                16795,
                17152,
                17453,
            ]
            # fmt: on
        )  # 104, 114, 116 have multiple recordings

    def get_train_patient_ids(self) -> npt.NDArray:
        """Get dataset training patient IDs

        Returns:
            npt.NDArray: patient IDs
        """
        return self.patient_ids[:82]

    def get_test_patient_ids(self) -> npt.NDArray:
        """Get dataset patient IDs reserved for testing only

        Returns:
            npt.NDArray: patient IDs
        """
        return self.patient_ids[82:]

    def _pt_key(self, patient_id: int):
        """Get patient key"""
        return f"{patient_id}"

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
                    pt_data["segmentations"] = h5["segmentations"][:]
                    pt_data["fiducials"] = h5["fiducials"][:]
                # END WITH
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
            Generator[npt.NDArray, None, None]: Generator of input data of shape (frame_size, 1)
        """
        if target_rate is None:
            target_rate = self.sampling_rate

        input_size = int(np.ceil((self.sampling_rate / target_rate) * frame_size))

        for pt in patient_generator:
            with self.patient_data(pt) as h5:
                data: h5py.Dataset = h5["data"][:]
            # END WITH
            for _ in range(samples_per_patient):
                lead = random.choice(data.shape[1])
                start = np.random.randint(0, data.shape[0] - input_size)
                x = data[start : start + input_size, lead].squeeze()
                x = np.nan_to_num(x).astype(np.float32)
                if self.sampling_rate != target_rate:
                    x = pk.signal.resample_signal(x, self.sampling_rate, target_rate, axis=0)
                    x = x[:frame_size]
                # END IF
                yield x
            # END FOR
        # END FOR

    def get_patient_data_segments(self, patient_id: int) -> tuple[npt.NDArray, npt.NDArray]:
        """Get patient's entire data and segments

        Args:
            patient_id (int): Patient ID (1-based)

        Returns:
            tuple[npt.NDArray, npt.NDArray]: (data, segment labels)
        """
        with self.patient_data(patient_id) as pt:
            data: npt.NDArray = pt["data"][:]
            segs: npt.NDArray = pt["segmentations"][:]
        labels = np.zeros_like(data)
        for seg_idx in range(segs.shape[0]):  # pylint: disable=no-member
            seg = segs[seg_idx]
            labels[seg[2] : seg[3] + 0, seg[0]] = seg[1]
        return data, labels

    def download(self, num_workers: int | None = None, force: bool = False):
        """Download QT dataset

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
        logger.debug("Downloading QTDB dataset")
        ds_url = "https://physionet.org/static/published-projects/qtdb/qt-database-1.0.0.zip"
        ds_zip_path = self.path / "qtdb.zip"
        os.makedirs(self.path, exist_ok=True)
        if os.path.exists(ds_zip_path) and not force:
            logger.warning(
                f"Zip file already exists. Please delete or set `force` flag to redownload. PATH={ds_zip_path}"
            )
        else:
            helia.utils.download_file(ds_url, ds_zip_path, progress=True)

        # 2. Extract and convert patient ECG data to H5 files
        logger.debug("Generating QT patient data")
        self.convert_dataset_zip_to_hdf5(zip_path=ds_zip_path, force=force, num_workers=num_workers)
        logger.debug("Finished QTDB patient data")

    def convert_pt_wfdb_to_hdf5(
        self, patient: int, src_path: os.PathLike, dst_path: os.PathLike, force: bool = False
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Convert QTDB patient data from WFDB to more consumable HDF5 format.

        Args:
            patient (int): Patient id (1-based)
            src_path (PathLike): Source path to WFDB folder
            dst_path (PathLike): Destination path to store HDF5 file

        Returns:
            tuple[npt.NDArray, npt.NDArray, npt.NDArray]: data, segments, and fiducials
        """
        import wfdb  # pylint: disable=import-outside-toplevel

        pt_id = f"sel{patient}" if os.path.isfile(src_path / f"sel{patient}.dat") else f"sele{patient:04d}"
        pt_src_path = str(src_path / pt_id)
        rec = wfdb.rdrecord(pt_src_path)
        data = np.zeros_like(rec.p_signal)
        segs = []
        fids = []
        for i, _ in enumerate(rec.sig_name):
            lead_id = i

            ann = wfdb.rdann(pt_src_path, extension=f"pu{lead_id}")

            seg_start = seg_stop = sym_id = None
            data[:, lead_id] = rec.p_signal[:, i]
            for j, symbol in enumerate(ann.symbol):
                # Start of segment
                if symbol == "(":
                    seg_start = ann.sample[j]
                    seg_stop = None
                # Fiducial / segment type
                elif symbol in QtdbSymbolMap:
                    sym_id = QtdbSymbolMap.get(symbol)
                    if seg_start is None:
                        seg_start = ann.sample[j]
                    fids.append([lead_id, sym_id, ann.sample[j]])
                elif symbol == ")" and seg_start and sym_id:
                    seg_stop = ann.sample[j]
                    segs.append([lead_id, sym_id, seg_start, seg_stop])
                else:
                    seg_start = seg_stop = None
                    sym_id = None
            # END FOR
        # END FOR
        fids = np.array(fids)
        segs = np.array(segs)

        if dst_path:
            os.makedirs(dst_path, exist_ok=True)
            pt_dst_path = dst_path / f"{patient}.h5"
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

        subdir = "qt-database-1.0.0"
        with tempfile.TemporaryDirectory() as tmpdir, zipfile.ZipFile(zip_path, mode="r") as zp:
            qtdb_dir = tmpdir / "qtdb"
            zp.extractall(qtdb_dir)

            f = functools.partial(
                self.convert_pt_wfdb_to_hdf5,
                src_path=qtdb_dir / subdir,
                dst_path=self.path,
                force=force,
            )
            _ = process_map(f, patient_ids)
        # END WITH

    def close(self):
        """Close dataset"""
        self._cached_data.clear()
