import os
import threading
import time
from enum import Enum
from typing import List, Optional, Tuple

import erpc
import numpy as np
import numpy.typing as npt
import plotext as plt
import pydantic_argparse
from rich.ansi import AnsiDecoder
from rich.console import Group
from rich.jupyter import JupyterMixin
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text
from scipy.special import softmax
from serial.serialutil import SerialException
from sklearn.utils import shuffle

from neuralspot.rpc import GenericDataOperations_EvbToPc as gen_evb2pc
from neuralspot.rpc import GenericDataOperations_PcToEvb as gen_pc2evb
from neuralspot.rpc.utils import get_serial_transport

from .datasets.icentia11k import IcentiaDataset
from .types import HeartDemoParams, get_class_names
from .utils import setup_logger

logger = setup_logger(__name__)


class DemoBlockCommands(str, Enum):
    """Demo EVB block commands"""

    SEND_SAMPLES = "SEND_SAMPLES"
    SEND_RESULTS = "SEND_RESULTS"
    FETCH_SAMPLES = "FETCH_SAMPLES"


class EvbAppState(str, Enum):
    """EVB App FSM states"""

    IDLE_STATE = "IDLE_STATE"
    COLLECT_STATE = "COLLECT_STATE"
    PREPROCESS_STATE = "PREPROCESS_STATE"
    INFERENCE_STATE = "INFERENCE_STATE"
    DISPLAY_STATE = "DISPLAY_STATE"
    FAIL_STATE = "FAIL_STATE"


class PlotextMixin(JupyterMixin):
    """Mixin to allow Plotext to work with Rich"""

    def __init__(self, make_plot):
        self.decoder = AnsiDecoder()
        self.make_plot = make_plot

    def __rich_console__(self, console, options):
        self.width = options.max_width or console.width
        self.height = options.height or console.height
        canvas = self.make_plot(self.width, self.height)
        self.rich_canvas = Group(*self.decoder.decode(canvas))
        yield self.rich_canvas


class EvbDemo(gen_evb2pc.interface.Ievb_to_pc):
    """EVB Demo app. Acts as delegate for eRPC generic data operations."""

    def __init__(self, params: HeartDemoParams) -> None:
        super().__init__()
        self.params = params
        self.test_data: Tuple[npt.ArrayLike, npt.ArrayLike] = self.load_test_data()
        self.state = EvbAppState.IDLE_STATE
        self._sample_idx = 0
        self._frame_idx = 0
        self._plot_data = np.array([])
        self._plot_results = np.zeros(len(self.class_labels))
        self._true_label: Optional[int] = None
        self._run = False

    @property
    def class_labels(self) -> List[str]:
        """Get class labels for demo."""
        return get_class_names(self.params.task)

    @property
    def window_size(self) -> int:
        """Full window size samples [PAD+FRAME]."""
        return self.params.pad_size + self.params.frame_size

    @property
    def true_label(self) -> str:
        """True label for given sample if using test data."""
        if self._true_label is None:
            return "Unknown"
        return self.class_labels[self._true_label]

    def load_test_data(self) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """Load test data

        Returns:
            Tuple[npt.ArrayLike, npt.ArrayLike]: x,y
        """
        ds = IcentiaDataset(
            ds_path=str(self.params.ds_path),
            task=self.params.task,
            frame_size=self.window_size,
        )
        test_ds = ds.load_test_dataset(
            test_patients=200,
            test_pt_samples=self.params.samples_per_patient,
            num_workers=self.params.data_parallelism,
        )
        test_x, test_y = next(test_ds.batch(2000).as_numpy_iterator())
        return shuffle(test_x, test_y)

    def clear_plot_data(self):
        """Clear plot data."""
        self._plot_data: npt.ArrayLike = np.full(
            self.window_size, fill_value=np.nan, dtype=np.float32
        )
        self._plot_results = np.zeros(len(self.class_labels))

    def make_live_plot(self, width: float, height: float) -> str:
        """Create live plot string"""
        plt.clf()
        plt.theme("clear")
        plt.plotsize(width, height)
        plt.title("Live Sensor Data")
        plt.xlabel("Time (sample)")
        plt.plot(self._plot_data)
        return plt.build()

    def make_bar_plot(self, width: float, height: float) -> str:
        """Create bar plot string"""
        plt.clf()
        plt.theme("clear")
        plt.plotsize(width, height)
        plt.ylim(0, 100)
        label_title = "Predicted Label"
        # if self._true_label is not None:
        #     label_title = f"[Y = {self.class_labels[self._true_label]}]"
        plt.title("EVB Classification")
        plt.xlabel(label_title)
        plt.bar(self.class_labels, self._plot_results)
        return plt.build()

    def increment_frame_idx(self, inc: int):
        """Increment frame index and optional increment sample index

        Args:
            inc (int): increment
        """
        self._frame_idx += inc
        if self._frame_idx >= self.window_size:
            logger.debug("Fetching next sample")
            self._frame_idx = 0
            self._sample_idx = (self._sample_idx + 1) % self.test_data[0].shape[0]

    def log_data(self, x):
        """Log data to file."""
        with open(self.params.job_dir / "evb_data.csv", "a+", encoding="utf-8") as f:
            f.write("\n".join((f"{v:0.1f}" for v in x)) + "\n")

    def ns_rpc_data_sendBlockToPC(self, block: gen_pc2evb.common.dataBlock):
        if DemoBlockCommands.SEND_SAMPLES in block.description:
            self._true_label = None
            x: npt.NDArray = np.frombuffer(block.buffer, dtype=np.float32)
            for v in x:
                self._plot_data[self._frame_idx] = v
                self.increment_frame_idx(1)
        if DemoBlockCommands.SEND_RESULTS in block.description:
            self._plot_results: npt.NDArray = 100 * softmax(
                np.frombuffer(block.buffer, dtype=np.float32)
            )
        return 0  # SUCCESS

    def ns_rpc_data_fetchBlockFromPC(self, block):
        return 0  # SUCCESS

    def ns_rpc_data_computeOnPC(
        self, in_block: gen_evb2pc.common.dataBlock, result_block
    ):
        if DemoBlockCommands.FETCH_SAMPLES in in_block.description:
            num_samples = in_block.length
            f_len = min(self.window_size - self._frame_idx, num_samples)
            x = (
                self.test_data[0][
                    self._sample_idx, self._frame_idx : self._frame_idx + f_len
                ]
                .squeeze()
                .astype(np.float32)
            )
            self._true_label = self.test_data[1][self._sample_idx]
            self._plot_data[self._frame_idx : self._frame_idx + f_len] = x
            self.increment_frame_idx(f_len)
            x = np.ascontiguousarray(x, dtype=np.float32).tobytes("C")

            result_block.value = gen_evb2pc.common.dataBlock(
                length=f_len,
                dType=gen_pc2evb.common.dataType.float32_e,
                description="ECG_SENSOR_RESPONSE",
                cmd=gen_evb2pc.common.command.generic_cmd,
                buffer=bytearray(x),
            )
        return 0  # SUCCESS

    def ns_rpc_data_remotePrintOnPC(self, msg):
        # Check for EVB state machine updates
        self.state = next((s for s in EvbAppState if s in msg), self.state)
        if self.state == EvbAppState.IDLE_STATE:
            pass
        elif self.state == EvbAppState.COLLECT_STATE:
            self._frame_idx = 0
            self.clear_plot_data()
        elif self.state == EvbAppState.PREPROCESS_STATE:
            pass
        elif self.state == EvbAppState.INFERENCE_STATE:
            pass
        elif self.state == EvbAppState.DISPLAY_STATE:
            pass
        elif self.state == EvbAppState.FAIL_STATE:
            pass
        return 0  # SUCCESS

    def make_layout(self):
        """Create live Rich layout"""
        layout = Layout(name="root")
        layout.split(
            Layout(name="header", size=1),
            Layout(name="main", ratio=1),
        )
        layout["main"].split_row(
            Layout(name="liveLayout", ratio=3),
            Layout(name="rightLayout", ratio=1),
        )
        layout["rightLayout"].split_column(
            Layout(name="barLayout", ratio=7),
            Layout(name="textLayout", ratio=1, minimum_size=1),
        )
        return layout

    def handle_thread_exception(self, *args):
        """Handle thread exceptions."""
        self._run = False

    def stop(self):
        """Stop running demo."""
        self._run = False

    def start(self):
        """Start running demo."""
        self._run = True
        os.makedirs(str(self.params.job_dir), exist_ok=True)

        transport = get_serial_transport(
            vid_pid=self.params.vid_pid, baudrate=self.params.baudrate
        )
        server = erpc.simple_server.ServerThread(transport, erpc.basic_codec.BasicCodec)
        server.add_service(gen_evb2pc.server.evb_to_pcService(self))

        logger.info("Server running")

        threading.excepthook = self.handle_thread_exception
        server.start()

        layout = self.make_layout()
        headerLayout = layout["header"]
        liveLayout = layout["liveLayout"]
        barLayout = layout["barLayout"]
        textLayout = layout["textLayout"]

        with Live(layout, refresh_per_second=4) as live:
            while self._run:
                title = f"♥️  Heart Demo [{self.state.value}]"
                headerLayout.update(Text(title, justify="left"))
                livePanel = Panel(PlotextMixin(make_plot=self.make_live_plot))
                liveLayout.update(livePanel)

                barPanel = Panel(PlotextMixin(make_plot=self.make_bar_plot))
                barLayout.update(barPanel)
                textLayout.update(
                    Panel(
                        Padding(Text(self.true_label, justify="center"), (1, 0)),
                        title="True Label",
                    )
                )
                live.refresh()
                time.sleep(0.5)
            # END WHILE
        # END WITH


def evb_demo(params: HeartDemoParams):
    """EVB Demo

    Args:
        params (HeartDemoParams): Demo parameters
    """
    try:
        demo = EvbDemo(params=params)
        demo.start()
    except (KeyboardInterrupt, SerialException):
        logger.info("Server stopping")
        demo.stop()
    except Exception as err:  # pylint: disable=broad-except
        logger.exception(f"Unhandled error {err}")


def create_parser():
    """Create CLI parser"""
    return pydantic_argparse.ArgumentParser(
        model=HeartDemoParams,
        prog="Heart EVB demo",
        description="Demo heart model on EVB",
    )


if __name__ == "__main__":
    parser = create_parser()
    evb_demo(parser.parse_typed_args())
