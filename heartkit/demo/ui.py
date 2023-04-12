import os
import time

import numpy as np
import plotext as plt
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import ConnectTimeout, HTTPError
from rich.ansi import AnsiDecoder
from rich.console import Group
from rich.jupyter import JupyterMixin
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.styled import Styled
from rich.table import Table
from rich.text import Text

from ..utils import setup_logger
from .client import HKRestClient
from .defines import HeartKitState

logger = setup_logger(__name__)

rhythym_names = ["Normal", "Tachycardia", "Bradycardia"]


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


class ConsoleUi:
    """Simple console UI front-end for Heart Kit"""

    def __init__(self, addr: str) -> None:
        super().__init__()
        self.layout = None
        self.state = HeartKitState()
        self.client = HKRestClient(addr=addr)

    def make_ecg_plot(self, width: float, height: float) -> str:
        """Create ecg plot"""
        plt.clf()
        plt.theme("clear")
        beat_idxs = np.where(np.array(self.state.seg_mask) >= 16)[0]
        plt.plotsize(width, height)
        plt.xlabel("Time (sample)")
        for beat_idx in beat_idxs:
            plt.vline(beat_idx, "magenta+")
        plt.plot(self.state.data, color="cyan+")
        return plt.build()

    def make_result_table(self) -> Table:
        """Create result table"""
        result = self.state.results
        table = Table(leading=2, expand=True, header_style="bold magenta")
        table.add_column("Result")
        table.add_column("Value")
        table.add_row("Heart Rate", f"{result.heart_rate:0.0f} BPM")
        table.add_row(
            "Heart Rhythm",
            "Arrhythmia" if result.arrhythmia else rhythym_names[result.heart_rhythm],
        )
        table.add_row(
            "Total Beats",
            f"{result.num_norm_beats + result.num_pac_beats + result.num_pvc_beats}",
        )
        table.add_row(
            "Normal Beats", "--" if result.arrhythmia else f"{result.num_norm_beats}"
        )
        table.add_row(
            "PAC Beats", "--" if result.arrhythmia else f"{result.num_pac_beats}"
        )
        table.add_row(
            "PVC Beats", "--" if result.arrhythmia else f"{result.num_pvc_beats}"
        )
        return table

    def create_layout(self):
        """Create layout"""
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
            Layout(name="stateLayout", ratio=1, minimum_size=1),
            Layout(name="tableLayout", ratio=4, minimum_size=1),
        )
        return layout

    def update_layout(self):
        """Update layout"""
        self.layout["liveLayout"].update(
            Panel(PlotextMixin(make_plot=self.make_ecg_plot), title="ECG Data")
        )
        self.layout["stateLayout"].update(
            Panel(
                Text(f"{self.state.app_state}", justify="center"),
                title="State",
                padding=1,
            )
        )
        self.layout["tableLayout"].update(
            Panel(self.make_result_table(), title="Results")
        )

    def fetch_state(self) -> bool:
        """Fetch state"""
        try:
            data_id = self.client.get_data_id()
            if data_id != self.state.data_id:
                self.state = self.client.get_state()
                return True
            self.state.app_state = self.client.get_app_state()
            return False
        except (HTTPError, ReqConnectionError, ConnectTimeout) as err:
            logger.warning(f"Failed to fetch state {err}")
            return False

    def run_forever(self):
        """Start running UI"""
        self.layout = self.create_layout()
        title = ":red_heart-emoji: Live Heart Kit Demo"
        self.layout["header"].update(Styled(title, style=Style(color="magenta")))
        stateLayout = self.layout["stateLayout"]
        run = True
        with Live(self.layout, refresh_per_second=4) as live:
            while run:
                try:
                    time.sleep(1)
                    did_update = self.fetch_state()
                    if did_update:
                        self.update_layout()
                    else:
                        stateLayout.update(
                            Panel(
                                Text(f"{self.state.app_state}", justify="center"),
                                title="State",
                                padding=1,
                            )
                        )
                    live.refresh()
                except KeyboardInterrupt:
                    run = False
                except Exception as err:  # pylint: disable=broad-exception-caught
                    logger.exception(err)
            # END WHILE
        # END WITH


if __name__ == "__main__":
    rest_addr = f"{os.getenv('REST_HOST', 'http://0.0.0.0')}:{os.getenv('REST_HOST', '8000')}/api/v1"
    ui = ConsoleUi(addr=rest_addr)
    ui.run_forever()
