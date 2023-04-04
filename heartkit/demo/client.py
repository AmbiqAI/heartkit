import abc
from typing import cast

import requests

from .defines import AppState, HeartKitState, HKResult


class HKClient(abc.ABC):
    """HKClient abstract base class"""

    def __init__(self) -> None:
        raise NotImplementedError()

    def get_state(self) -> HeartKitState:
        """Get state"""
        raise NotImplementedError()

    def set_state(self, state: HeartKitState):
        """Set state"""
        raise NotImplementedError()

    def get_app_state(self) -> AppState:
        """Get app state"""
        raise NotImplementedError()

    def set_app_state(self, app_state: AppState):
        """Set app state"""
        raise NotImplementedError()

    def get_data_id(self) -> int:
        """Get data id"""
        raise NotImplementedError()

    def set_data_id(self, data_id: int):
        """Set data id"""
        raise NotImplementedError()

    def get_data(self) -> list[float]:
        """Get ECG data"""
        raise NotImplementedError()

    def set_data(self, data: list[float]):
        """Set ECG data"""
        raise NotImplementedError()

    def get_segmentation(self) -> list[int]:
        """Get segmentation mask"""
        raise NotImplementedError()

    def set_segmentation(self, seg_mask: list[int]):
        """Set segmentation mask"""
        raise NotImplementedError()

    def get_results(self) -> HKResult:
        """Get result breakdown"""
        raise NotImplementedError()

    def set_results(self, result: HKResult):
        """Set result breakdown"""
        raise NotImplementedError()


class HKRestClient(HKClient):
    """REST implementation of HKClient"""

    def __init__(self, addr) -> None:
        self.addr = addr

    def get_state(self) -> HeartKitState:
        r = requests.get(f"{self.addr}/state", timeout=10)
        r.raise_for_status()
        return HeartKitState.parse_obj(r.json())

    def set_state(self, state: HeartKitState):
        r = requests.post(
            f"{self.addr}/state", data=state.json(by_alias=True), timeout=10
        )
        r.raise_for_status()

    def get_app_state(self) -> AppState:
        r = requests.get(f"{self.addr}/app/state", timeout=10)
        r.raise_for_status()
        return AppState(r.json())

    def set_app_state(self, app_state: AppState):
        r = requests.post(
            f"{self.addr}/app/state", params=dict(app_state=app_state.value), timeout=10
        )
        r.raise_for_status()

    def get_data_id(self):
        r = requests.get(f"{self.addr}/data_id", timeout=10)
        r.raise_for_status()
        return r.json()

    def set_data_id(self, data_id: int):
        r = requests.post(
            f"{self.addr}/data_id", params=dict(data_id=data_id), timeout=10
        )
        r.raise_for_status()

    def get_data(self) -> list[float]:
        r = requests.get(f"{self.addr}/data", timeout=10)
        r.raise_for_status()
        return cast(list[float], r.json())

    def set_data(self, data: list[float]):
        r = requests.post(f"{self.addr}/data", json=data, timeout=10)
        r.raise_for_status()

    def get_segmentation(self) -> list[int]:
        r = requests.get(f"{self.addr}/segmentation", timeout=10)
        r.raise_for_status()
        return cast(list[int], r.json())

    def set_segmentation(self, seg_mask: list[int]):
        r = requests.post(f"{self.addr}/segmentation", json=seg_mask, timeout=10)
        r.raise_for_status()

    def get_results(self) -> HKResult:
        r = requests.get(f"{self.addr}/results", timeout=10)
        r.raise_for_status()
        return HKResult.parse_obj(r.json())

    def set_results(self, result: HKResult):
        r = requests.post(
            f"{self.addr}/results", data=result.json(by_alias=True), timeout=10
        )
        r.raise_for_status()
