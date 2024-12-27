"""
This program was developed as a Final Project for the High Performance Computing course
in the Department of Electrical and Electronics Engineering, Korea University.
Author: Cho Soo Hwan
Student ID: 2024020798
E-mail: soohwancho@korea.ac.kr

Purpose:
The purpose of this program is to find the optimal parameters for a Dataloader during data loading,
including `num_workers`, `prefetch_factor`, `pin_memory`, and `persistent_workers`, for the given
device and batch size. Please note that the testing process may take some time, so it is recommended
to save the results separately after testing in your environment.

Copyright Notice:
This program is licensed under the MIT License. You are free to copy, modify, and redistribute
this program, provided that the original author is properly credited in any derived work.
"""

import warnings
import time
import os
import torch
from torch.utils.data import DataLoader, Dataset
import cpuinfo
import psutil
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, Tuple
import json
import copy


class SearchRange:
    """class for serarch range"""

    def __init__(self):
        cpu_core_count = os.cpu_count()  # psutil.cpu_count(logical=True)
        self.num_workers = (1, cpu_core_count)
        self.prefetch_factor = [2, 3, 4, 1]
        self.prefetch_factor2 = list(range(5, 8))
        self.pin_memory = [1, 0]
        self.persistent_workers = [1, 0]


@dataclass
class TestParams:
    device: torch.device
    batch_size: int
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool

    @classmethod
    def empty_obj(cls):
        return cls(None, 0, 0, 0, False, False)

    def to_tuple(self) -> Tuple:
        return (
            self.device.type,
            self.batch_size,
            self.num_workers,
            self.prefetch_factor,
            self.pin_memory,
            self.persistent_workers,
        )

    def to_string(self) -> str:
        return f"device: {self.device.type}, batch_size: {self.batch_size}, num_workers: {self.num_workers}, perfetch_factor: {self.prefetch_factor}, pin_memory: {self.pin_memory}, persistent_workers: {self.persistent_workers}"

    def reset(self):
        self.device = None
        self.batch_size = 0
        self.num_workers = 0
        self.prefetch_factor = 0
        self.pin_memory = False
        self.persistent_workers = False


class TestResult:
    """class for test result"""

    ERR_VAL = float("inf")

    def __init__(self, avr_runtime_ms, test_time_ms, tcid, try_cnt, totoal_cnt, test_aborted, error_msg=""):
        self.avr_runtime_ms = round(avr_runtime_ms, 3)
        self.test_time_ms = round(test_time_ms, 3)
        self.test_aborted = test_aborted
        self.tcid = tcid
        self.try_cnt = try_cnt
        self.total_cnt = totoal_cnt
        self.error_msg = error_msg

    @classmethod
    def empty_obj(cls):
        return cls(TestResult.ERR_VAL, 0, 0, 0, 0, False, "")

    def reset(self):
        self.avr_runtime_ms = self.ERR_VAL
        self.test_time_ms = 0
        self.test_aborted = False
        self.tcid = 0
        self.try_cnt = 0
        self.total_cnt = 0
        self.error_msg = ""

    def to_string(self) -> str:
        s = f"tcid: {self.tcid}, runtime_ms: {self.avr_runtime_ms}, test_time_sec: {round(self.test_time_ms/1000.0, 3)} abort: {self.test_aborted}, try_cnt: {self.try_cnt}, total_cnt: {self.total_cnt}, errmsg: {self.error_msg}"
        return s


class DataloaderParamHelper:
    """
    Singleton class for assisting in finding the optimal parameters for a Dataloader.

    This class is designed to help determine the best-performing parameters for loading data efficiently.
    It ensures that only one instance of the class exists, and access is strictly provided via the
    `getInstance()` method.

    Usage:
        - Use `DataloaderParamHelper.getInstance()` to obtain the singleton instance.
        - Direct instantiation of this class is not allowed.

    Responsibilities:
        - Evaluate and identify the optimal parameters for a Dataloader, such as `batch_size`,
        `num_workers`, `prefetch_factor`, `pin_memory`, and `persistent_workers`.
        - Facilitate performance optimization for data loading in machine learning workflows.

    Attributes:
        _instance: The single instance of the class, created and managed internally.

    Methods:
        @classmethod
        getInstance(): Provides access to the singleton instance of the class.
    """

    _instance = None
    STABLE_TIME_THREASHOLD_MS = 3.0
    EARLY_STABLE_COUNT = 3
    TEST_ABORT_COND_TIMES = 1.3
    DEBUG_MODE = False
    HARDWARE_INFO_STR = ""
    
    @classmethod
    def getInstance(cls):
        if not cls._instance:
            cls._instance = DataloaderParamHelper()
        return cls._instance

    def __init__(self):
        self._range = SearchRange()
        self.dataset: Dataset = None
        self.test_result_dict: Dict[Tuple, TestResult] = {}
        self.tcid = 0
        self.total_tc_count = 0
        self.best_params = TestParams.empty_obj()
        self.best_result = TestResult.empty_obj()
        DataloaderParamHelper.get_hardware_info_str()
        
    def get_quick_predict_count(self, batch_size) -> int:
        if batch_size < 100:
            return 30
        elif batch_size < 200:
            return 10
        else:
            return 5

    def calculate_scaled_percentage(self, value, min_value=2, max_value=2048, new_min=0.03, new_max=0.4):
        if value < min_value:
            value = min_value
        if value > max_value:
            value = max_value

        # Scale the percentage to the new range
        scaled_percentage = new_min + ((value - min_value) / (max_value - min_value)) * (new_max - new_min)
        return scaled_percentage

    def get_max_test_cnt_per(self, batch_size) -> int:
        return self.calculate_scaled_percentage(batch_size)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_test_result(self, test_params: TestParams) -> TestResult:
        key = test_params.to_tuple()
        if key in self.test_result_dict[key]:
            return self.test_result_dict[key]
        else:
            None

    def set_test_result(self, test_params: TestParams, test_result: TestResult):
        key = test_params.to_tuple()
        self.test_result_dict[key] = test_result
        # if self.use_filedb:
        #    self.insert_tuple(test_params, test_result)

    def print_console(self, message, debug_mode_only=True):

        if not debug_mode_only:
            print(message)
        else:
            if DataloaderParamHelper.DEBUG_MODE:
                print(message)

    def print_progress_message(self, device_str, batch_size, done=False):

        test_progress = min(100.0, (float)(self.tcid / self.total_tc_count * 100))
        if done:
            test_progress = 100.0

        self.print_console(
            f"Searching for Optimal Dataloader Parameters. device: {device_str}, batch_size: {batch_size} --- {test_progress:.1f}%",
            debug_mode_only=False,
        )

    def find_best_elapsed_time(self, device, batch_size, result_filepath, load_result_file_if_exists, full_test):

        if load_result_file_if_exists and len(result_filepath) > 0:
            result_dict = DataloaderParamHelper.load_from_json(batch_size=batch_size, device=device, file_path=result_filepath)
            if result_dict != None:
                self.print_console(f"DataLoaderParamHelper result loaded from file. -> {result_filepath}", False)
                return result_dict
            else:
                self.print_console(f"DataLoaderParamHelper result file or data does not exist. -> {result_filepath}", False)
            
        self.best_params.reset()
        self.best_result.reset()
        self.tcid = 0
        logcnt = math.ceil(math.log2(self._range.num_workers[1])) - 1
        prefetch_cnt = len(self._range.prefetch_factor) + len(self._range.prefetch_factor2)
        self.total_tc_count = int(logcnt * prefetch_cnt * 4) * 2  # mid and mid +1
        left, right = self._range.num_workers

        test_start_time = time.time()
        while left <= right:
            mid = (left + right) // 2

            test_params, test_result = self.find_best_co(device, batch_size, num_workers=mid, full_test=full_test)
            test_params2, test_result2 = self.find_best_co(device, batch_size, num_workers=mid + 1, full_test=full_test)

            # Compare average runtime
            if test_result.avr_runtime_ms > test_result2.avr_runtime_ms:
                left = mid + 1
            else:
                right = mid - 1

        self.print_progress_message(device.type, batch_size, done=True)

        test_end_time = time.time()
        test_time_sec = int(test_end_time - test_start_time)
        params = {
            "batch_size": batch_size,
            "num_workers": self.best_params.num_workers,
            "prefetch_factor": self.best_params.prefetch_factor,
            "pin_memory": self.best_params.pin_memory,
            "persistent_workers": self.best_params.persistent_workers,
            "avr_runtime_ms": self.best_result.avr_runtime_ms,
            "test_time_sec": test_time_sec,
        }

        try:
            
            if len(result_filepath) > 0:
                # Example usage
                DataloaderParamHelper.save_to_json(
                    batch_size=batch_size,
                    device=device.type,
                    params=params,
                    file_path=result_filepath,
                )
                self.print_console(f" >> Searching Result Saved in {result_filepath} << ", False)

        except Exception as expt:
            pass

        return params

    def find_best_co(self, device, batch_size, num_workers, full_test=False) -> Tuple[TestParams, TestResult]:

        current_best_params = TestParams.empty_obj()
        current_best_result = TestResult.empty_obj()
        first_num_workers_test = True

        for pin_memory in self._range.pin_memory:
            for persistent_workers in self._range.persistent_workers:
                prefetch_factor_list = copy.deepcopy(self._range.prefetch_factor)
                prefetch_factor_list.extend(self._range.prefetch_factor2)  # 5 ~ 21
                prev_prefetch_time = float("inf")

                test_cnt_decrease2 = len(self._range.prefetch_factor) + len(self._range.prefetch_factor2)
                for _i, prefetch_factor in enumerate(prefetch_factor_list):

                    self.tcid += 1
                    self.print_progress_message(device.type, batch_size)
                    self.print_console(f"    -> tcid: {self.tcid} / total: {self.total_tc_count}")

                    test_params = TestParams(
                        device, batch_size, num_workers, prefetch_factor, pin_memory, persistent_workers
                    )

                    if full_test:
                        self.print_console(f"Full Test : {str(test_params.to_string())}")
                        result_obj: TestResult = self.do_full_test(test_params)
                    else:
                        self.print_console(f"Quick Test : {str(test_params.to_string())}")
                        result_obj: TestResult = self.do_quick_test(test_params)

                    self.print_console(f"    -> {result_obj.to_string()}")

                    if current_best_result.avr_runtime_ms > result_obj.avr_runtime_ms:
                        current_best_result = result_obj
                        current_best_params = test_params

                    if self.best_result.avr_runtime_ms > result_obj.avr_runtime_ms:
                        self.best_params = test_params
                        self.best_result = result_obj
                        self.print_console(f"    -> Best Record!")
                    else:
                        # num_workers 다를 때, 최고 성능 num_workers와 성능차이가 크게 나면 이 num_workers에서 더이상 테스틀 진행하지 않는다.
                        # num_workers에서 오는 속도 차이를 극복하기 어렵다고 판단한다.
                        if (
                            self.best_result.avr_runtime_ms < result_obj.avr_runtime_ms * 2
                            and self.best_params.num_workers != test_params.num_workers
                            and first_num_workers_test
                        ):
                            self.print_console(f"    -> Stop Testing Current num_workers: {test_params.num_workers}!")
                            return current_best_params, current_best_result

                    if prefetch_factor in self._range.prefetch_factor2:
                        if prev_prefetch_time < result_obj.avr_runtime_ms:
                            # don't need to test bigger prefetch factors.
                            self.print_console(f"    stop testing bigger prefetch factor.")
                            break

                    self.print_console("")
                    prev_prefetch_time = result_obj.avr_runtime_ms
                    first_num_workers_test = False

        return current_best_params, current_best_result

    def get_predict_time(self, elapsed_ms_list, stabilized_start_index, total_cnt, threashold_per):
        """
        Predicts the total time required to complete the test based on the measured times.

        Args:
            elapsed_ms_list (list[float]): List of measured elapsed times in milliseconds so far.
            stabilized_start_index (int): Index where the values became stabilized.
            total_cnt (int): Total number of iterations for the test.
            threshold_per (float): Threshold value used for filtering outliers.

        Returns:
            float: The predicted total time in milliseconds.
        """
        if stabilized_start_index < 3:
            stabilized_start_index = 3
        unstablized_values = elapsed_ms_list[0 : stabilized_start_index - 1]
        stable_after_values = elapsed_ms_list[stabilized_start_index:-1]

        if threashold_per < 100:
            """remove outliers"""
            threshold_value = np.percentile(stable_after_values, threashold_per)
            stabilized_values = [x for x in stable_after_values if x <= threshold_value]
        else:
            stabilized_values = stable_after_values

        stabilized_length = len(stabilized_values)
        stabilized_mean = np.mean(stabilized_values)

        partial_cnt = total_cnt - len(unstablized_values)
        a = stabilized_mean * (stabilized_length / partial_cnt)
        b = stabilized_mean * ((partial_cnt - stabilized_length) / partial_cnt)
        c = sum(unstablized_values) / len(unstablized_values) * (len(unstablized_values) / total_cnt)
        predicted_time = a + b + c
        return predicted_time

    def do_quick_test(self, test_params: TestParams) -> TestResult:
        """
        Performs a quick test to measure dataloader time based on the provided parameters.

        The test measures the time without completing the full loop, comparing it against the best recorded time.
        The test will run up to the maximum configured number of iterations but will terminate early if specific
        early termination conditions are met. The result is returned upon completion or early termination.

        Args:
            test_params (TestParams): Parameters for configuring the dataloader test.

        Returns:
            TestResult: The result of the test, including the measured times and status.
        """
        predicted_time = TestResult.ERR_VAL
        test_time_ms = -1
        try_cnt = 0
        total_cnt = 0
        test_aborted = False
        error_msg = ""
        start_test_time = time.perf_counter_ns()
        best_time_ms = self.best_result.avr_runtime_ms
        try:

            dataloader = None
            # Capture warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")  # Ensure all warnings are captured

                dataloader = DataLoader(
                    dataset=self.dataset,
                    batch_size=int(test_params.batch_size),
                    num_workers=int(test_params.num_workers),
                    prefetch_factor=bool(test_params.prefetch_factor),
                    pin_memory=bool(test_params.pin_memory),
                    persistent_workers=bool(test_params.persistent_workers),
                )
                # Inspect warnings
                for warning in w:
                    raise Exception(warning.message)

            elapsed_ms_list = []
            stable_count = 0
            stabilized = False
            stabilized_start_index = None
            total_cnt = len(dataloader)
            quick_predict_count = self.get_quick_predict_count(test_params.batch_size)
            flag_early_abort = True

            stime = time.perf_counter_ns()
            for images, labels in dataloader:
                try_cnt += 1
                images, labels = images.to(test_params.device), labels.to(test_params.device)

                torch.cuda.synchronize()
                etime = time.perf_counter_ns()
                elapsed_ms = (etime - stime) * 1e-6
                stime = time.perf_counter_ns()
                elapsed_ms_list.append(elapsed_ms)

                # Check stabilization
                if not stabilized:
                    if len(elapsed_ms_list) > 1:
                        if abs(elapsed_ms_list[-1] - elapsed_ms_list[-2]) <= self.STABLE_TIME_THREASHOLD_MS:
                            stable_count += 1
                            self.print_console(f"stalble += 1 : {stable_count}")
                        else:
                            stable_count = 0

                    if stable_count >= self.EARLY_STABLE_COUNT:
                        self.print_console(f"stalblized")
                        stabilized = True
                        stabilized_start_index = len(elapsed_ms_list) - self.EARLY_STABLE_COUNT - 1

                elif stabilized and flag_early_abort:
                    # If stabilized, calculate the mean of the last few stabilized values
                    if len(elapsed_ms_list) - stabilized_start_index >= quick_predict_count:
                        self.print_console(f"try predict!")
                        flag_early_abort = False
                        _predicted_time = self.get_predict_time(elapsed_ms_list, stabilized_start_index, total_cnt, 90)
                        # Test aborted: Predicted time exceeds twice the best average time.
                        if _predicted_time > best_time_ms * self.TEST_ABORT_COND_TIMES:
                            predicted_time = _predicted_time
                            test_aborted = True
                            self.print_console(f"early abort")
                            break
                        self.print_console(f"not aborted")

                else:
                    if try_cnt >= total_cnt * self.get_max_test_cnt_per(test_params.batch_size):

                        _predicted_time = self.get_predict_time(elapsed_ms_list, stabilized_start_index, total_cnt, 100)
                        predicted_time = _predicted_time
                        test_aborted = False
                        break

        except Exception as expt:
            error_msg = str(expt)

        end_test_time = time.perf_counter_ns()
        test_time_ms = (end_test_time - start_test_time) * 1e-6

        result_obj = TestResult(
            avr_runtime_ms=predicted_time,
            test_time_ms=test_time_ms,
            tcid=self.tcid,
            try_cnt=try_cnt,
            totoal_cnt=total_cnt,
            test_aborted=test_aborted,
            error_msg=error_msg,
        )

        self.set_test_result(test_params, result_obj)
        return result_obj

    def do_full_test(self, test_params: TestParams) -> TestResult:
        """
        Performs a full test by completing all iterations of the loop.

        This function runs the test until the end of the loop and can be used to verify the accuracy of the results
        obtained from the `do_quick_test` function.

        Args:
            test_params (TestParams): Parameters for configuring the dataloader test.

        Returns:
            TestResult: The result of the full test, including the measured times and status.
        """
        predicted_time = TestResult.ERR_VAL
        test_time_ms = -1
        try_cnt = 0
        totoal_cnt = 0
        test_aborted = False
        error_msg = ""
        start_test_time = time.perf_counter_ns()

        try:

            dataloader = None
            # Capture warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")  # Ensure all warnings are captured

                dataloader = DataLoader(
                    dataset=self.dataset,
                    batch_size=test_params.batch_size,
                    num_workers=test_params.num_workers,
                    prefetch_factor=test_params.prefetch_factor,
                    pin_memory=test_params.pin_memory,
                    persistent_workers=test_params.persistent_workers,
                )
                # Inspect warnings
                for warning in w:
                    raise Exception(warning.message)

            totoal_cnt = len(dataloader)
            cumulative_avg_list = []
            cycle_time_list = []
            prdt_list = []
            cumulative_avg = 0
            stime = time.perf_counter_ns()  # start
            for images, labels in dataloader:
                try_cnt += 1
                images, labels = images.to(test_params.device), labels.to(test_params.device)
                torch.cuda.synchronize()
                etime = time.perf_counter_ns()  # end
                elasped_ms = (etime - stime) * 1e-6
                cycle_time_list.append(elasped_ms)
                stime = time.perf_counter_ns()  # start

            predicted_time = np.mean(cycle_time_list)

        except Exception as expt:
            error_msg = str(expt)

        end_test_time = time.perf_counter_ns()
        test_time_ms = (end_test_time - start_test_time) * 1e-6

        result_obj = TestResult(
            avr_runtime_ms=predicted_time,
            test_time_ms=test_time_ms,
            tcid=self.tcid,
            try_cnt=try_cnt,
            totoal_cnt=totoal_cnt,
            test_aborted=test_aborted,
            error_msg=error_msg,
        )

        self.set_test_result(test_params, result_obj)
        return result_obj

    
    @staticmethod
    def get_hardware_info_str():
        """
        Retrieves the hardware name (CPU and GPU if available).

        Returns:
            str: A string combining the CPU and GPU names (e.g., "x86_64_NVIDIA GeForce RTX 4060 Laptop GPU'").
        """
        if len(DataloaderParamHelper.HARDWARE_INFO_STR) == 0:
            cpu_name = cpuinfo.get_cpu_info().get("brand_raw", "Unknown CPU")
            memory_info = psutil.virtual_memory()
            total_memory_gb = memory_info.total / (1024 ** 3)
            total_memory_gb = round(total_memory_gb, 1)
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NoGPU"
            DataloaderParamHelper.HARDWARE_INFO_STR = f"[CPU: {cpu_name}] [CPU Memory: {total_memory_gb} GB] [GPU: {gpu_name}]"

    @staticmethod
    def save_to_json(batch_size, device, params, file_path):
        """
        Saves parameters to a JSON file with batch_size, device, and hardware name as keys.

        Args:
            batch_size (int): The batch size used for the configuration.
            device (str): The device used (e.g., "cpu" or "cuda").
            params (dict): The parameters to save.
            file_path (str): The file path to save the JSON.
        """
        # Get the hardware name
        hardware_info_str = DataloaderParamHelper.HARDWARE_INFO_STR

        # Load existing data if the file exists
        try:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
        except FileNotFoundError:
            data = {}

        # Update the data with new nested structure
        if hardware_info_str not in data:
            data[hardware_info_str] = {}
        if device not in data[hardware_info_str]:
            data[hardware_info_str][device] = {}
        data[hardware_info_str][device][f"batch_size_{batch_size}"] = params

        # Save the updated data to the JSON file
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

    @staticmethod
    def load_from_json(batch_size, device, file_path):
        """
        Loads parameters from a JSON file for the given batch_size and device on the current hardware.

        Args:
            batch_size (int): The batch size used for the configuration.
            device (str): The device used (e.g., "cpu" or "cuda").
            file_path (str): The file path to load the JSON.

        Returns:
            dict: The parameters if found, or None if not found.
        """
        # Get the hardware name
        hardware_name = hardware_info_str = DataloaderParamHelper.HARDWARE_INFO_STR


        # Load existing data
        try:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
        except FileNotFoundError:
            return None

        # Retrieve parameters
        try:
            return data[hardware_name][device.type][f"batch_size_{batch_size}"]
        except KeyError:
            return None
        


DataloaderParamHelper.getInstance()  # init singleton first


def find_optimal_params(dataset, device, batch_size, result_filepath = "dataloader_params.json", load_result_file_if_exists = True, full_test=False) -> dict:
    """
    Finds the optimal dataloader parameters for the given device and batch size.

    This function evaluates the dataset loading performance for the specified device and batch size,
    and returns the best-performing dataloader parameters as a dictionary.

    Args:
        dataset: The dataset to be used for searching dataloader performance.
        device: The device (e.g., CPU or GPU) to perform the searching on.
        batch_size (int): The batch size to be used during searching.
        result_filepath (str): The file path to save or load the searching results.
        load_result_file_if_exists (bool): If True, It searches the parameter data in the 'result_filepath' 
                                            and return the result directly. Or it starts searching.
        full_test (bool): If True, performs a comprehensive full test for accuracy.
                          Defaults to False for a quicker test.

    Returns:
        dict: A dictionary containing the optimal dataloader parameters with the following structure:
            {
                "batch_size": int,
                "num_workers": int,
                "prefetch_factor": int,
                "pin_memory": bool,
                "persistent_workers": bool,
            }
    """
    DataloaderParamHelper.getInstance().set_dataset(dataset)
    return DataloaderParamHelper.getInstance().find_best_elapsed_time(device, batch_size, result_filepath="dataloader_params.json", load_result_file_if_exists=True, full_test=False)

