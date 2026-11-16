"""Trusted physical-device reporting without importing Torch or using an NPU."""
import builtins
import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class PhysicalDeviceTests(unittest.TestCase):
    def setUp(self):
        import blue_worker_probe
        self.probe = blue_worker_probe
        self.assertTrue(hasattr(self.probe, 'physical_device_id'), 'trusted physical device parser is missing')

    def test_launcher_assignment_is_parsed_without_defaulting_or_restricting_historical_cards(self):
        for number in (0, 3, 6, 63):
            with self.subTest(number=number), patch.dict(os.environ,
                    {'LLM_OPT_PHYSICAL_NPU_ID': str(number)}, clear=True):
                self.assertEqual(self.probe.physical_device_id(), number)
        with patch.dict(os.environ, {'LLM_OPT_PHYSICAL_NPU_ID': '0'}, clear=True):
            self.assertEqual(self.probe.physical_device_id(), 0)
            os.environ['LLM_OPT_PHYSICAL_NPU_ID'] = '3'
            self.assertEqual(self.probe.physical_device_id(), 3)

    def test_missing_malformed_and_out_of_range_assignment_fails_closed(self):
        for value in (None, '', ' ', 'wrong', '1.0', '-1', '64'):
            environment = {} if value is None else {'LLM_OPT_PHYSICAL_NPU_ID': value}
            with self.subTest(value=value), patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(RuntimeError): self.probe.physical_device_id()

    def test_actual_probe_rejects_missing_identity_before_importing_torch(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == 'torch': raise AssertionError('hardware import preceded device identity validation')
            return original(name, *args, **kwargs)
        with patch.dict(os.environ, {}, clear=True), patch('builtins.__import__', side_effect=guarded):
            with self.assertRaises(RuntimeError): self.probe.probe(SimpleNamespace())

    def test_actual_probe_reports_physical_assignment_even_with_logical_device_zero(self):
        torch = ModuleType('torch')
        logical_devices = []
        def device_name(logical):
            logical_devices.append(logical)
            return 'Ascend910B3'
        torch.npu = SimpleNamespace(mem_get_info=lambda: (8, 16), memory_allocated=lambda: 1,
            memory_reserved=lambda: 2, max_memory_allocated=lambda: 3, max_memory_reserved=lambda: 4,
            get_device_name=device_name)
        modules = {name: None for name in sys.modules if name == 'vllm' or name.startswith('vllm.')
                   or name == 'vllm_ascend' or name.startswith('vllm_ascend.')}
        modules['torch'] = torch
        worker = SimpleNamespace(model_runner=SimpleNamespace(drafter=None))
        with patch.dict(sys.modules, modules), patch.dict(os.environ,
                {'LLM_OPT_PHYSICAL_NPU_ID': '6'}, clear=True):
            result = self.probe.probe(worker)
        self.assertEqual(result['physical_device'], 6)
        self.assertIs(type(result['physical_device']), int)
        self.assertEqual(logical_devices, [0])


if __name__ == '__main__': unittest.main()
