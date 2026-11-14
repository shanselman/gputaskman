"""Intel training-runtime library fixtures; tags do not prove model residency."""
import unittest

from gputaskman.procinfo import ProcDetail, _lib_name, _match_runtimes, classify


class IntelRuntimeTests(unittest.TestCase):
    def libraries(self, *paths):
        return _match_runtimes({name for path in paths if (name := _lib_name(path.rsplit('/', 1)[-1].lower()))})

    def test_pytorch_xpu_and_sycl_training_stack(self):
        runtimes = self.libraries('/env/site-packages/torch/lib/libtorch_xpu.so',
                                  '/env/lib/libsycl.so.8.0.0',
                                  '/runtime/lib/libze_loader.so.1.28.2')
        self.assertEqual(runtimes, ['PyTorch', 'SYCL', 'Level Zero'])
        detail = ProcDetail(123, 'python', runtimes=runtimes)
        self.assertEqual(classify(detail, {}), 'AI')
        self.assertEqual(detail.models, [])  # loaded libraries never invent model-file evidence

    def test_xpu_kernel_library_and_preview_runtime(self):
        self.assertEqual(self.libraries('torch_xpu.dll'), ['PyTorch'])
        self.assertEqual(self.libraries('/torch/lib/libtorch_xpu_ops_sycl_kernels.so'), ['PyTorch'])
        self.assertEqual(self.libraries('/env/lib/libsycl-preview.so.8'), ['SYCL'])

    def test_sycl_compute_is_not_automatically_an_ai_model(self):
        detail = ProcDetail(123, 'worker', runtimes=self.libraries('/env/lib/libsycl.so.8'))
        self.assertEqual(classify(detail, {'compute': 50}), 'compute')
        self.assertFalse(detail.is_ai)
        self.assertEqual(detail.models, [])

    def test_management_loader_alone_is_not_compute_or_ai(self):
        detail = ProcDetail(123, 'monitor', runtimes=self.libraries('/runtime/lib/libze_loader.so.1'))
        self.assertEqual(detail.runtimes, ['Level Zero'])
        self.assertEqual(classify(detail, {}), 'app')
        self.assertFalse(detail.is_ai)
        self.assertEqual(self.libraries('/env/lib/libsycl_ur_trace_collector.so',
                                        '/tmp/libsycl.so.8.0.0-gdb.py'), [])
