# import before tinygrad.
# tinygrad's runtime/ops_disk.py uses io_uring in a return annotation but only imports it on posix, so loading weights
# on Windows dies with NameError. uring() returns None off Linux before touching it, so a placeholder name is enough.
import sys, builtins, types
if sys.platform == "win32" and not hasattr(builtins, "io_uring"):
  builtins.io_uring = types.SimpleNamespace(struct_io_uring_params=object)
