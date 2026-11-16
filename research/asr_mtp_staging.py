# SPDX-License-Identifier: Apache-2.0
"""Persistent host staging with one H2D lifetime fence per auxiliary program.

Used outside capture, serially on the runner stream. The fence protects CPU
storage reuse, not graph completion; stream order protects device storage.
"""

import torch


class UploadBatch:
    def __init__(self, device, *, event=None):
        self.device = torch.device(device)
        self.event = event
        if self.event is None and self.device.type == "npu":
            self.event = torch.npu.Event()
        self.active = self.pending = self.dirty = False

    def buffer(self, target):
        return UploadBuffer(self, target)

    def __enter__(self):
        if self.active:
            raise RuntimeError("nested MTP upload batch")
        self.active = True
        return self

    def before_write(self):
        # Defer the query until a write is actually needed: unchanged tables
        # alone do not query, synchronize or record an event.
        if self.pending:
            if not self.event.query():
                self.event.synchronize()
            self.pending = False

    def __exit__(self, *exc):
        try:
            # Also fence successful copies preceding a later staging error.
            if self.dirty and self.event is not None:
                self.event.record()
                self.pending = True
        finally:
            self.active = self.dirty = False


class UploadBuffer:
    def __init__(self, owner, target):
        self.owner, self.target = owner, target
        self.host = torch.empty(
            target.shape,
            dtype=target.dtype,
            device="cpu",
            pin_memory=owner.device.type == "npu",
        )
        self.array = self.host.numpy()

    def copy(self, values):
        if not self.owner.active:
            with self.owner:
                self.copy(values)
            return
        self.owner.before_write()
        self.array[...] = values
        self.upload()

    def upload(self):
        """Copy staged bytes after the caller fences and fills the host views."""
        if not self.owner.active:
            raise RuntimeError("direct MTP upload requires an active batch")
        self.owner.dirty = True
        self.target.copy_(self.host, non_blocking=True)
