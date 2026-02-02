#!/usr/bin/env python3
"""
ISD9160 Unified Audio Decoder

Extracts and decodes audio segments from the full firmware binary.
Uses the segment index table from the firmware itself (no hardcoded offsets).

This combines segment extraction with emulator-based decoding for accurate results.
"""

import sys
import struct
import wave
from pathlib import Path
from typing import List, Tuple

from unicorn import *
from unicorn.arm_const import *

from parser import FirmwareParser


# Memory layout matching the firmware
APROM_BASE = 0x00000000
APROM_SIZE = 0x8000  # 32KB

VPE_BASE = 0x00008000
VPE_SIZE = 0x1C400  # ~113KB

SRAM_BASE = 0x20000000
SRAM_SIZE = 0x3000  # 12KB

STACK_TOP = SRAM_BASE + SRAM_SIZE

PERIPH_BASE = 0x40000000
PERIPH_SIZE = 0x100000

EXT_MEM_BASE = 0x10000000
EXT_MEM_SIZE = 0x40000  # 256KB

OUTPUT_BUF = SRAM_BASE + 0x1000
OUTPUT_SIZE = 0x1000

WORK_BUF = SRAM_BASE + 0x100
WORK_SIZE = 0x200


# Hardcoded function adresses, specific to PHAT / Retail firmware
FUNC_02C2 = 0x02C2 # minecraft: 0x02BA
FUNC_0626 = 0x0626 # minecraft: 0x061E
FUNC_804C = 0x804C
FUNC_8092 = 0x8092


class ISD9160Emulator:
    """Emulates the ISD9160 decoder using Unicorn."""

    def __init__(self, firmware: bytes):
        self.firmware = firmware
        self.aprom = firmware[0:APROM_SIZE]
        self.vpe = firmware[APROM_SIZE:APROM_SIZE + VPE_SIZE]
        self.uc = None

    def setup(self):
        """Initialize the emulator."""
        self.uc = Uc(UC_ARCH_ARM, UC_MODE_THUMB)

        # Map memory regions
        self.uc.mem_map(APROM_BASE, APROM_SIZE)
        self.uc.mem_map(VPE_BASE, VPE_SIZE)
        self.uc.mem_map(SRAM_BASE, SRAM_SIZE)
        self.uc.mem_map(PERIPH_BASE, PERIPH_SIZE)
        self.uc.mem_map(EXT_MEM_BASE, EXT_MEM_SIZE)

        # Load firmware
        self.uc.mem_write(APROM_BASE, self.aprom)
        self.uc.mem_write(VPE_BASE, self.vpe)

        # Initialize stack
        self.uc.reg_write(UC_ARM_REG_SP, STACK_TOP)

        # Zero SRAM
        self.uc.mem_write(SRAM_BASE, b'\x00' * SRAM_SIZE)

    def call_function(self, addr: int, r0: int = 0, r1: int = 0, r2: int = 0, r3: int = 0,
                      timeout: int = 10000) -> int:
        """Call a function at the given address."""
        return_addr = SRAM_BASE + 0x10
        self.uc.mem_write(return_addr, b'\x00\xbe')  # BKPT

        self.uc.reg_write(UC_ARM_REG_R0, r0)
        self.uc.reg_write(UC_ARM_REG_R1, r1)
        self.uc.reg_write(UC_ARM_REG_R2, r2)
        self.uc.reg_write(UC_ARM_REG_R3, r3)
        self.uc.reg_write(UC_ARM_REG_LR, return_addr | 1)

        try:
            self.uc.emu_start(addr | 1, return_addr, timeout=timeout * UC_SECOND_SCALE)
        except UcError as e:
            if e.errno != UC_ERR_FETCH_UNMAPPED:
                raise

        return self.uc.reg_read(UC_ARM_REG_R0)

    def decode_dpcm(self, data: bytes) -> Tuple[List[int], int]:
        """Decode a DPCM segment using firmware emulation."""
        self.setup()

        # Write segment data to extended memory
        data_addr = EXT_MEM_BASE
        self.uc.mem_write(data_addr, data)
        end_addr = data_addr + len(data)

        # Context structure
        ctx_addr = WORK_BUF
        self.uc.mem_write(ctx_addr, b'\x00' * 0x80)

        # Initialize with FUN_000002c2
        try:
            self.call_function(FUNC_02C2, r0=data_addr, r1=end_addr, r2=ctx_addr)
        except Exception as e:
            print(f"    Init failed: {e}")
            return [], 16000

        # Read samples_per_frame from context
        ctx = self.uc.mem_read(ctx_addr, 0x80)
        samples_per_frame = struct.unpack_from('<I', ctx, 0x48)[0]
        if samples_per_frame == 0:
            samples_per_frame = 256

        # Decode frames
        all_samples = []
        frame_count = 0
        max_frames = 10000

        while frame_count < max_frames:
            # Set output buffer
            frame_output_addr = OUTPUT_BUF
            ctx = bytearray(self.uc.mem_read(ctx_addr, 0x80))
            struct.pack_into('<I', ctx, 0x4C, frame_output_addr)
            self.uc.mem_write(ctx_addr, bytes(ctx))

            # Zero output buffer
            self.uc.mem_write(frame_output_addr, b'\x00' * (samples_per_frame * 2 + 64))

            # Call FUN_00000626
            try:
                result = self.call_function(FUNC_0626, r0=ctx_addr, r1=frame_output_addr)
            except Exception as e:
                print(f"    Frame {frame_count} error: {e}")
                break

            # Read decoded samples
            output_data = self.uc.mem_read(frame_output_addr, samples_per_frame * 2)
            for i in range(0, samples_per_frame * 2, 2):
                sample = struct.unpack_from('<h', output_data, i)[0]
                all_samples.append(sample)

            frame_count += 1

            if result == 0:
                break

            # Check position
            ctx = self.uc.mem_read(ctx_addr, 0x80)
            current_pos = struct.unpack_from('<I', ctx, 0)[0]
            end_pos = struct.unpack_from('<I', ctx, 8)[0]
            if current_pos >= end_pos:
                break

        return all_samples, 16000

    def decode_vpe(self, data: bytes) -> Tuple[List[int], int]:
        """Decode a VPE/Siren7 segment using firmware emulation."""
        self.setup()

        # Write segment data
        data_addr = EXT_MEM_BASE
        self.uc.mem_write(data_addr, data)

        # Parse VPE header (at offset 4 in segment)
        header_addr = data_addr + 4
        encoded_data_addr = data_addr + 16

        header = data[4:20]
        sample_rate = struct.unpack_from('<H', header, 0)[0]
        bits_per_frame = struct.unpack_from('<H', header, 6)[0]
        samples_per_frame = struct.unpack_from('<H', header, 10)[0]

        if samples_per_frame == 0:
            samples_per_frame = 320
        if bits_per_frame == 0:
            bits_per_frame = 80
        if sample_rate == 0:
            sample_rate = 4000

        # Work buffer
        work_addr = SRAM_BASE + 0x2000
        self.uc.mem_write(work_addr, b'\x00' * 0x1000)

        # Calculate frames
        encoded_size = len(data) - 16
        bytes_per_frame = (bits_per_frame + 7) // 8
        if bytes_per_frame == 0:
            bytes_per_frame = 10
        num_frames = encoded_size // bytes_per_frame if bytes_per_frame > 0 else 1

        # Initialize VPE decoder
        try:
            self.call_function(FUNC_804C, r0=header_addr, r1=work_addr)
        except Exception as e:
            print(f"    VPE init failed: {e}")
            return [], sample_rate

        # Decode frames
        all_samples = []
        current_data = encoded_data_addr
        volume = 0x2000

        for frame_idx in range(num_frames):
            output_addr = OUTPUT_BUF
            self.uc.mem_write(output_addr, b'\x00' * (samples_per_frame * 2 + 64))

            try:
                sp = self.uc.reg_read(UC_ARM_REG_SP)
                self.uc.mem_write(sp - 4, struct.pack('<I', volume))
                self.uc.reg_write(UC_ARM_REG_SP, sp - 4)

                self.call_function(FUNC_8092, r0=header_addr, r1=work_addr,
                                   r2=current_data, r3=output_addr)

                self.uc.reg_write(UC_ARM_REG_SP, sp)
            except Exception as e:
                print(f"    VPE frame {frame_idx} failed: {e}")
                break

            # Read samples
            output_data = self.uc.mem_read(output_addr, samples_per_frame * 2)
            for i in range(0, samples_per_frame * 2, 2):
                sample = struct.unpack_from('<h', output_data, i)[0]
                all_samples.append(sample)

            current_data += bytes_per_frame

        return all_samples, sample_rate


def save_wav(samples: List[int], output_path: Path, sample_rate: int = 16000):
    """Save samples as WAV file."""
    if not samples:
        return False

    # Clamp samples
    samples = [max(-32768, min(32767, s)) for s in samples]

    with wave.open(str(output_path), 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f'<{len(samples)}h', *samples))

    duration = len(samples) / sample_rate
    nonzero = sum(1 for s in samples if s != 0)
    print(f"    Saved {len(samples)} samples ({duration:.2f}s), {nonzero} non-zero")
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [firmware.bin]")
        return 1

    firmware_path = Path(sys.argv[1])

    base_dir = Path(__file__).parent
    output_dir = base_dir / "decoded_audio_emu"
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("ISD9160 Unified Audio Decoder")
    print("=" * 70)
    print()
    print("Extracts segments using index table")
    print("Decodes using Unicorn emulation of original firmware functions")
    print()

    if not firmware_path.exists():
        print(f"Error: Firmware not found: {firmware_path}")
        return 2

    # Parse firmware
    print("Parsing firmware...")
    parser = FirmwareParser(firmware_path)
    segments = parser.parse_segments()
    print()

    # Initialize emulator
    emu = ISD9160Emulator(parser.firmware)

    # Decode each segment
    print("=" * 70)
    print("Decoding segments...")
    print("=" * 70)

    decoded_count = 0
    failed_count = 0

    for segment in segments:
        print(f"\nSegment {segment.index}: {segment.size} bytes, codec {segment.codec}")

        # Determine decoder based on codec type
        if segment.codec_lower5 in (0x1D, 0x1E):
            print("  Format: VPE/Siren7")
            samples, sample_rate = emu.decode_vpe(segment.data)
        else:
            print("  Format: DPCM")
            samples, sample_rate = emu.decode_dpcm(segment.data)

        if samples:
            out_path = output_dir / f"segment_{segment.index:02d}.wav"
            if save_wav(samples, out_path, sample_rate):
                decoded_count += 1
            else:
                failed_count += 1
        else:
            print("    No samples decoded")
            failed_count += 1

    print()
    print("=" * 70)
    print(f"Results: {decoded_count} decoded, {failed_count} failed")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
