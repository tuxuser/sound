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


# Hardcoded function addresses, specific to Minecraft firmware
# (PHAT/Retail offsets are listed in comments for reference)
FUNC_02C2 = 0x02BA  # PHAT/Retail: 0x02C2
FUNC_0626 = 0x061E  # PHAT/Retail: 0x0626
# VPE header pointers (0x8010+): init=0x801D, setup=0x804D, decode=0x8093
#TODO: Verify via disassembly if these are responsible for VPE decoding
FUNC_804C = 0x804C  # PHAT/Retail: 0x804C
FUNC_8092 = 0x8092  # PHAT/Retail: 0x8092


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
        """Decode a VPE/Siren7 segment using firmware emulation.
        
        Based on cc_task analysis, VPE segments have this structure:
        - Offset +0x00-0x03: codec byte and header
        - Offset +0x04-0x0F:  init structure (read by decoder)
        - Offset +0x10+: frame data stream
        """
        self.setup()
        
        if len(data) < 0x20:
            print(f"    VPE segment too small: {len(data)} bytes")
            return [], 16000

        # Write segment data to extended memory
        data_addr = EXT_MEM_BASE
        self.uc.mem_write(data_addr, data)
        
        # Extract init structure embedded in segment at offset +4
        init_struct_addr = data_addr + 4
        frame_data_start = data_addr + 0x10
        
        # Allocate state buffer (needs ~0x800 bytes per buffer_setup analysis)
        state_buf_addr = SRAM_BASE + 0x300
        self.uc.mem_write(state_buf_addr, b'\x00' * 0x800)
        
        # Parse embedded init structure to understand parameters
        # Based on firmware analysis and actual segment data:
        # Offset +0: Often 48000 or 4000
        # Offset +4: Sample rate discriminator value (compared against 7000 in decoder_init)
        #            >= 7000 (like 14000) → 16kHz
        #            < 7000 (like 7000 exactly or less) → 8kHz
        init_bytes = data[4:16]
        val_0 = struct.unpack_from('<H', init_bytes, 0)[0] if len(init_bytes) >= 2 else 0
        val_4 = struct.unpack_from('<H', init_bytes, 4)[0] if len(init_bytes) >= 6 else 0
        
        # Determine sample rate based on value at offset +4
        # decoder_init compares against UINT_00008160 = 0x1B58 = 7000
        sample_rate = 16000 if val_4 > 7000 else 8000
        
        print(f"    VPE: Analyzing embedded init struct: val_0={val_0}, val_4={val_4} → {sample_rate}Hz")
        
        # Call buffer_setup to initialize state buffer
        # Parameters per cc_task: r0=init_struct_ptr, r1=state_buffer
        try:
            self.call_function(FUNC_804C, r0=init_struct_addr, r1=state_buf_addr)
        except Exception as e:
            print(f"    VPE buffer_setup failed: {e}")
            return [], sample_rate
        
        # Read initialized state to determine samples_per_frame
        # buffer_setup uses value at offset+10 of init struct
        state_data = self.uc.mem_read(state_buf_addr, 0x20)
        init_data = self.uc.mem_read(init_struct_addr, 16)
        
        # From decoder_init: samples_per_frame is at offset 10 of init structure
        # It sets either 0x140 (320) or 0x280 (640) based on sample rate
        if len(init_data) >= 12:
            samples_per_frame = struct.unpack_from('<H', init_data, 10)[0]
            if samples_per_frame == 0 or samples_per_frame > 1024:
                samples_per_frame = 320  # Safe default
        else:
            samples_per_frame = 320
        
        # Calculate frame advance in bytes from cc_task algorithm
        # Frame advance = ((val_at_offset_6 >> 4) * 2)
        if len(init_data) >= 8:
            val_6 = struct.unpack_from('<H', init_data, 6)[0]
            frame_advance_bytes = ((val_6 >> 4) * 2)
        else:
            frame_advance_bytes = 120  # Default
            
        print(f"    VPE: samples_per_frame={samples_per_frame}, sample_rate={sample_rate}, frame_advance={frame_advance_bytes}B")
        
        # Decode frames
        # Frame data parsing is complex - the decode function reads variable-length frames
        all_samples = []
        frame_count = 0
        max_frames = 1000
        
        # Track current position in frame data
        current_frame_ptr = frame_data_start
        
        for frame_idx in range(max_frames):
            # Stop if we've reached end of segment
            if current_frame_ptr >= data_addr + len(data):
                break
            
            # Output buffer for this frame
            frame_output_addr = OUTPUT_BUF
            self.uc.mem_write(frame_output_addr, b'\x00' * (samples_per_frame * 2 + 128))
            
            # Call decode_frame function (0x8093)
            # Parameters from cc_task analysis:
            # r0 = init_struct_addr (pointer to embedded init data)
            # r1 = state_buf_addr (state/history buffer)
            # r2 = frame_data_ptr (current position in frame stream - may be updated)
            # r3 = output_buf_addr
            # Stack param [SP+0]: gain/volume (0x2000 = unity gain)
            
            # We need to track where the frame pointer advances to
            # Create a small structure to hold frame_ptr that decode can update
            frame_ptr_holder = SRAM_BASE + 0x200
            self.uc.mem_write(frame_ptr_holder, struct.pack('<I', current_frame_ptr))
            
            try:
                # Stack: push volume parameter
                stack_ptr = self.uc.reg_read(UC_ARM_REG_SP)
                stack_ptr -= 4
                self.uc.mem_write(stack_ptr, struct.pack('<I', 0x2000))  # Unity gain
                self.uc.reg_write(UC_ARM_REG_SP, stack_ptr)
                
                result = self.call_function(FUNC_8092, 
                                           r0=init_struct_addr,
                                           r1=state_buf_addr,
                                           r2=current_frame_ptr,
                                           r3=frame_output_addr,
                                           timeout=100000)
                
                # Restore stack
                self.uc.reg_write(UC_ARM_REG_SP, stack_ptr + 4)
                
            except Exception as e:
                print(f"    VPE frame {frame_idx} decode error: {e}")
                break
            
            # Read decoded samples
            output_data = self.uc.mem_read(frame_output_addr, samples_per_frame * 2)
            frame_samples = []
            for i in range(0, samples_per_frame *  2, 2):
                sample = struct.unpack_from('<h', output_data, i)[0]
                frame_samples.append(sample)
            
            # Check if frame has meaningful data
            nonzero = sum(1 for s in frame_samples if s != 0)
            if nonzero == 0 and frame_idx > 2:
                # Stop if we get multiple empty frames
                break
                
            all_samples.extend(frame_samples)
            frame_count += 1
            
            # Advance frame pointer using the calculated frame size from init structure
            # This is how cc_task advances: frame_advance = ((val_6 >> 4) * 2)
            current_frame_ptr += frame_advance_bytes
            
            # Safety: stop at reasonable audio length
            if len(all_samples) >= 160000:  # 10 seconds at 16kHz
                break
        
        print(f"    VPE: Decoded {frame_count} frames, {len(all_samples)} samples total")
        
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
