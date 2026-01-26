#!/usr/bin/env python3
"""
ISD9160 Standalone Audio Decoder

Pure Python implementation of the DPCM decoder. (no worky yet)

Based on analysis of firmware functions:
- FUN_000002c2: Initialization (sets samples_per_frame = 256)
- FUN_00000626: Main decode loop (reads control bytes, dispatches to frame decoders)
- FUN_00000428: Frame decoder with delta prediction
- FUN_00000392: Simple frame decoder (direct bit expansion)
- FUN_000005e0: Initial frame decoder
- FUN_000003ee: Delta decoder with step table
- FUN_0000033a: Bit reader (MSB first)
"""

import sys
import struct
import wave
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

from parser import FirmwareParser

# Step table from firmware at 0x71D0
STEP_TABLE = [0x0000, 0x0084, 0x018C, 0x039C, 0x07BC, 0x0FFC, 0x207C, 0x417C]

# Constants
MAX_SAMPLE = 0x7FFF
MIN_SAMPLE = -0x8000
DEFAULT_SAMPLES_PER_FRAME = 256

@dataclass
class DecoderState:
    """Decoder state matching firmware context structure."""
    data: bytes
    pos: int = 0
    end_pos: int = 0
    bits_left: int = 8
    bits_per_sample: int = 4
    samples_per_frame: int = DEFAULT_SAMPLES_PER_FRAME
    predictor: int = 0
    is_first_frame: bool = True
    frame_count: int = 0


class BitReader:
    """Bit reader matching FUN_0000033a - reads MSB first."""

    def __init__(self, state: DecoderState):
        self.state = state

    def read_bits(self, n: int) -> int:
        """Read n bits from stream (MSB first)."""
        if n == 0:
            return 0

        result = 0
        bits_needed = n

        while bits_needed > 0 and self.state.pos < self.state.end_pos:
            current_byte = self.state.data[self.state.pos]

            if bits_needed <= self.state.bits_left:
                shift = self.state.bits_left - bits_needed
                mask = (1 << bits_needed) - 1
                result |= (current_byte >> shift) & mask
                self.state.bits_left -= bits_needed
                if self.state.bits_left == 0:
                    self.state.pos += 1
                    self.state.bits_left = 8
                break
            else:
                mask = (1 << self.state.bits_left) - 1
                result |= (current_byte & mask) << (bits_needed - self.state.bits_left)
                bits_needed -= self.state.bits_left
                self.state.pos += 1
                self.state.bits_left = 8

        return result

    def align_byte(self):
        """Align to next byte boundary."""
        if self.state.bits_left != 8:
            self.state.pos += 1
            self.state.bits_left = 8

    def read_byte(self) -> int:
        """Read aligned byte."""
        self.align_byte()
        if self.state.pos >= self.state.end_pos:
            return 0
        b = self.state.data[self.state.pos]
        self.state.pos += 1
        return b

    def peek_byte(self) -> int:
        """Peek at next aligned byte."""
        if self.state.bits_left != 8:
            pos = self.state.pos + 1
        else:
            pos = self.state.pos
        if pos >= self.state.end_pos:
            return 0
        return self.state.data[pos]


def decode_delta(code: int, bits_per_sample: int) -> int:
    """
    Delta decoder matching FUN_000003ee.

    code format:
    - bit 0: sign (1 = negative)
    - bits 1-3: step table index
    - bits 4+: magnitude

    delta = step_table[index] + (magnitude << (8 - bps + index + 3))
    """
    index = (code >> 1) & 0x7
    sign = code & 1
    magnitude = code >> 4

    step = STEP_TABLE[index]
    shift = (8 - bits_per_sample) + index + 3

    if shift > 0:
        delta = step + (magnitude << shift)
    else:
        delta = step + magnitude

    if sign:
        delta = -delta

    return delta


def clamp(value: int) -> int:
    """Clamp to 16-bit signed range."""
    return max(MIN_SAMPLE, min(MAX_SAMPLE, value))


def decode_frame_predicted(state: DecoderState, reader: BitReader) -> List[int]:
    """
    Decode frame with prediction (FUN_00000428).

    First frame: sample = delta
    Later frames: sample = delta*2 + predictor*4
    predictor = sample >> 2
    """
    samples = []

    for _ in range(state.samples_per_frame):
        if state.pos >= state.end_pos:
            samples.append(0)
            continue

        code = reader.read_bits(state.bits_per_sample)
        delta = decode_delta(code, state.bits_per_sample)

        if state.is_first_frame:
            sample = delta
        else:
            sample = delta * 2 + state.predictor * 4
            sample = clamp(sample)

        samples.append(sample)

        # Update predictor (arithmetic right shift with rounding toward zero)
        if sample >= 0:
            state.predictor = sample >> 2
        else:
            state.predictor = -((-sample) >> 2)

    state.is_first_frame = False
    state.frame_count += 1
    return samples


def decode_frame_simple(state: DecoderState, reader: BitReader) -> List[int]:
    """
    Simple frame decoder (FUN_00000392).
    Just shift bits to 16-bit range.
    """
    samples = []
    shift = 16 - state.bits_per_sample

    for _ in range(state.samples_per_frame):
        if state.pos >= state.end_pos:
            samples.append(0)
            continue

        code = reader.read_bits(state.bits_per_sample)
        sample = code << shift

        # Sign extend
        if sample >= 0x8000:
            sample -= 0x10000

        samples.append(sample)

    # Update predictor from last sample
    if samples:
        last = samples[-1]
        state.predictor = last >> 2 if last >= 0 else -((-last) >> 2)

    state.frame_count += 1
    return samples


def decode_dpcm_segment(data: bytes, debug: bool = False) -> Tuple[List[int], int]:
    """
    Decode a DPCM segment.

    Control byte format (from decode_frame_main @ 0x626):
    - bits 3-4 (mode): 0x00=end check, 0x08=set params, 0x10=dpcm frame, 0x18=simple frame
    - bits 0-2 (value): mode-specific parameter

    Key insight from firmware:
    - If lower5 == 0x1C: control bytes are read from data stream
    - If lower5 != 0x1C: first byte is reused as control byte for all frames
    """
    if len(data) < 4:
        return [], 16000

    # Parse first byte
    first_byte = data[0]
    lower5 = first_byte & 0x1F

    # Determine if control bytes come from stream or are fixed
    stream_control = (lower5 == 0x1C)

    # Initialize state
    state = DecoderState(
        data=data,
        pos=1,  # Skip first byte (codec identifier)
        end_pos=len(data),
        bits_per_sample=4,
        samples_per_frame=DEFAULT_SAMPLES_PER_FRAME,
    )
    reader = BitReader(state)
    all_samples = []

    if debug:
        print(f"  lower5=0x{lower5:02X}, stream_control={stream_control}")

    max_frames = 10000
    max_iterations = 100000
    iterations = 0

    while state.pos < state.end_pos and state.frame_count < max_frames and iterations < max_iterations:
        iterations += 1

        # Align to byte boundary
        reader.align_byte()

        if state.pos >= state.end_pos:
            break

        # Read control byte based on mode
        if stream_control:
            # Control bytes come from data stream
            control = reader.read_byte()
        else:
            # First byte is reused as control byte (don't advance position)
            control = first_byte

        mode = control & 0x18
        value = control & 0x07

        if debug and state.frame_count < 3:
            print(f"  Frame {state.frame_count}: control=0x{control:02X}, mode=0x{mode:02X}, value={value}")

        if mode == 0x00:
            # Mode 0x00: End check - check if we've reached end
            if not stream_control:
                # For fixed control mode, mode 0 means we should stop
                break
            continue

        elif mode == 0x08:
            # Set parameters mode (only used with stream_control)
            # value = bits_per_sample (2-7 typically)
            state.bits_per_sample = value if value > 0 else 4

            # Read two parameter bytes
            if state.pos + 2 > state.end_pos:
                break
            param1 = reader.read_byte()
            param2 = reader.read_byte()

            # param1 >> 2 encodes samples_per_frame / 4
            spf = (param1 >> 2) & 0x3F
            if spf > 0:
                state.samples_per_frame = spf * 4

            if debug:
                print(f"    Mode 0x08: bps={state.bits_per_sample}, spf={state.samples_per_frame}")

            # Decode initial frame (uses decode_frame_initial in firmware)
            # Firmware doesn't pre-check data size - decoder handles end-of-data
            state.is_first_frame = True
            samples = decode_frame_predicted(state, reader)
            all_samples.extend(samples)

        elif mode == 0x10:
            # DPCM frame mode - uses value to select bits_per_sample
            # Based on switch table at 0x72b
            bits_map = {0: 6, 1: 7, 2: 6, 3: 6, 4: 6, 5: 8, 6: 8, 7: 8}
            state.bits_per_sample = bits_map.get(value, 6)

            # Decode DPCM frame with prediction
            # Firmware doesn't pre-check - decoder fills zeros if data runs out
            samples = decode_frame_predicted(state, reader)
            all_samples.extend(samples)

        elif mode == 0x18:
            # Simple bit expansion mode
            bits_map = {0: 8, 1: 10, 2: 16, 3: 12}
            state.bits_per_sample = bits_map.get(value, 8)

            # Decode simple frame
            # Firmware doesn't pre-check - decoder fills zeros if data runs out
            samples = decode_frame_simple(state, reader)
            all_samples.extend(samples)

        else:
            # Unknown mode - stop
            break

    if debug:
        print(f"  Decoded {state.frame_count} frames, {len(all_samples)} samples")

    return all_samples, 16000


def decode_vpe_segment(data: bytes) -> Tuple[List[int], int]:
    """
    VPE/Siren7 codec - too complex for standalone implementation.
    Returns empty to indicate emulator is needed.
    """
    if len(data) < 20:
        return [], 4000

    header = data[4:20]
    sample_rate = struct.unpack_from('<H', header, 0)[0]
    if sample_rate == 0:
        sample_rate = 4000

    print(f"  VPE codec requires emulation (rate={sample_rate}Hz)")
    return [], sample_rate

def save_wav(samples: List[int], output_path: Path, sample_rate: int = 16000):
    """Save samples as WAV file."""
    if not samples:
        print("  No samples to save")
        return

    samples = [clamp(s) for s in samples]

    with wave.open(str(output_path), 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f'<{len(samples)}h', *samples))

    duration = len(samples) / sample_rate
    nonzero = sum(1 for s in samples if s != 0)
    print(f"  Saved {len(samples)} samples ({duration:.2f}s), {nonzero} non-zero ({100*nonzero/len(samples):.1f}%)")


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [firmware.bin]")
        return 1

    firmware_path = Path(sys.argv[1])

    base_dir = Path(__file__).parent
    output_dir = base_dir / "decoded_audio_standalone"

    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("ISD9160 Standalone Audio Decoder")
    print("=" * 60)
    print()
    print("Pure Python DPCM decoder based on firmware reverse engineering.")
    print("Note: VPE segments require emulator for accurate decode.")
    print()

    if not firmware_path.exists():
        print(f"Error: Firmware not found: {firmware_path}")
        return 2

    # Parse firmware
    print("Parsing firmware...")
    parser = FirmwareParser(firmware_path)
    segments = parser.parse_segments()
    print()

    decoded = 0
    skipped = 0

    for idx, segment in enumerate(segments):
        print(f"\nSegment {segment.index}: {segment.size} bytes, codec {segment.codec}")

        if segment.size < 4:
            print("  Segment too small")
            return 3

        data = segment.data
        first_byte = data[0]

        print(f"  Codec byte: 0x{first_byte:02X} (codec={segment.codec}, lower5=0x{segment.codec_lower5:02X})")

        # VPE check
        if segment.codec_lower5 in (0x1D, 0x1E):
            print("  Format: VPE/Siren7")
            samples, sample_rate = decode_vpe_segment(data)
        else:
            print("  Format: DPCM")
            samples, sample_rate = decode_dpcm_segment(data)


        if samples:
            out_name =  f"seg_{idx:02d}_standalone.wav"
            out_path = output_dir / out_name
            save_wav(samples, out_path, sample_rate)
            decoded += 1
        else:
            print("  Skipped")
            skipped += 1

    print()
    print("=" * 60)
    print(f"Results: {decoded} decoded, {skipped} skipped (VPE)")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    raise NotImplementedError("Output is pure noise right now")

    return 0

if __name__ == "__main__":
    sys.exit(main())
