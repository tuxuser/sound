#!/usr/bin/env python3
"""
ISD9160 Standalone DPCM Audio Decoder

Pure Python implementation based on Ghidra reverse engineering.

Functions mapped from firmware:
- bitreader_read_bits (0x033A): Reads N bits MSB first
- decode_delta_step (0x03EE): DPCM delta decoder using step table
- decode_frame_dpcm (0x0428): DPCM frame decoder with prediction
- decode_frame_simple (0x0392): Simple bit expansion decoder
- decode_frame_initial (0x05E0): ADPCM initial frame decoder
- decode_adpcm_sample (0x04EC): ADPCM sample decoder
- decode_frame_main (0x0626): Main decoder loop
- codec_dispatcher_init (0x02C2): Codec initialization
"""

import sys
import struct
import wave
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

from parser import FirmwareParser

# DPCM Step table from firmware at 0x71D0 (8 x 16-bit values)
DPCM_STEP_TABLE = [0x0000, 0x0084, 0x018C, 0x039C, 0x07BC, 0x0FFC, 0x207C, 0x417C]

# ADPCM Step table from firmware at 0x7150 (64 x 16-bit values)
ADPCM_STEP_TABLE = [
    0x0020, 0x0024, 0x0028, 0x002C, 0x0030, 0x0034, 0x0038, 0x003C,
    0x0040, 0x0048, 0x0050, 0x0058, 0x0060, 0x0068, 0x0070, 0x0078,
    0x0080, 0x0090, 0x00A0, 0x00B0, 0x00C0, 0x00D0, 0x00E0, 0x00F0,
    0x0100, 0x0120, 0x0140, 0x0160, 0x0180, 0x01A0, 0x01C0, 0x01E0,
    0x0200, 0x0240, 0x0280, 0x02C0, 0x0300, 0x0340, 0x0380, 0x03C0,
    0x0400, 0x0480, 0x0500, 0x0580, 0x0600, 0x0680, 0x0700, 0x0780,
    0x0800, 0x0900, 0x0A00, 0x0B00, 0x0C00, 0x0D00, 0x0E00, 0x0F00,
    0x1000, 0x1200, 0x1400, 0x1600, 0x1800, 0x1A00, 0x1C00, 0x1E00,
]

# ADPCM step index adaptation table (signed values)
# Used to adjust step_index based on encoded sample bits
ADPCM_INDEX_TABLE = [
    -1, -1, -1, -1, 2, 4, 6, 8,
    -1, -1, -1, -1, 2, 4, 6, 8,
]

# Constants
MAX_SAMPLE = 0x7FFF
MIN_SAMPLE = -0x8000
DEFAULT_SAMPLES_PER_FRAME = 256


@dataclass
class DecoderContext:
    """Decoder context matching firmware structure."""
    data: bytes
    pos: int = 0
    end_pos: int = 0
    bits_left: int = 8

    # Decoder state
    bits_per_sample: int = 6
    samples_per_frame: int = DEFAULT_SAMPLES_PER_FRAME

    # Predictor state (for DPCM mode 0x10)
    predictor: int = 0
    prev_predictor: int = 0

    # ADPCM state (for mode 0x08)
    step_index: int = 0
    adpcm_predictor: int = 0

    # Control mode
    stream_control: bool = True
    fixed_control: int = 0

    # Frame tracking
    frame_count: int = 0


def clamp16(value: int) -> int:
    """Clamp to signed 16-bit range."""
    if value > MAX_SAMPLE:
        return MAX_SAMPLE
    if value < MIN_SAMPLE:
        return MIN_SAMPLE
    return value


def div4_round_to_zero(value: int) -> int:
    """Divide by 4 with rounding toward zero."""
    if value >= 0:
        return value >> 2
    else:
        return -(-value >> 2)


class BitReader:
    """Bit reader matching bitreader_read_bits (0x033A)."""

    def __init__(self, ctx: DecoderContext):
        self.ctx = ctx

    def read_bits(self, n: int) -> int:
        """Read n bits from stream (MSB first)."""
        if n == 0:
            return 0

        result = 0
        bits_needed = n

        while bits_needed > 0:
            if self.ctx.pos >= self.ctx.end_pos:
                return result

            current_byte = self.ctx.data[self.ctx.pos]
            bits_available = self.ctx.bits_left

            if bits_needed <= bits_available:
                shift = bits_available - bits_needed
                mask = (1 << bits_needed) - 1
                result |= (current_byte >> shift) & mask
                self.ctx.bits_left = shift
                if self.ctx.bits_left == 0:
                    self.ctx.pos += 1
                    self.ctx.bits_left = 8
                break
            else:
                mask = (1 << bits_available) - 1
                result |= (current_byte & mask) << (bits_needed - bits_available)
                bits_needed -= bits_available
                self.ctx.pos += 1
                self.ctx.bits_left = 8

        return result

    def align_byte(self):
        """Align to next byte boundary."""
        if self.ctx.bits_left != 8:
            self.ctx.pos += 1
            self.ctx.bits_left = 8

    def read_byte(self) -> int:
        """Read one aligned byte."""
        self.align_byte()
        if self.ctx.pos >= self.ctx.end_pos:
            return 0
        b = self.ctx.data[self.ctx.pos]
        self.ctx.pos += 1
        return b


def decode_delta_step(encoded: int, bits_per_sample: int) -> int:
    """
    DPCM delta decoder matching decode_delta_step (0x03EE).

    Encoded format:
    - bit 0: sign (1 = negative)
    - bits 1-3: step_index (0-7)
    - bits 4+: magnitude
    """
    sign = encoded & 1
    step_index = (encoded >> 1) & 7
    magnitude = encoded >> 4

    shift1 = 8 - bits_per_sample
    shift2 = step_index + 3

    scaled_magnitude = (magnitude << shift1) << shift2
    delta = DPCM_STEP_TABLE[step_index] + scaled_magnitude

    delta = delta & 0xFFFF
    if delta >= 0x8000:
        delta -= 0x10000

    if sign:
        delta = -delta

    return delta


def decode_adpcm_sample(ctx: DecoderContext, encoded: int) -> int:
    """
    ADPCM sample decoder matching decode_adpcm_sample (0x04EC).

    Uses adaptive step size based on step_index.
    """
    # Get current step size
    step_index = max(0, min(len(ADPCM_STEP_TABLE) - 1, ctx.step_index))
    step = ADPCM_STEP_TABLE[step_index]

    # Decode delta using step
    # Standard ADPCM algorithm
    delta = 0
    sign = encoded & (1 << (ctx.bits_per_sample - 1))
    code = encoded & ((1 << (ctx.bits_per_sample - 1)) - 1)

    # Compute delta from code bits
    step_shift = step
    for i in range(ctx.bits_per_sample - 2, -1, -1):
        if code & (1 << i):
            delta += step_shift
        step_shift >>= 1
    delta += step_shift  # Add final half step

    # Apply sign
    if sign:
        delta = -delta

    # Update predictor
    sample = ctx.adpcm_predictor + delta
    sample = clamp16(sample)
    ctx.adpcm_predictor = sample

    # Update step index using adaptation table
    index_adjust = ADPCM_INDEX_TABLE[encoded & 0x0F] if encoded < 16 else 0
    ctx.step_index = max(0, min(len(ADPCM_STEP_TABLE) - 1, ctx.step_index + index_adjust))

    return sample


def decode_frame_adpcm(ctx: DecoderContext, reader: BitReader) -> List[int]:
    """
    ADPCM frame decoder for mode 0x08.
    """
    samples = []

    for _ in range(ctx.samples_per_frame):
        if ctx.pos >= ctx.end_pos:
            samples.append(0)
            continue

        code = reader.read_bits(ctx.bits_per_sample)
        sample = decode_adpcm_sample(ctx, code)
        samples.append(sample)

    # Update DPCM predictor from ADPCM state for subsequent mode 0x10 frames
    ctx.predictor = div4_round_to_zero(ctx.adpcm_predictor)

    ctx.frame_count += 1
    return samples


def decode_frame_dpcm(ctx: DecoderContext, reader: BitReader, use_prediction: bool) -> List[int]:
    """
    DPCM frame decoder matching decode_frame_dpcm (0x0428).
    """
    samples = []

    for _ in range(ctx.samples_per_frame):
        if ctx.pos >= ctx.end_pos:
            samples.append(0)
            continue

        code = reader.read_bits(ctx.bits_per_sample)
        delta = decode_delta_step(code, ctx.bits_per_sample)

        if use_prediction:
            sample = delta * 2 + ctx.predictor * 4
            sample = clamp16(sample)
            ctx.predictor = div4_round_to_zero(sample)
        else:
            sample = delta

        samples.append(sample)

    if len(samples) >= 2:
        ctx.prev_predictor = div4_round_to_zero(samples[-2])

    ctx.frame_count += 1
    return samples


def decode_frame_simple(ctx: DecoderContext, reader: BitReader) -> List[int]:
    """
    Simple frame decoder matching decode_frame_simple (0x0392).
    """
    samples = []
    shift = 16 - ctx.bits_per_sample

    for _ in range(ctx.samples_per_frame):
        if ctx.pos >= ctx.end_pos:
            samples.append(0)
            continue

        code = reader.read_bits(ctx.bits_per_sample)
        sample = code << shift

        # Sign extend
        sign_bit = 1 << (ctx.bits_per_sample - 1)
        if code & sign_bit:
            sample = sample - (1 << 16)

        samples.append(sample)

    if len(samples) >= 2:
        ctx.predictor = div4_round_to_zero(samples[-1])
        ctx.prev_predictor = div4_round_to_zero(samples[-2])

    ctx.frame_count += 1
    return samples


def decode_dpcm_segment(data: bytes, debug: bool = False) -> Tuple[List[int], int]:
    """
    Decode a DPCM segment matching decode_frame_main (0x0626).
    """
    if len(data) < 2:
        return [], 16000

    first_byte = data[0]
    lower5 = first_byte & 0x1F

    stream_control = (lower5 == 0x1C)

    ctx = DecoderContext(
        data=data,
        pos=1,
        end_pos=len(data),
        bits_left=8,
        bits_per_sample=6,
        samples_per_frame=DEFAULT_SAMPLES_PER_FRAME,
        stream_control=stream_control,
        fixed_control=first_byte,
    )

    reader = BitReader(ctx)
    all_samples = []

    if debug:
        print(f"  first_byte=0x{first_byte:02X}, lower5=0x{lower5:02X}, stream_control={stream_control}")

    # Mode 0x10 switch table from Ghidra disassembly at 0x72b:
    # value -> (bits_per_sample, use_prediction, is_end)
    # Values 2, 6, 7 terminate decoding (return 0)
    dpcm_mode_table = {
        0: (7, False, False),   # 7 bits, no prediction
        1: (8, False, False),   # 8 bits, no prediction
        2: (0, False, True),    # END DECODING
        3: (6, True, False),    # 6 bits, with prediction
        4: (7, True, False),    # 7 bits, with prediction
        5: (7, True, False),    # 8 bits, with prediction
        6: (0, False, True),    # END DECODING
        7: (0, False, True),    # END DECODING
    }

    # Bits per sample lookup for mode 0x18
    simple_bits_map = {0: 8, 1: 10, 2: 16, 3: 12}

    max_frames = 10000

    while ctx.pos < ctx.end_pos and ctx.frame_count < max_frames:
        reader.align_byte()

        if ctx.pos >= ctx.end_pos:
            break

        # Get control byte
        if stream_control:
            control = reader.read_byte()
        else:
            control = ctx.fixed_control

        mode = control & 0x18
        value = control & 0x07

        if debug and ctx.frame_count < 10:
            print(f"  Frame {ctx.frame_count}: pos={ctx.pos}, control=0x{control:02X}, mode=0x{mode:02X}, value={value}")

        if mode == 0x00:
            # Mode 0x00: Continue/end check
            if not stream_control:
                # Fixed control - decode as DPCM with prediction
                bps, pred, is_end = dpcm_mode_table.get(value, (6, True, False))
                if is_end:
                    break
                ctx.bits_per_sample = bps
                samples = decode_frame_dpcm(ctx, reader, pred)
                all_samples.extend(samples)
            # For stream control mode 0x00, just continue to check for more data
            continue

        elif mode == 0x08:
            # Mode 0x08: Set parameters + ADPCM frame
            ctx.bits_per_sample = value if value > 0 else 2

            if ctx.pos + 2 > ctx.end_pos:
                break

            param1 = reader.read_byte()
            param2 = reader.read_byte()

            # param1 >> 2 might encode step index or other config
            # For now, use it to potentially set initial step_index
            ctx.step_index = (param1 >> 2) & 0x3F

            if debug:
                print(f"    Mode 0x08: bps={ctx.bits_per_sample}, p1=0x{param1:02X}, p2=0x{param2:02X}")

            # Decode ADPCM frame
            samples = decode_frame_adpcm(ctx, reader)
            all_samples.extend(samples)

        elif mode == 0x10:
            # Mode 0x10: DPCM frame (switch table at 0x72b)
            bps, pred, is_end = dpcm_mode_table.get(value, (6, True, False))
            if is_end:
                # Values 2, 6, 7 terminate decoding
                break
            ctx.bits_per_sample = bps
            samples = decode_frame_dpcm(ctx, reader, pred)
            all_samples.extend(samples)

        elif mode == 0x18:
            # Mode 0x18: Simple bit expansion
            if value < 4:
                ctx.bits_per_sample = simple_bits_map.get(value, 8)
            else:
                break
            samples = decode_frame_simple(ctx, reader)
            all_samples.extend(samples)

        else:
            break

    if debug:
        print(f"  Decoded {ctx.frame_count} frames, {len(all_samples)} samples")

    return all_samples, 16000


def save_wav(samples: List[int], output_path: Path, sample_rate: int = 16000):
    """Save samples as WAV file."""
    if not samples:
        print("  No samples to save")
        return False

    samples = [clamp16(s) for s in samples]

    with wave.open(str(output_path), 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f'<{len(samples)}h', *samples))

    duration = len(samples) / sample_rate
    nonzero = sum(1 for s in samples if s != 0)
    print(f"  Saved {len(samples)} samples ({duration:.2f}s), {nonzero} non-zero ({100*nonzero/len(samples):.1f}%)")
    return True


def compare_wav_files(file1: Path, file2: Path) -> Tuple[bool, str]:
    """Compare two WAV files and return similarity info."""
    try:
        with wave.open(str(file1), 'r') as w1, wave.open(str(file2), 'r') as w2:
            frames1 = w1.readframes(w1.getnframes())
            frames2 = w2.readframes(w2.getnframes())

            samples1 = struct.unpack(f'<{len(frames1)//2}h', frames1)
            samples2 = struct.unpack(f'<{len(frames2)//2}h', frames2)

            if len(samples1) != len(samples2):
                return False, f"Length mismatch: {len(samples1)} vs {len(samples2)}"

            total_diff = 0
            max_diff = 0
            exact_matches = 0

            for s1, s2 in zip(samples1, samples2):
                diff = abs(s1 - s2)
                total_diff += diff
                max_diff = max(max_diff, diff)
                if s1 == s2:
                    exact_matches += 1

            avg_diff = total_diff / len(samples1) if samples1 else 0
            match_pct = 100 * exact_matches / len(samples1) if samples1 else 0

            if exact_matches == len(samples1):
                return True, "EXACT MATCH"
            elif avg_diff < 10:
                return True, f"Close match: avg_diff={avg_diff:.1f}, max={max_diff}, {match_pct:.1f}% exact"
            else:
                return False, f"Mismatch: avg_diff={avg_diff:.1f}, max={max_diff}, {match_pct:.1f}% exact"

    except Exception as e:
        return False, f"Error: {e}"


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <firmware.bin>")
        return 1

    firmware_path = Path(sys.argv[1])
    base_dir = Path(__file__).parent
    output_dir = base_dir / "decoded_audio_standalone"
    ref_dir = base_dir / "decoded_audio_emu"

    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("ISD9160 Standalone DPCM Decoder")
    print("=" * 70)
    print()
    print("Pure Python DPCM/ADPCM decoder based on Ghidra reverse engineering.")
    print("VPE/Siren7 segments are skipped (require emulator).")
    print()

    if not firmware_path.exists():
        print(f"Error: Firmware not found: {firmware_path}")
        return 2

    print("Parsing firmware...")
    parser = FirmwareParser(firmware_path)
    segments = parser.parse_segments()
    print()

    decoded = 0
    skipped = 0
    results = []

    for segment in segments:
        print(f"\nSegment {segment.index}: {segment.size} bytes, codec {segment.codec}")

        if segment.size < 4:
            print("  Segment too small")
            skipped += 1
            continue

        first_byte = segment.data[0]
        print(f"  First byte: 0x{first_byte:02X} (codec={segment.codec}, lower5=0x{segment.codec_lower5:02X})")

        # Skip VPE segments
        if segment.codec_lower5 in (0x1D, 0x1E):
            print("  Format: VPE/Siren7 - skipping (requires emulator)")
            skipped += 1
            continue

        print("  Format: DPCM")
        samples, sample_rate = decode_dpcm_segment(segment.data, debug=(segment.index == 0))

        if samples:
            out_name = f"seg_{segment.index:02d}_standalone.wav"
            out_path = output_dir / out_name
            if save_wav(samples, out_path, sample_rate):
                decoded += 1

                ref_path = ref_dir / f"segment_{segment.index:02d}.wav"
                if ref_path.exists():
                    match, info = compare_wav_files(out_path, ref_path)
                    status = "OK" if match else "FAIL"
                    print(f"  Compare: [{status}] {info}")
                    results.append((segment.index, match, info))
        else:
            print("  No samples decoded")
            skipped += 1

    print()
    print("=" * 70)
    print(f"Results: {decoded} decoded, {skipped} skipped")
    print(f"Output: {output_dir}")

    if results:
        print()
        print("Comparison with emulator reference:")
        for idx, match, info in results:
            status = "OK" if match else "FAIL"
            print(f"  Segment {idx}: [{status}] {info}")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
