import sys
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Memory layout matching the firmware
APROM_BASE = 0x00000000
APROM_SIZE = 0x8000  # 32KB

VPE_BASE = 0x00008000
VPE_SIZE = 0x1C400  # ~113KB

# VPE structure constants
VPE_MAGIC = bytes([0xFF, 0xAA, 0x55, 0x11])
VPE_HEADER_OFFSET = 0x8000
AUDIO_INDEX_OFFSET_FIELD = 0x8008
SEGMENT_TABLE_PTR_FIELD = 0x0C
SEGMENT_COUNT_FIELD = 0x10

@dataclass
class VPEHeader:
    """VPE header structure."""
    magic: bytes
    audio_index_offset: int
    total_size: int
    func_init: int
    func_setup: int
    func_decode: int


@dataclass
class AudioIndex:
    """Audio index table structure."""
    signature: bytes
    segment_table_ptr: int
    segment_count: int


@dataclass
class Segment:
    """Audio segment information."""
    index: int
    start_addr: int
    end_addr: int
    size: int
    codec: int
    codec_lower5: int
    data: bytes

class FirmwareParser:
    """Parses firmware binary to extract segment information dynamically."""

    def __init__(self, firmware_path: Path):
        with open(firmware_path, 'rb') as f:
            self.firmware = f.read()

        self.aprom = self.firmware[0:APROM_SIZE]
        self.vpe = self.firmware[APROM_SIZE:APROM_SIZE + VPE_SIZE]

    def parse_vpe_header(self) -> VPEHeader:
        """Parse the VPE header at 0x8000."""
        vpe_data = self.vpe

        magic = vpe_data[0:4]
        if magic != VPE_MAGIC:
            raise ValueError(f"Invalid VPE magic: {magic.hex()} (expected {VPE_MAGIC.hex()})")

        return VPEHeader(
            magic=magic,
            audio_index_offset=struct.unpack_from('<I', vpe_data, 0x08)[0],
            total_size=struct.unpack_from('<I', vpe_data, 0x0C)[0],
            func_init=struct.unpack_from('<I', vpe_data, 0x10)[0],
            func_setup=struct.unpack_from('<I', vpe_data, 0x14)[0],
            func_decode=struct.unpack_from('<I', vpe_data, 0x18)[0],
        )

    def parse_audio_index(self, offset: int) -> AudioIndex:
        """Parse the audio index table."""
        # offset is firmware address, convert to VPE-relative
        vpe_offset = offset - VPE_BASE
        idx_data = self.vpe[vpe_offset:]

        signature = idx_data[0:4]
        segment_table_ptr = struct.unpack_from('<I', idx_data, SEGMENT_TABLE_PTR_FIELD)[0]
        segment_count = struct.unpack_from('<I', idx_data, SEGMENT_COUNT_FIELD)[0]

        return AudioIndex(
            signature=signature,
            segment_table_ptr=segment_table_ptr,
            segment_count=segment_count,
        )

    def parse_segments(self) -> List[Segment]:
        """Parse all segments from the firmware."""
        vpe_header = self.parse_vpe_header()
        audio_index = self.parse_audio_index(vpe_header.audio_index_offset)

        print("VPE Header:")
        print(f"  Magic: {vpe_header.magic.hex()}")
        print(f"  Audio index offset: 0x{vpe_header.audio_index_offset:X}")
        print(f"  Total size: 0x{vpe_header.total_size:X}")
        print(f"  Decoder init: 0x{vpe_header.func_init:X}")
        print(f"  Decoder frame: 0x{vpe_header.func_decode:X}")
        print()
        print("Audio Index:")
        print(f"  Signature: {audio_index.signature}")
        print(f"  Segment table ptr: 0x{audio_index.segment_table_ptr:X}")
        print(f"  Segment count: {audio_index.segment_count}")
        print()

        # Read segment index table
        # The table is at segment_table_ptr (firmware address)
        # Each entry is 8 bytes: (start_addr, end_addr) as 32-bit LE values
        table_offset = audio_index.segment_table_ptr - VPE_BASE
        segments = []

        for i in range(audio_index.segment_count):
            entry_offset = table_offset + i * 8
            start_addr = struct.unpack_from('<I', self.vpe, entry_offset)[0]
            end_addr = struct.unpack_from('<I', self.vpe, entry_offset + 4)[0]

            # Extract segment data
            # Addresses are firmware addresses, convert to VPE-relative
            start_vpe = start_addr - VPE_BASE
            end_vpe = end_addr - VPE_BASE
            size = end_addr - start_addr

            data = self.vpe[start_vpe:end_vpe]

            # Parse codec info from first byte
            first_byte = data[0] if data else 0
            codec = first_byte >> 5
            codec_lower5 = first_byte & 0x1F

            segments.append(Segment(
                index=i,
                start_addr=start_addr,
                end_addr=end_addr,
                size=size,
                codec=codec,
                codec_lower5=codec_lower5,
                data=data,
            ))

            print(f"Segment {i}: addr=0x{start_addr:X}-0x{end_addr:X}, size={size}, "
                  f"codec={codec}, first=0x{first_byte:02X}")

        return segments

def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [firmware.bin]")
        return 1

    firmware_path = Path(sys.argv[1])

    base_dir = Path(__file__).parent
    output_dir = base_dir / "segments_raw"

    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("ISD9160 segment extractor")
    print("=" * 70)
    print()
    print("Extracts segments using index table")
    print()

    if not firmware_path.exists():
        print(f"Error: Firmware not found: {firmware_path}")
        return 1

    # Parse firmware
    print("Parsing firmware...")
    parser = FirmwareParser(firmware_path)
    segments = parser.parse_segments()

    # Decode each segment
    print("=" * 70)
    print("Decoding segments...")
    print("=" * 70)

    success_count = 0
    failed_count = 0

    for segment in segments:
        print(f"\nSegment {segment.index}: {segment.size} bytes, codec {segment.codec}")

        fmt = "unk"
        # Determine decoder based on codec type
        if segment.codec_lower5 in (0x1D, 0x1E):
            print("  Format: VPE/Siren7")
            fmt = "siren7"
        else:
            print("  Format: DPCM")
            fmt = "dpcm"

        if segment.data:
            out_path = output_dir / f"segment_{segment.index:02d}_{fmt}.raw"
            with open(out_path, "wb") as fout:
                fout.write(segment.data)
            success_count += 1
        else:
            print("    No samples decoded")
            failed_count += 1

    print()
    print("=" * 70)
    print(f"Results: {success_count} extracted, {failed_count} failed")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
