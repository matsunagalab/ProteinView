//! DCD trajectory file reader.
//!
//! Ports the parsing logic from VMD's `molfile_plugin/src/dcdplugin.c`
//! (cross-checked against the verbatim PLUMED2 mirror at
//! `plumed/plumed2/src/molfile/dcdplugin.cpp`).
//!
//! Handles:
//! - Endianness auto-detection (LE / BE)
//! - 32-bit and 64-bit Fortran record markers
//! - CHARMM and X-PLOR header variants
//! - Optional unit-cell (CHARMM extra block) per frame
//! - Fixed-atom reconstruction (NAMNF > 0)
//! - Optional 4D (W) coordinate record (consumed but discarded)
//! - Frame-count recomputation from file size when header NSET is unreliable
//!
//! Velocity DCDs (magic `VELD`) are explicitly rejected.

use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use crate::model::trajectory::Trajectory;

const CORD_MAGIC: &[u8; 4] = b"CORD";
const VELD_MAGIC: &[u8; 4] = b"VELD";

/// Cap on the per-frame atom count.  Defends against allocating gigabytes
/// when the parser hits a malformed file masquerading as DCD.
const MAX_ATOMS: usize = 100_000_000;

/// Cap on title-record entries (each is 80 bytes).
const MAX_TITLES: i32 = 1024;

struct DcdHeader {
    natoms: usize,
    nset_estimate: usize,
    is_charmm: bool,
    has_extra_block: bool,
    has_4dims: bool,
    /// 1 = 32-bit Fortran length markers, 2 = 64-bit (each marker is two u32).
    rec_scale: u8,
    reverse_endian: bool,
    delta_akma: f64,
    nfixed: usize,
    /// 0-based atom indices of free (non-fixed) atoms; only populated when nfixed > 0.
    free_indices: Vec<u32>,
}

/// Load a DCD trajectory.  `expected_natoms` must match the topology;
/// a mismatch is reported as an error so the caller can guide the user
/// to a topology that includes the same atom set the DCD was written with.
pub fn load_dcd(path: &str, expected_natoms: usize) -> Result<Trajectory> {
    let file = File::open(path).with_context(|| format!("cannot open DCD file: {path}"))?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let header = read_header(&mut reader, expected_natoms, file_size)?;

    let mut frames: Vec<Vec<[f32; 3]>> = Vec::with_capacity(header.nset_estimate);
    let mut unit_cells: Vec<Option<[f32; 6]>> = Vec::with_capacity(header.nset_estimate);
    let mut fixed_template: Option<Vec<[f32; 3]>> = None;

    for frame_idx in 0..header.nset_estimate {
        let cell = if header.has_extra_block {
            match read_unit_cell(&mut reader, header.reverse_endian, header.rec_scale) {
                Ok(c) => Some(c),
                Err(_) if frame_idx > 0 => break, // tolerate truncated tail
                Err(e) => return Err(e),
            }
        } else {
            None
        };

        let n_in_frame = if header.nfixed > 0 && frame_idx > 0 {
            header.natoms - header.nfixed
        } else {
            header.natoms
        };

        let xs = match read_coord_record(&mut reader, n_in_frame, &header) {
            Ok(v) => v,
            Err(_) if frame_idx > 0 => break,
            Err(e) => return Err(e),
        };
        let ys = read_coord_record(&mut reader, n_in_frame, &header)?;
        let zs = read_coord_record(&mut reader, n_in_frame, &header)?;

        if header.has_4dims {
            // Skip W axis record, same length as a coord record.
            skip_record(&mut reader, header.rec_scale, header.reverse_endian)?;
        }

        let coords: Vec<[f32; 3]> = if header.nfixed > 0 && frame_idx > 0 {
            // Reconstruct: copy fixed_template (frame 0), scatter free atom coords.
            let template = fixed_template
                .as_ref()
                .expect("frame 0 must have populated fixed_template");
            let mut full = template.clone();
            for (k, &idx) in header.free_indices.iter().enumerate() {
                full[idx as usize] = [xs[k], ys[k], zs[k]];
            }
            full
        } else {
            (0..header.natoms)
                .map(|i| [xs[i], ys[i], zs[i]])
                .collect()
        };

        if header.nfixed > 0 && frame_idx == 0 {
            fixed_template = Some(coords.clone());
        }

        frames.push(coords);
        unit_cells.push(cell);
    }

    if frames.is_empty() {
        bail!("DCD file contains no readable frames");
    }

    Ok(Trajectory {
        frames,
        unit_cells,
        timestep_akma: header.delta_akma,
        is_charmm: header.is_charmm,
    })
}

fn read_header<R: Read + Seek>(
    reader: &mut R,
    expected_natoms: usize,
    file_size: u64,
) -> Result<DcdHeader> {
    // Step 1: peek first 8 bytes to detect endianness + record-marker width.
    //
    // Layout candidates:
    //   - 32-bit Fortran marker: [u32 len = 84][magic "CORD" or "VELD" (4 ASCII)]
    //   - 64-bit Fortran marker: [u32 lo = 84][u32 hi = 0][...payload begins...]
    //
    // The 4 ASCII magic bytes never need endian swapping; we compare them
    // directly. The length marker is what tells us LE vs BE.
    let mut peek = [0u8; 8];
    reader.read_exact(&mut peek)?;
    let next4 = &peek[4..8];
    let magic_is_cord = next4 == CORD_MAGIC;
    let magic_is_veld = next4 == VELD_MAGIC;
    let known_magic = magic_is_cord || magic_is_veld;
    let lo_le = u32::from_le_bytes([peek[0], peek[1], peek[2], peek[3]]);
    let lo_be = u32::from_be_bytes([peek[0], peek[1], peek[2], peek[3]]);
    let len_le_64 = u64::from_le_bytes(peek);
    let len_be_64 = u64::from_be_bytes(peek);

    let (reverse_endian, rec_scale) = if known_magic && lo_le == 84 {
        (false, 1u8)
    } else if known_magic && lo_be == 84 {
        (true, 1u8)
    } else if len_le_64 == 84 {
        (false, 2u8)
    } else if len_be_64 == 84 {
        (true, 2u8)
    } else {
        bail!("not a DCD file (unexpected first record marker)");
    };

    reader.seek(SeekFrom::Start(0))?;

    // CORD record: 4 bytes magic + 80 bytes (20 i32 ICNTRL) = 84 bytes payload.
    let header_payload = read_record(reader, rec_scale, reverse_endian)?;
    if header_payload.len() != 84 {
        bail!(
            "DCD header record has unexpected size {} (expected 84)",
            header_payload.len()
        );
    }
    let magic = &header_payload[0..4];
    if magic == VELD_MAGIC {
        bail!("velocity DCDs (VELD magic) are not supported");
    }
    if magic != CORD_MAGIC {
        bail!("DCD header does not start with CORD magic");
    }

    let mut icntrl = [0i32; 20];
    for (i, slot) in icntrl.iter_mut().enumerate() {
        let off = 4 + i * 4;
        *slot = read_i32(&header_payload[off..off + 4], reverse_endian);
    }

    let charmm_ver = icntrl[19];
    let is_charmm = charmm_ver != 0;
    let has_extra_block = is_charmm && (icntrl[11] != 0);
    let has_4dims = is_charmm && (icntrl[12] != 0);
    let nfixed: usize = if is_charmm {
        icntrl[9].max(0) as usize
    } else {
        0
    };

    // DELTA: f32 in CHARMM (slot 10 = bytes 44..48), f64 in X-PLOR (spans
    // slots 8 & 9 = bytes 36..44).  Read raw bytes from the un-permuted payload
    // and apply a single endian swap appropriate to the wider type.
    let delta_akma: f64 = if is_charmm {
        read_f32(&header_payload[44..48], reverse_endian) as f64
    } else {
        read_f64(&header_payload[36..44], reverse_endian)
    };

    // ----- TITLE record -----
    let title_payload = read_record(reader, rec_scale, reverse_endian)?;
    if title_payload.len() < 4 {
        bail!("DCD title record too short");
    }
    let ntitle = read_i32(&title_payload[0..4], reverse_endian);
    if !(0..=MAX_TITLES).contains(&ntitle) {
        bail!("DCD title count out of range: {ntitle}");
    }
    let expected_title_size = 4 + (ntitle as usize) * 80;
    if title_payload.len() < expected_title_size {
        bail!(
            "DCD title record truncated: {} < {expected_title_size}",
            title_payload.len()
        );
    }

    // ----- NATOMS record -----
    let natoms_payload = read_record(reader, rec_scale, reverse_endian)?;
    if natoms_payload.len() != 4 {
        bail!(
            "DCD NATOMS record has unexpected size {} (expected 4)",
            natoms_payload.len()
        );
    }
    let natoms_raw = read_i32(&natoms_payload, reverse_endian);
    if natoms_raw <= 0 {
        bail!("DCD NATOMS is non-positive: {natoms_raw}");
    }
    let natoms = natoms_raw as usize;
    if natoms > MAX_ATOMS {
        bail!("DCD NATOMS {natoms} exceeds sanity cap {MAX_ATOMS}");
    }
    if natoms != expected_natoms {
        bail!(
            "DCD atom count mismatch: topology has {expected_natoms} atoms but DCD has {natoms}. \
             Use a topology PDB whose atom set matches the trajectory \
             (e.g. solvated PDB if the DCD includes water)."
        );
    }

    // ----- FREEINDEX record (only when fixed atoms are present) -----
    let mut free_indices = Vec::new();
    if nfixed > 0 {
        if nfixed > natoms {
            bail!("DCD has more fixed atoms ({nfixed}) than total atoms ({natoms})");
        }
        let nfree = natoms - nfixed;
        let payload = read_record(reader, rec_scale, reverse_endian)?;
        if payload.len() != 4 * nfree {
            bail!(
                "DCD freeindex record size mismatch: got {} bytes for {nfree} free atoms",
                payload.len()
            );
        }
        free_indices.reserve(nfree);
        for i in 0..nfree {
            let v = read_i32(&payload[i * 4..i * 4 + 4], reverse_endian);
            if v < 1 || (v as usize) > natoms {
                bail!("DCD freeindex out of range: {v} (natoms={natoms})");
            }
            free_indices.push((v - 1) as u32);
        }
    }

    let header_end = reader.stream_position()?;
    let nset_header = icntrl[0].max(0) as usize;
    let nset_estimate = estimate_nset(
        file_size,
        header_end,
        natoms,
        nfixed,
        rec_scale,
        has_extra_block,
        has_4dims,
    );
    // Prefer the file-size estimate (header NSET is often wrong/zero).
    // Only fall back to the header value if estimation says zero AND
    // header claims something — useful for streaming/non-seekable wrappers,
    // not the case here, but harmless.
    let nset = if nset_estimate > 0 {
        nset_estimate
    } else {
        nset_header
    };

    Ok(DcdHeader {
        natoms,
        nset_estimate: nset,
        is_charmm,
        has_extra_block,
        has_4dims,
        rec_scale,
        reverse_endian,
        delta_akma,
        nfixed,
        free_indices,
    })
}

fn read_unit_cell<R: Read>(
    reader: &mut R,
    reverse_endian: bool,
    rec_scale: u8,
) -> Result<[f32; 6]> {
    let payload = read_record(reader, rec_scale, reverse_endian)?;
    if payload.len() != 48 {
        bail!(
            "DCD unit-cell record has unexpected size {} (expected 48)",
            payload.len()
        );
    }
    // CHARMM order: tmp[0]=a, tmp[1]=γ, tmp[2]=b, tmp[3]=β, tmp[4]=α, tmp[5]=c
    let mut tmp = [0f64; 6];
    for (i, slot) in tmp.iter_mut().enumerate() {
        let off = i * 8;
        *slot = read_f64(&payload[off..off + 8], reverse_endian);
    }

    // CHARMM/NAMD>2.5 store cosines of the angles in tmp[1], tmp[3], tmp[4]
    // (range [-1, 1]); NAMD≤2.5 stores raw degrees.
    let in_cos_range = |x: f64| (-1.0..=1.0).contains(&x);
    let (alpha, beta, gamma) = if in_cos_range(tmp[1]) && in_cos_range(tmp[3]) && in_cos_range(tmp[4]) {
        // 90 - asin(x) * 90 / (π/2) = acos(x) * 180/π, but exact 90 for orthogonal cells
        let to_deg = |x: f64| 90.0 - x.asin() * 90.0 / std::f64::consts::FRAC_PI_2;
        (to_deg(tmp[4]), to_deg(tmp[3]), to_deg(tmp[1]))
    } else {
        (tmp[4], tmp[3], tmp[1])
    };

    Ok([
        tmp[0] as f32,
        tmp[2] as f32,
        tmp[5] as f32,
        alpha as f32,
        beta as f32,
        gamma as f32,
    ])
}

fn read_coord_record<R: Read>(
    reader: &mut R,
    n: usize,
    header: &DcdHeader,
) -> Result<Vec<f32>> {
    let payload = read_record(reader, header.rec_scale, header.reverse_endian)?;
    let expected = 4 * n;
    if payload.len() != expected {
        bail!(
            "DCD coord record size mismatch: got {} bytes, expected {expected}",
            payload.len()
        );
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(read_f32(&payload[i * 4..i * 4 + 4], header.reverse_endian));
    }
    Ok(out)
}

fn skip_record<R: Read>(reader: &mut R, rec_scale: u8, reverse_endian: bool) -> Result<()> {
    let len = read_record_marker(reader, rec_scale, reverse_endian)?;
    let mut sink = vec![0u8; len];
    reader.read_exact(&mut sink)?;
    let trailer = read_record_marker(reader, rec_scale, reverse_endian)?;
    if trailer != len {
        bail!("DCD record marker mismatch (skip): leading {len} != trailing {trailer}");
    }
    Ok(())
}

fn read_record<R: Read>(
    reader: &mut R,
    rec_scale: u8,
    reverse_endian: bool,
) -> Result<Vec<u8>> {
    let len = read_record_marker(reader, rec_scale, reverse_endian)?;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    let trailer = read_record_marker(reader, rec_scale, reverse_endian)?;
    if trailer != len {
        bail!("DCD record marker mismatch: leading {len} != trailing {trailer}");
    }
    Ok(buf)
}

fn read_record_marker<R: Read>(
    reader: &mut R,
    rec_scale: u8,
    reverse_endian: bool,
) -> Result<usize> {
    let mut b4 = [0u8; 4];
    reader.read_exact(&mut b4)?;
    let lo = u32::from_le_bytes(b4);
    let lo = if reverse_endian { lo.swap_bytes() } else { lo };
    if rec_scale == 2 {
        let mut b4_hi = [0u8; 4];
        reader.read_exact(&mut b4_hi)?;
        let hi = u32::from_le_bytes(b4_hi);
        let hi = if reverse_endian { hi.swap_bytes() } else { hi };
        if hi != 0 {
            bail!("DCD 64-bit record length too large: high word {hi} != 0");
        }
    }
    Ok(lo as usize)
}

fn read_i32(buf: &[u8], reverse: bool) -> i32 {
    let v = i32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if reverse { v.swap_bytes() } else { v }
}

fn read_f32(buf: &[u8], reverse: bool) -> f32 {
    let bits = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let bits = if reverse { bits.swap_bytes() } else { bits };
    f32::from_bits(bits)
}

fn read_f64(buf: &[u8], reverse: bool) -> f64 {
    let bits = u64::from_le_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ]);
    let bits = if reverse { bits.swap_bytes() } else { bits };
    f64::from_bits(bits)
}

fn estimate_nset(
    file_size: u64,
    header_end: u64,
    natoms: usize,
    nfixed: usize,
    rec_scale: u8,
    has_extra_block: bool,
    has_4dims: bool,
) -> usize {
    let marker_overhead = 2u64 * 4 * rec_scale as u64; // 8 (32-bit) or 16 (64-bit) bytes per record
    let extra_block_size = if has_extra_block {
        marker_overhead + 48
    } else {
        0
    };
    let ndims = if has_4dims { 4u64 } else { 3u64 };
    let nfree = natoms.saturating_sub(nfixed) as u64;

    let first_frame_size = ndims * (marker_overhead + 4 * natoms as u64) + extra_block_size;
    let later_frame_size = if nfixed > 0 {
        ndims * (marker_overhead + 4 * nfree) + extra_block_size
    } else {
        first_frame_size
    };

    let trj_bytes = file_size.saturating_sub(header_end);
    if trj_bytes < first_frame_size {
        return 0;
    }
    if later_frame_size == 0 {
        return 1;
    }
    let after_first = trj_bytes - first_frame_size;
    1 + (after_first / later_frame_size) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Builder for synthetic DCD files in tests.
    struct DcdBuilder {
        bytes: Vec<u8>,
        be: bool,
        rec_scale: u8,
    }

    impl DcdBuilder {
        fn new(be: bool, rec_scale: u8) -> Self {
            Self {
                bytes: Vec::new(),
                be,
                rec_scale,
            }
        }

        fn put_marker(&mut self, len: u32) {
            let buf = if self.be {
                len.swap_bytes().to_le_bytes()
            } else {
                len.to_le_bytes()
            };
            self.bytes.extend_from_slice(&buf);
            if self.rec_scale == 2 {
                self.bytes.extend_from_slice(&[0u8; 4]); // high word = 0
            }
        }

        fn put_i32(&mut self, v: i32) {
            let buf = if self.be {
                v.swap_bytes().to_le_bytes()
            } else {
                v.to_le_bytes()
            };
            self.bytes.extend_from_slice(&buf);
        }

        fn put_f32(&mut self, v: f32) {
            let bits = v.to_bits();
            let buf = if self.be {
                bits.swap_bytes().to_le_bytes()
            } else {
                bits.to_le_bytes()
            };
            self.bytes.extend_from_slice(&buf);
        }

        fn put_f64(&mut self, v: f64) {
            let bits = v.to_bits();
            let buf = if self.be {
                bits.swap_bytes().to_le_bytes()
            } else {
                bits.to_le_bytes()
            };
            self.bytes.extend_from_slice(&buf);
        }

        fn put_record<F: FnOnce(&mut Self)>(&mut self, len: u32, body: F) {
            self.put_marker(len);
            body(self);
            self.put_marker(len);
        }

        fn write_temp(&self) -> NamedTempFile {
            let mut tmp = NamedTempFile::new().unwrap();
            tmp.write_all(&self.bytes).unwrap();
            tmp.flush().unwrap();
            tmp
        }
    }

    fn write_charmm_header(b: &mut DcdBuilder, magic: &[u8; 4], icntrl: &[i32; 20]) {
        b.put_record(84, |b| {
            b.bytes.extend_from_slice(magic);
            for &v in icntrl {
                b.put_i32(v);
            }
        });
    }

    fn write_title(b: &mut DcdBuilder, ntitle: i32) {
        let len = 4 + (ntitle as u32) * 80;
        b.put_record(len, |b| {
            b.put_i32(ntitle);
            for _ in 0..ntitle {
                b.bytes.extend_from_slice(&[b' '; 80]);
            }
        });
    }

    fn write_natoms(b: &mut DcdBuilder, natoms: i32) {
        b.put_record(4, |b| b.put_i32(natoms));
    }

    fn write_coord_record(b: &mut DcdBuilder, vals: &[f32]) {
        let len = 4 * vals.len() as u32;
        b.put_record(len, |b| {
            for &v in vals {
                b.put_f32(v);
            }
        });
    }

    fn write_unit_cell(b: &mut DcdBuilder, abc: [f64; 3], angles_deg: [f64; 3]) {
        b.put_record(48, |b| {
            // tmp[0]=a, tmp[1]=γ, tmp[2]=b, tmp[3]=β, tmp[4]=α, tmp[5]=c
            // We'll write angles as cosines (CHARMM/NAMD>2.5 style).
            let to_cos = |deg: f64| (deg.to_radians()).cos();
            b.put_f64(abc[0]);
            b.put_f64(to_cos(angles_deg[2]));
            b.put_f64(abc[1]);
            b.put_f64(to_cos(angles_deg[1]));
            b.put_f64(to_cos(angles_deg[0]));
            b.put_f64(abc[2]);
        });
    }

    /// Standard CHARMM ICNTRL with most fields zero.
    fn charmm_icntrl(nset: i32, nfixed: i32, has_extra: bool) -> [i32; 20] {
        let mut ic = [0i32; 20];
        ic[0] = nset; // NSET
        ic[1] = 1; // ISTART
        ic[2] = 1; // NSAVC
        ic[9] = nfixed; // NAMNF
        ic[10] = 1.0f32.to_bits() as i32; // DELTA = 1.0 as f32
        ic[11] = if has_extra { 1 } else { 0 }; // HAS_BOX
        ic[12] = 0; // HAS_4D
        ic[19] = 1; // CHARMM_VER (non-zero -> CHARMM)
        ic
    }

    fn build_simple_dcd(be: bool, rec_scale: u8, nset: i32, natoms: i32) -> NamedTempFile {
        let mut b = DcdBuilder::new(be, rec_scale);
        write_charmm_header(&mut b, CORD_MAGIC, &charmm_icntrl(nset, 0, false));
        write_title(&mut b, 1);
        write_natoms(&mut b, natoms);
        // Two frames of N atoms each.
        for f in 0..nset {
            let xs: Vec<f32> = (0..natoms).map(|i| (f * 100 + i) as f32).collect();
            let ys: Vec<f32> = (0..natoms).map(|i| (f * 100 + i + 1) as f32).collect();
            let zs: Vec<f32> = (0..natoms).map(|i| (f * 100 + i + 2) as f32).collect();
            write_coord_record(&mut b, &xs);
            write_coord_record(&mut b, &ys);
            write_coord_record(&mut b, &zs);
        }
        b.write_temp()
    }

    #[test]
    fn parses_simple_charmm_le_32bit() {
        let tmp = build_simple_dcd(false, 1, 2, 3);
        let traj = load_dcd(tmp.path().to_str().unwrap(), 3).unwrap();
        assert_eq!(traj.frames.len(), 2);
        assert_eq!(traj.frames[0].len(), 3);
        assert_eq!(traj.frames[0][0], [0.0, 1.0, 2.0]);
        assert_eq!(traj.frames[1][2], [102.0, 103.0, 104.0]);
        assert_eq!(traj.unit_cells.len(), 2);
        assert!(traj.unit_cells[0].is_none());
        assert!(traj.is_charmm);
    }

    #[test]
    fn parses_simple_charmm_be_32bit() {
        let tmp = build_simple_dcd(true, 1, 2, 3);
        let traj = load_dcd(tmp.path().to_str().unwrap(), 3).unwrap();
        assert_eq!(traj.frames.len(), 2);
        assert_eq!(traj.frames[1][1], [101.0, 102.0, 103.0]);
    }

    #[test]
    fn parses_simple_charmm_le_64bit() {
        let tmp = build_simple_dcd(false, 2, 2, 3);
        let traj = load_dcd(tmp.path().to_str().unwrap(), 3).unwrap();
        assert_eq!(traj.frames.len(), 2);
        assert_eq!(traj.frames[0][0], [0.0, 1.0, 2.0]);
    }

    #[test]
    fn recomputes_nset_when_header_says_zero() {
        // Build a file that actually contains 3 frames but lies about it
        // in the header (NSET=0).  The estimator should recover the true count.
        let mut b = DcdBuilder::new(false, 1);
        let mut ic = charmm_icntrl(0, 0, false);
        ic[0] = 0; // lying NSET
        write_charmm_header(&mut b, CORD_MAGIC, &ic);
        write_title(&mut b, 1);
        write_natoms(&mut b, 4);
        for f in 0..3i32 {
            let xs: Vec<f32> = (0..4).map(|i| (f * 10 + i) as f32).collect();
            write_coord_record(&mut b, &xs);
            write_coord_record(&mut b, &xs);
            write_coord_record(&mut b, &xs);
        }
        let tmp = b.write_temp();
        let traj = load_dcd(tmp.path().to_str().unwrap(), 4).unwrap();
        assert_eq!(traj.frames.len(), 3);
    }

    #[test]
    fn rejects_velocity_dcd() {
        let mut b = DcdBuilder::new(false, 1);
        write_charmm_header(&mut b, VELD_MAGIC, &charmm_icntrl(1, 0, false));
        let tmp = b.write_temp();
        let err = load_dcd(tmp.path().to_str().unwrap(), 3).unwrap_err();
        assert!(err.to_string().to_lowercase().contains("velocity"));
    }

    #[test]
    fn rejects_natoms_mismatch() {
        let tmp = build_simple_dcd(false, 1, 1, 5);
        let err = load_dcd(tmp.path().to_str().unwrap(), 3).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("3"), "msg={msg}");
        assert!(msg.contains("5"), "msg={msg}");
    }

    #[test]
    fn parses_unit_cell_with_cosines() {
        let mut b = DcdBuilder::new(false, 1);
        write_charmm_header(&mut b, CORD_MAGIC, &charmm_icntrl(1, 0, true));
        write_title(&mut b, 1);
        write_natoms(&mut b, 2);
        write_unit_cell(&mut b, [10.0, 11.0, 12.0], [90.0, 90.0, 90.0]);
        write_coord_record(&mut b, &[0.0, 1.0]);
        write_coord_record(&mut b, &[2.0, 3.0]);
        write_coord_record(&mut b, &[4.0, 5.0]);
        let tmp = b.write_temp();

        let traj = load_dcd(tmp.path().to_str().unwrap(), 2).unwrap();
        assert_eq!(traj.frames.len(), 1);
        let cell = traj.unit_cells[0].unwrap();
        assert!((cell[0] - 10.0).abs() < 1e-3);
        assert!((cell[1] - 11.0).abs() < 1e-3);
        assert!((cell[2] - 12.0).abs() < 1e-3);
        // Orthogonal cell -> all 90°
        for &deg in &cell[3..6] {
            assert!((deg - 90.0).abs() < 1e-3, "got {deg}");
        }
    }

    #[test]
    fn parses_fixed_atoms() {
        // 5 atoms total, 2 fixed (atoms 1 & 4 in 1-based -> 0 & 3 in 0-based).
        // free indices = [2, 3, 5] (1-based) = [1, 2, 4] (0-based).
        let natoms: i32 = 5;
        let nfixed: i32 = 2;
        let nfree: i32 = natoms - nfixed; // 3
        let mut b = DcdBuilder::new(false, 1);
        write_charmm_header(&mut b, CORD_MAGIC, &charmm_icntrl(2, nfixed, false));
        write_title(&mut b, 1);
        write_natoms(&mut b, natoms);
        // FREEINDEX record: 1-based indices of free atoms
        b.put_record(4 * nfree as u32, |b| {
            for &i in &[2i32, 3, 5] {
                b.put_i32(i);
            }
        });
        // Frame 0: full natoms
        let f0_x: Vec<f32> = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let f0_y: Vec<f32> = vec![20.0, 21.0, 22.0, 23.0, 24.0];
        let f0_z: Vec<f32> = vec![30.0, 31.0, 32.0, 33.0, 34.0];
        write_coord_record(&mut b, &f0_x);
        write_coord_record(&mut b, &f0_y);
        write_coord_record(&mut b, &f0_z);
        // Frame 1: only nfree atoms (indices 1, 2, 4 -> values for those positions)
        let f1_x: Vec<f32> = vec![111.0, 112.0, 114.0];
        let f1_y: Vec<f32> = vec![211.0, 212.0, 214.0];
        let f1_z: Vec<f32> = vec![311.0, 312.0, 314.0];
        write_coord_record(&mut b, &f1_x);
        write_coord_record(&mut b, &f1_y);
        write_coord_record(&mut b, &f1_z);
        let tmp = b.write_temp();

        let traj = load_dcd(tmp.path().to_str().unwrap(), natoms as usize).unwrap();
        assert_eq!(traj.frames.len(), 2);
        // Frame 0: as written
        assert_eq!(traj.frames[0][0], [10.0, 20.0, 30.0]);
        assert_eq!(traj.frames[0][3], [13.0, 23.0, 33.0]);
        // Frame 1: fixed atoms (0 and 3) keep frame-0 values; free atoms (1, 2, 4) get new values.
        assert_eq!(traj.frames[1][0], [10.0, 20.0, 30.0]); // fixed
        assert_eq!(traj.frames[1][1], [111.0, 211.0, 311.0]); // free
        assert_eq!(traj.frames[1][2], [112.0, 212.0, 312.0]); // free
        assert_eq!(traj.frames[1][3], [13.0, 23.0, 33.0]); // fixed
        assert_eq!(traj.frames[1][4], [114.0, 214.0, 314.0]); // free
    }
}
