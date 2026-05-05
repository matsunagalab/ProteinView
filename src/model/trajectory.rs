/// A loaded MD trajectory.  Coordinates are in the same Cartesian frame
/// as the topology (DCD does not center).  `unit_cells` holds an optional
/// (a, b, c, alpha, beta, gamma) tuple per frame in Angstroms / degrees.
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub frames: Vec<Vec<[f32; 3]>>,
    /// Reserved for future PBC box rendering; populated when the DCD has
    /// the CHARMM extra block but currently unused by the renderer.
    #[allow(dead_code)]
    pub unit_cells: Vec<Option<[f32; 6]>>,
    /// Reserved for future ps/fs time-axis display; the DCD DELTA field is
    /// in CHARMM AKMA units (1 AKMA ≈ 0.0488882 ps).
    #[allow(dead_code)]
    pub timestep_akma: f64,
    pub is_charmm: bool,
}
