use std::sync::mpsc;
use std::time::Instant;

use ratatui_image::picker::Picker;

use crate::model::interface::{InterfaceAnalysis, analyze_binding_pockets, analyze_interface};
use crate::model::protein::Protein;
use crate::model::trajectory::Trajectory;
use crate::render::camera::Camera;
use crate::render::color::{ColorScheme, ColorSchemeType};
use crate::render::ribbon::{RibbonTriangle, generate_ribbon_mesh};

/// Structures with more residues than this threshold trigger performance
/// optimizations (background interface analysis, backbone default, reduced LOD).
pub const LARGE_STRUCTURE_THRESHOLD: usize = 5000;

/// Visualization mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VizMode {
    Backbone,
    Cartoon,
    Wireframe,
}

impl VizMode {
    pub fn next(&self) -> Self {
        match self {
            Self::Backbone => Self::Cartoon,
            Self::Cartoon => Self::Wireframe,
            Self::Wireframe => Self::Backbone,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Backbone => "Backbone",
            Self::Cartoon => "Cartoon",
            Self::Wireframe => "Wireframe",
        }
    }
}

/// Rendering mode for the 3D viewport
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderMode {
    /// Braille dots - highest text-mode spatial resolution, monochrome per cell
    Braille,
    /// HD-quality colored braille via software rasterizer (Lambert shading,
    /// z-buffer, depth fog).  Fast everywhere including SSH.
    HalfBlock,
    /// Full pixel graphics via Sixel/Kitty/iTerm2 - best quality, high bandwidth
    FullHD,
}

impl RenderMode {
    pub fn name(&self) -> &str {
        match self {
            Self::Braille => "Braille",
            Self::HalfBlock => "HD",
            Self::FullHD => "FullHD",
        }
    }
}

/// Whether the terminal session is local or over SSH.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConnectionType {
    Local,
    Ssh,
}

impl ConnectionType {
    /// Detect whether the current session is running over SSH.
    ///
    /// This checks the `SSH_CLIENT`, `SSH_TTY`, and `SSH_CONNECTION`
    /// environment variables. Note that this can produce false positives
    /// in containers, CI environments, or VS Code Remote sessions where
    /// these variables may be inherited. Users can override the default
    /// render mode with `--fullhd` if detection is wrong.
    pub fn detect() -> Self {
        if std::env::var("SSH_CLIENT").is_ok()
            || std::env::var("SSH_TTY").is_ok()
            || std::env::var("SSH_CONNECTION").is_ok()
        {
            Self::Ssh
        } else {
            Self::Local
        }
    }
}

/// Configuration bundle for [`App::new`], replacing individual parameters
/// to avoid too_many_arguments.
pub struct AppConfig {
    pub render_mode: RenderMode,
    pub viz_mode: VizMode,
    pub user_explicit_mode: bool,
    pub color_override: Option<ColorSchemeType>,
    pub trajectory: Option<Trajectory>,
}

/// Main application state
pub struct App {
    pub protein: Protein,
    pub camera: Camera,
    pub color_scheme: ColorScheme,
    pub viz_mode: VizMode,
    pub current_chain: usize,
    pub render_mode: RenderMode,
    pub show_help: bool,
    pub show_ligands: bool,
    pub show_interface: bool,
    pub show_interactions: bool,
    pub interface_analysis: InterfaceAnalysis,
    pub should_quit: bool,
    /// Whether the B-factor column likely contains pLDDT confidence scores.
    pub has_plddt: bool,
    /// Cached ribbon mesh — regenerated only when color scheme changes.
    pub mesh_cache: Vec<RibbonTriangle>,
    mesh_dirty: bool,
    /// ratatui-image protocol picker for Sixel/Kitty/iTerm2 graphics.
    pub picker: Picker,
    /// Detected connection type (local vs SSH).
    pub connection_type: ConnectionType,
    /// Temporary warning when user enters FullHD over SSH.
    pub ssh_hd_warning: bool,
    /// Countdown frames to auto-dismiss the SSH HD warning (~90 frames = 3 seconds at 30fps).
    pub ssh_hd_warning_frames: u8,
    /// Set to `true` after a render-mode switch so the main loop can call
    /// `terminal.clear()` before the next draw, forcing ratatui to redraw
    /// every cell and preventing stale content from the previous mode.
    pub needs_clear: bool,
    /// Saved color scheme type to restore when leaving interface mode.
    /// When interface mode is active, we display Interface colors but
    /// preserve the user's chosen scheme so it can be restored on exit.
    saved_color_scheme_type: ColorSchemeType,
    /// Whether interface analysis has been computed. For large structures
    /// (> LARGE_STRUCTURE_THRESHOLD residues), computation starts on a
    /// background thread at startup and completes before the user needs it.
    /// If the user requests interface mode before computation completes,
    /// the toggle is a no-op until the next frame.
    interface_computed: bool,
    /// Receiver for background interface analysis (large structures only).
    interface_rx: Option<mpsc::Receiver<InterfaceAnalysis>>,
    /// Cached result of `total_residues > LARGE_STRUCTURE_THRESHOLD`, set once
    /// in `App::new` to avoid per-frame O(n) `residue_count()` calls.
    pub is_large: bool,
    /// Loaded MD trajectory; `None` when only a single static structure is shown.
    pub trajectory: Option<Trajectory>,
    /// Index of the currently displayed frame in `trajectory.frames`.
    pub frame_index: usize,
    /// True while frames are auto-advanced on each tick.
    pub playing: bool,
    /// Target frames-per-second for trajectory playback.  Wall-clock-driven so
    /// dropped renders never slow the perceived playback (important over SSH).
    pub playback_fps: f64,
    /// Wrap to frame 0 after the last frame instead of stopping.
    pub loop_playback: bool,
    /// `(when playback resumed, frame_index at that moment)` so wall-clock
    /// pacing can compute the target frame independently of render cadence.
    play_anchor: Option<(Instant, usize)>,
}

impl App {
    pub fn new(
        mut protein: Protein,
        config: AppConfig,
        term_cols: u16,
        term_rows: u16,
        picker: Picker,
    ) -> Self {
        let AppConfig {
            render_mode,
            viz_mode,
            user_explicit_mode,
            color_override,
            trajectory,
        } = config;
        protein.center();
        // If a trajectory is loaded, apply frame 0 immediately so the user sees
        // the first MD frame rather than the unadulterated PDB on startup.
        if let Some(traj) = &trajectory
            && let Some(first) = traj.frames.first()
        {
            let _ = protein.apply_frame(first);
        }
        // If user explicitly requested pLDDT via CLI, trust that even if
        // the heuristic disagrees.
        let has_plddt = protein.has_plddt() || color_override == Some(ColorSchemeType::Plddt);
        let total_residues = protein.residue_count();
        let radius = protein.bounding_radius().max(1.0);

        let vp_rows = term_rows.saturating_sub(4) as f64;
        let vp_cols = term_cols as f64;
        let (font_w, font_h) = picker.font_size();

        let auto_zoom = match render_mode {
            RenderMode::FullHD => {
                let proto = picker.protocol_type();
                let (px_w, px_h) = if proto != ratatui_image::picker::ProtocolType::Halfblocks
                    && font_w > 0
                    && font_h > 0
                {
                    (vp_cols * font_w as f64, vp_rows * font_h as f64)
                } else {
                    // Fallback to braille-like resolution
                    (vp_cols * 2.0, vp_rows * 4.0)
                };
                0.9 * px_w.min(px_h) / (2.0 * radius)
            }
            RenderMode::HalfBlock => {
                let px_w = vp_cols * 2.0;
                let px_h = vp_rows * 4.0;
                0.9 * px_w.min(px_h) / (2.0 * radius)
            }
            RenderMode::Braille => {
                let px_w = vp_cols * 2.0;
                let px_h = vp_rows * 4.0;
                0.9 * px_w.min(px_h) / (2.0 * radius)
            }
        };
        let mut camera = Camera::default();
        camera.zoom = auto_zoom;

        let is_large = total_residues > LARGE_STRUCTURE_THRESHOLD;

        // For large structures, start interface analysis on a background thread
        // so it's ready by the time the user presses 'f'.
        let interface_rx = if is_large {
            let bg_protein = protein.clone();
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let mut ia = analyze_interface(&bg_protein, 4.5);
                if !bg_protein.ligands.is_empty() {
                    ia.binding_pockets = Some(analyze_binding_pockets(&bg_protein, 4.5));
                }
                let _ = tx.send(ia);
            });
            // Interface analysis is running in the background — it'll be ready
            // by the time the user presses 'f'.
            Some(rx)
        } else {
            None
        };

        let (interface_analysis, interface_computed) = if is_large {
            let empty = InterfaceAnalysis {
                contacts: Vec::new(),
                interface_residues: std::collections::HashSet::new(),
                chain_interface_counts: vec![0; protein.chains.len()],
                total_interface_residues: 0,
                binding_pockets: None,
                interactions: Vec::new(),
            };
            (empty, false)
        } else {
            let mut ia = analyze_interface(&protein, 4.5);
            if !protein.ligands.is_empty() {
                ia.binding_pockets = Some(analyze_binding_pockets(&protein, 4.5));
            }
            (ia, true)
        };

        let connection_type = ConnectionType::detect();
        let has_trajectory = trajectory.is_some();

        // For large structures, default to Backbone mode for instant
        // interactivity — but only if the user didn't explicitly choose a mode.
        // Same default for trajectory playback over SSH: regenerating the
        // ribbon mesh per frame is expensive and the resulting traffic
        // saturates SSH bandwidth.
        let viz_mode = if !user_explicit_mode
            && viz_mode == VizMode::Cartoon
            && (is_large || (has_trajectory && connection_type == ConnectionType::Ssh))
        {
            VizMode::Backbone
        } else {
            viz_mode
        };

        // SSH gets a slower default playback rate to keep terminal data volume
        // reasonable; local sessions can run at the full render rate.
        let playback_fps = match connection_type {
            ConnectionType::Local => 30.0,
            ConnectionType::Ssh => 10.0,
        };

        let initial_scheme = color_override.unwrap_or(ColorSchemeType::Structure);
        let color_scheme = ColorScheme::new(initial_scheme, total_residues);
        // Only build ribbon mesh eagerly if we're actually in Cartoon mode.
        // For Backbone/Wireframe, defer until the user switches to Cartoon.
        let (mesh_cache, mesh_dirty) = if viz_mode == VizMode::Cartoon {
            (generate_ribbon_mesh(&protein, &color_scheme), false)
        } else {
            (Vec::new(), true)
        };

        Self {
            protein,
            camera,
            color_scheme,
            viz_mode,
            current_chain: 0,
            render_mode,
            show_help: false,
            show_ligands: true,
            show_interface: false,
            show_interactions: false,
            interface_analysis,
            should_quit: false,
            has_plddt,
            mesh_cache,
            mesh_dirty,
            picker,
            connection_type,
            ssh_hd_warning: false,
            ssh_hd_warning_frames: 0,
            needs_clear: false,
            saved_color_scheme_type: initial_scheme,
            interface_computed,
            interface_rx,
            is_large,
            trajectory,
            frame_index: 0,
            playing: false,
            playback_fps,
            loop_playback: true,
            play_anchor: None,
        }
    }

    pub fn cycle_color(&mut self) {
        if self.show_interface {
            // While interface mode is active, cycle the saved scheme so the
            // user's preference is tracked, but keep displaying Interface colors.
            self.saved_color_scheme_type = self.saved_color_scheme_type.next(self.has_plddt);
        } else {
            let next = self.color_scheme.scheme_type.next(self.has_plddt);
            self.color_scheme = ColorScheme::new(next, self.protein.residue_count());
            self.mesh_dirty = true;
        }
    }

    /// Poll the background interface analysis thread (non-blocking).
    /// Called each frame so results are absorbed as soon as they're ready.
    pub fn poll_background_interface(&mut self) {
        if self.interface_computed {
            return;
        }
        if let Some(rx) = &self.interface_rx {
            match rx.try_recv() {
                Ok(ia) => {
                    self.interface_analysis = ia;
                    self.interface_computed = true;
                    self.interface_rx = None;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    // Still computing — nothing to do yet.
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    // Background thread panicked or dropped the sender.
                    // Drop the rx and fall back to synchronous computation.
                    self.interface_rx = None;
                    let mut ia = analyze_interface(&self.protein, 4.5);
                    if !self.protein.ligands.is_empty() {
                        ia.binding_pockets = Some(analyze_binding_pockets(&self.protein, 4.5));
                    }
                    self.interface_analysis = ia;
                    self.interface_computed = true;
                }
            }
        }
    }

    pub fn cycle_viz_mode(&mut self) {
        self.viz_mode = self.viz_mode.next();
    }

    fn rebuild_interface_colors(&mut self) {
        self.color_scheme = ColorScheme::new_interface(
            self.protein.residue_count(),
            self.current_chain,
            &self.interface_analysis,
            &self.protein,
        );
        self.mesh_dirty = true;
    }

    pub fn toggle_interface(&mut self) {
        self.show_interface = !self.show_interface;
        if self.show_interface {
            // Check if background analysis is ready, otherwise compute synchronously.
            if !self.interface_computed {
                // Determine background thread status without holding a
                // long-lived borrow on self.interface_rx.
                let bg_status = self.interface_rx.as_ref().map(|rx| rx.try_recv());
                match bg_status {
                    Some(Ok(ia)) => {
                        self.interface_analysis = ia;
                        self.interface_computed = true;
                        self.interface_rx = None;
                    }
                    Some(Err(mpsc::TryRecvError::Empty)) => {
                        // Still computing — don't enter interface mode yet.
                        // poll_background_interface() will absorb the result
                        // when ready; the user can press `f` again.
                        self.show_interface = false;
                        return;
                    }
                    Some(Err(mpsc::TryRecvError::Disconnected)) => {
                        // Thread panicked — drop the rx and fall through to
                        // synchronous computation below.
                        self.interface_rx = None;
                    }
                    None => {
                        // No background thread was spawned.
                    }
                }
                // If we still don't have it (no rx, or disconnected), compute synchronously.
                if !self.interface_computed {
                    let mut ia = analyze_interface(&self.protein, 4.5);
                    if !self.protein.ligands.is_empty() {
                        ia.binding_pockets = Some(analyze_binding_pockets(&self.protein, 4.5));
                    }
                    self.interface_analysis = ia;
                    self.interface_computed = true;
                }
            }
            // Save the user's current color scheme before switching to Interface
            self.saved_color_scheme_type = self.color_scheme.scheme_type;
            self.rebuild_interface_colors();
        } else {
            self.show_interactions = false;
            // Restore the user's saved color scheme instead of hardcoding Structure
            self.color_scheme =
                ColorScheme::new(self.saved_color_scheme_type, self.protein.residue_count());
            self.mesh_dirty = true;
        }
    }

    pub fn toggle_interactions(&mut self) {
        if self.show_interface {
            self.show_interactions = !self.show_interactions;
        }
    }

    pub fn toggle_ligands(&mut self) {
        self.show_ligands = !self.show_ligands;
    }

    /// Get the cached ribbon mesh, regenerating if dirty.
    pub fn ribbon_mesh(&mut self) -> &[RibbonTriangle] {
        if self.mesh_dirty {
            self.mesh_cache = generate_ribbon_mesh(&self.protein, &self.color_scheme);
            self.mesh_dirty = false;
        }
        &self.mesh_cache
    }

    pub fn next_chain(&mut self) {
        if !self.protein.chains.is_empty() {
            self.current_chain = (self.current_chain + 1) % self.protein.chains.len();
            if self.show_interface {
                self.rebuild_interface_colors();
            }
        }
    }

    pub fn prev_chain(&mut self) {
        if !self.protein.chains.is_empty() {
            self.current_chain = if self.current_chain == 0 {
                self.protein.chains.len() - 1
            } else {
                self.current_chain - 1
            };
            if self.show_interface {
                self.rebuild_interface_colors();
            }
        }
    }

    pub fn chain_names(&self) -> Vec<String> {
        self.protein.chains.iter().map(|c| c.id.clone()).collect()
    }

    /// Returns `true` when the scene is being actively animated (e.g. auto-rotate).
    /// Used to trigger half-resolution rendering in FullHD mode for smoother
    /// frame rates on large structures.
    pub fn is_interacting(&self) -> bool {
        self.camera.auto_rotate
    }

    pub fn tick(&mut self) {
        self.camera.tick();

        // Tick down SSH HD warning
        if self.ssh_hd_warning && self.ssh_hd_warning_frames > 0 {
            self.ssh_hd_warning_frames -= 1;
            if self.ssh_hd_warning_frames == 0 {
                self.ssh_hd_warning = false;
            }
        }

        self.advance_trajectory();
    }

    /// Advance the displayed trajectory frame based on wall-clock elapsed
    /// time since `play_anchor` was set.  Using elapsed time (rather than
    /// "+= playback_step every tick") means dropped render frames don't slow
    /// the visible playback — important when streaming to a slow SSH client.
    fn advance_trajectory(&mut self) {
        let Some(traj) = self.trajectory.as_ref() else {
            return;
        };
        if !self.playing {
            return;
        }
        let nframes = traj.frames.len();
        if nframes == 0 {
            return;
        }
        let (anchor_instant, anchor_frame) = match self.play_anchor {
            Some(a) => a,
            None => {
                let a = (Instant::now(), self.frame_index);
                self.play_anchor = Some(a);
                a
            }
        };
        let elapsed = anchor_instant.elapsed().as_secs_f64();
        let advance = (elapsed * self.playback_fps) as usize;
        let raw_target = anchor_frame.saturating_add(advance);
        let new_idx = if self.loop_playback {
            raw_target % nframes
        } else if raw_target >= nframes {
            // Reached the end: clamp to last frame and stop.
            self.playing = false;
            self.play_anchor = None;
            nframes - 1
        } else {
            raw_target
        };

        if new_idx != self.frame_index {
            self.frame_index = new_idx;
            // apply_frame ignores any error here because the trajectory length
            // matched topology length at load time.
            if let Err(e) = self.protein.apply_frame(&traj.frames[new_idx]) {
                debug_assert!(false, "apply_frame failed at runtime: {e}");
            }
            self.mesh_dirty = true;
        }
    }

    /// Toggle play / pause.
    pub fn toggle_play(&mut self) {
        if self.trajectory.is_none() {
            return;
        }
        self.playing = !self.playing;
        if self.playing {
            self.play_anchor = Some((Instant::now(), self.frame_index));
        } else {
            self.play_anchor = None;
        }
    }

    /// Jump by `delta` frames (negative = backward).  Pauses playback.
    pub fn step_frame(&mut self, delta: i64) {
        let Some(traj) = self.trajectory.as_ref() else {
            return;
        };
        let nframes = traj.frames.len();
        if nframes == 0 {
            return;
        }
        self.playing = false;
        self.play_anchor = None;
        let n = nframes as i64;
        let mut new_idx = self.frame_index as i64 + delta;
        new_idx = new_idx.rem_euclid(n);
        let new_idx = new_idx as usize;
        if new_idx != self.frame_index {
            self.frame_index = new_idx;
            let _ = self.protein.apply_frame(&traj.frames[new_idx]);
            self.mesh_dirty = true;
        }
    }

    /// Seek to a specific frame.
    pub fn seek(&mut self, idx: usize) {
        let Some(traj) = self.trajectory.as_ref() else {
            return;
        };
        let nframes = traj.frames.len();
        if nframes == 0 {
            return;
        }
        let idx = idx.min(nframes - 1);
        self.playing = false;
        self.play_anchor = None;
        if idx != self.frame_index {
            self.frame_index = idx;
            let _ = self.protein.apply_frame(&traj.frames[idx]);
            self.mesh_dirty = true;
        }
    }

    /// Multiply / divide the playback rate by ~1.5×.  Re-anchors so the
    /// new pace begins from the current frame.
    pub fn change_speed(&mut self, faster: bool) {
        if self.trajectory.is_none() {
            return;
        }
        const FACTOR: f64 = 1.5;
        const MIN_FPS: f64 = 0.5;
        const MAX_FPS: f64 = 120.0;
        if faster {
            self.playback_fps = (self.playback_fps * FACTOR).min(MAX_FPS);
        } else {
            self.playback_fps = (self.playback_fps / FACTOR).max(MIN_FPS);
        }
        if self.playing {
            self.play_anchor = Some((Instant::now(), self.frame_index));
        }
    }

    /// Mark the ribbon mesh cache as dirty, forcing a rebuild on the next frame.
    /// Called when terminal resize occurs or other events invalidate the mesh.
    pub fn mesh_dirty_flag(&mut self) {
        self.mesh_dirty = true;
    }

    /// Recalculate the zoom factor based on current render mode and terminal size.
    /// Call this after changing `render_mode` so the protein fills the viewport
    /// correctly for the new framebuffer dimensions.
    pub fn recalculate_zoom(&mut self, term_cols: u16, term_rows: u16) {
        let radius = self.protein.bounding_radius().max(1.0);
        let vp_rows = term_rows.saturating_sub(4) as f64;
        let vp_cols = term_cols as f64;
        let (font_w, font_h) = self.picker.font_size();

        let (px_w, px_h) = match self.render_mode {
            RenderMode::FullHD => {
                let proto = self.picker.protocol_type();
                if proto != ratatui_image::picker::ProtocolType::Halfblocks
                    && font_w > 0
                    && font_h > 0
                {
                    (vp_cols * font_w as f64, vp_rows * font_h as f64)
                } else {
                    (vp_cols * 2.0, vp_rows * 4.0)
                }
            }
            RenderMode::HalfBlock => (vp_cols * 2.0, vp_rows * 4.0),
            RenderMode::Braille => (vp_cols * 2.0, vp_rows * 4.0),
        };
        self.camera.zoom = 0.9 * px_w.min(px_h) / (2.0 * radius);
    }

    /// Cycle lower render tiers: Braille -> HalfBlock -> Braille.
    /// From FullHD, steps down to HalfBlock (next lower tier).
    /// Bound to `m`.
    pub fn toggle_hd(&mut self, term_cols: u16, term_rows: u16) {
        self.render_mode = match self.render_mode {
            RenderMode::Braille => RenderMode::HalfBlock,
            RenderMode::HalfBlock => RenderMode::Braille,
            RenderMode::FullHD => RenderMode::HalfBlock,
        };
        // Dismiss any stale SSH warning (no longer in FullHD)
        self.ssh_hd_warning = false;
        self.ssh_hd_warning_frames = 0;
        self.needs_clear = true;
        self.recalculate_zoom(term_cols, term_rows);
    }

    /// Upgrade to FullHD (Sixel/Kitty) or back to HalfBlock.
    /// Bound to `M` (Shift+M).  Warns when entering FullHD over SSH.
    pub fn toggle_fullhd(&mut self, term_cols: u16, term_rows: u16) {
        self.render_mode = match self.render_mode {
            RenderMode::FullHD => RenderMode::HalfBlock,
            _ => RenderMode::FullHD,
        };

        self.needs_clear = true;

        if self.render_mode == RenderMode::FullHD && self.connection_type == ConnectionType::Ssh {
            self.ssh_hd_warning = true;
            self.ssh_hd_warning_frames = 90;
        } else {
            // Leaving FullHD — dismiss any active SSH warning
            self.ssh_hd_warning = false;
            self.ssh_hd_warning_frames = 0;
        }

        self.recalculate_zoom(term_cols, term_rows);
    }
}
