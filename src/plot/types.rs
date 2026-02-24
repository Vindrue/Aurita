/// Plot specification and rendered output types.

/// Default number of sample points per curve.
pub const DEFAULT_SAMPLES: usize = 500;
/// Default x-axis range.
pub const DEFAULT_X_MIN: f64 = -10.0;
pub const DEFAULT_X_MAX: f64 = 10.0;
/// Output image dimensions (pixels).
pub const PLOT_WIDTH: u32 = 800;
pub const PLOT_HEIGHT: u32 = 500;

/// Color palette for multiple curves on a dark background (RGB).
pub const SERIES_COLORS: &[(u8, u8, u8)] = &[
    (137, 180, 250), // blue
    (166, 227, 161), // green
    (249, 226, 175), // yellow
    (243, 139, 168), // red
    (203, 166, 247), // mauve
    (148, 226, 213), // teal
    (250, 179, 135), // peach
    (180, 190, 254), // lavender
];

/// A single data series for plotting.
#[derive(Debug, Clone)]
pub struct Series {
    pub label: String,
    /// Sample points. `None` = discontinuity (break the line).
    pub points: Vec<Option<(f64, f64)>>,
}

/// Fully specified plot ready for rendering.
#[derive(Debug, Clone)]
pub struct PlotSpec {
    pub series: Vec<Series>,
    pub x_min: f64,
    pub x_max: f64,
    pub title: Option<String>,
}

/// A rendered plot image.
#[derive(Debug, Clone)]
pub struct RenderedPlot {
    pub png_bytes: Vec<u8>,
    pub spec: PlotSpec,
    pub width: u32,
    pub height: u32,
}
