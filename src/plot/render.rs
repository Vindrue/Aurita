/// Plot rendering pipeline: PlotSpec → PNG bytes via plotters.

use crate::plot::types::*;
use image::codecs::png::PngEncoder;
use image::ImageEncoder;
use plotters::prelude::*;

/// Background color (Catppuccin Mocha base).
const BG_COLOR: RGBColor = RGBColor(30, 30, 46);
/// Axis / grid color.
const AXIS_COLOR: RGBColor = RGBColor(88, 91, 112);
/// Label color (reserved for future font-based labels).
#[allow(dead_code)]
const LABEL_COLOR: RGBColor = RGBColor(186, 194, 222);

/// Render a PlotSpec to a PNG image.
pub fn render_plot(spec: &PlotSpec) -> Result<RenderedPlot, String> {
    let width = PLOT_WIDTH;
    let height = PLOT_HEIGHT;
    let mut buf = vec![0u8; (width * height * 3) as usize];

    {
        let root = BitMapBackend::with_buffer(&mut buf, (width, height)).into_drawing_area();
        root.fill(&BG_COLOR).map_err(|e| format!("fill: {}", e))?;

        let (y_min, y_max) = compute_y_range(&spec.series);

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .build_cartesian_2d(spec.x_min..spec.x_max, y_min..y_max)
            .map_err(|e| format!("chart build: {}", e))?;

        chart
            .configure_mesh()
            .axis_style(AXIS_COLOR)
            .bold_line_style(AXIS_COLOR.mix(0.3))
            .light_line_style(AXIS_COLOR.mix(0.1))
            .x_labels(0)
            .y_labels(0)
            .draw()
            .map_err(|e| format!("mesh: {}", e))?;

        for (i, series) in spec.series.iter().enumerate() {
            let (r, g, b) = SERIES_COLORS[i % SERIES_COLORS.len()];
            let color = RGBColor(r, g, b);
            let segments = split_segments(&series.points);

            for segment in &segments {
                chart
                    .draw_series(LineSeries::new(segment.iter().copied(), color.stroke_width(2)))
                    .map_err(|e| format!("draw series: {}", e))?;
            }
        }

        root.present().map_err(|e| format!("present: {}", e))?;
    }

    // Encode RGB buffer to PNG
    let png_bytes = encode_rgb_to_png(&buf, width, height)?;

    Ok(RenderedPlot {
        png_bytes,
        spec: spec.clone(),
        width,
        height,
    })
}

/// Split a point series at None (discontinuities) into continuous segments.
fn split_segments(points: &[Option<(f64, f64)>]) -> Vec<Vec<(f64, f64)>> {
    let mut segments = Vec::new();
    let mut current = Vec::new();

    for pt in points {
        match pt {
            Some(p) => current.push(*p),
            None => {
                if !current.is_empty() {
                    segments.push(std::mem::take(&mut current));
                }
            }
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

/// Compute a good y-axis range from the data, with padding and clamping.
fn compute_y_range(all_series: &[Series]) -> (f64, f64) {
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for series in all_series {
        for pt in &series.points {
            if let Some((_, y)) = pt {
                if y.is_finite() {
                    y_min = y_min.min(*y);
                    y_max = y_max.max(*y);
                }
            }
        }
    }

    // Clamp to avoid asymptote blowup
    y_min = y_min.max(-1000.0);
    y_max = y_max.min(1000.0);

    // Fallback for empty/constant data
    if !y_min.is_finite() || !y_max.is_finite() {
        return (-1.0, 1.0);
    }
    if (y_max - y_min).abs() < 1e-10 {
        return (y_min - 1.0, y_max + 1.0);
    }

    // Add 10% padding
    let pad = (y_max - y_min) * 0.1;
    (y_min - pad, y_max + pad)
}

/// Encode a raw RGB pixel buffer to PNG.
fn encode_rgb_to_png(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let mut png = Vec::new();
    let encoder = PngEncoder::new(&mut png);
    encoder
        .write_image(rgb, width, height, image::ExtendedColorType::Rgb8)
        .map_err(|e| format!("PNG encode: {}", e))?;
    Ok(png)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_spec() -> PlotSpec {
        let points: Vec<Option<(f64, f64)>> = (0..100)
            .map(|i| {
                let x = -5.0 + 10.0 * i as f64 / 99.0;
                Some((x, x * x))
            })
            .collect();
        PlotSpec {
            series: vec![Series {
                label: "x^2".to_string(),
                points,
            }],
            x_min: -5.0,
            x_max: 5.0,
            title: None,
        }
    }

    #[test]
    fn test_render_simple() {
        let spec = make_simple_spec();
        let result = render_plot(&spec).unwrap();
        assert!(!result.png_bytes.is_empty());
        assert_eq!(result.width, PLOT_WIDTH);
        assert_eq!(result.height, PLOT_HEIGHT);
        // PNG magic bytes
        assert_eq!(&result.png_bytes[1..4], b"PNG");
    }

    #[test]
    fn test_render_multi_series() {
        let sin_pts: Vec<Option<(f64, f64)>> = (0..100)
            .map(|i| {
                let x = -3.14 + 6.28 * i as f64 / 99.0;
                Some((x, x.sin()))
            })
            .collect();
        let cos_pts: Vec<Option<(f64, f64)>> = (0..100)
            .map(|i| {
                let x = -3.14 + 6.28 * i as f64 / 99.0;
                Some((x, x.cos()))
            })
            .collect();
        let spec = PlotSpec {
            series: vec![
                Series { label: "sin".to_string(), points: sin_pts },
                Series { label: "cos".to_string(), points: cos_pts },
            ],
            x_min: -3.14,
            x_max: 3.14,
            title: None,
        };
        let result = render_plot(&spec).unwrap();
        assert!(!result.png_bytes.is_empty());
    }

    #[test]
    fn test_render_with_gaps() {
        let mut points: Vec<Option<(f64, f64)>> = Vec::new();
        for i in 0..50 {
            let x = -5.0 + 10.0 * i as f64 / 99.0;
            points.push(Some((x, x)));
        }
        points.push(None); // discontinuity
        for i in 50..100 {
            let x = -5.0 + 10.0 * i as f64 / 99.0;
            points.push(Some((x, x + 2.0)));
        }
        let spec = PlotSpec {
            series: vec![Series { label: "f".to_string(), points }],
            x_min: -5.0,
            x_max: 5.0,
            title: None,
        };
        let result = render_plot(&spec).unwrap();
        assert!(!result.png_bytes.is_empty());
    }

    #[test]
    fn test_split_segments() {
        let points = vec![Some((0.0, 0.0)), Some((1.0, 1.0)), None, Some((3.0, 3.0))];
        let segs = split_segments(&points);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].len(), 2);
        assert_eq!(segs[1].len(), 1);
    }

    #[test]
    fn test_y_range_with_empty() {
        let (y_min, y_max) = compute_y_range(&[]);
        assert!(y_min < y_max);
    }

    #[test]
    fn test_y_range_constant() {
        let series = Series {
            label: "c".to_string(),
            points: vec![Some((0.0, 5.0)), Some((1.0, 5.0))],
        };
        let (y_min, y_max) = compute_y_range(&[series]);
        assert!(y_min < 5.0);
        assert!(y_max > 5.0);
    }
}
