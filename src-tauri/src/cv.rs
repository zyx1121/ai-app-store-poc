//! Computer vision through the CV platform service. The service is a KServe v2
//! server (Triton on NVIDIA, OpenVINO Model Server elsewhere) that loads ONNX
//! models from a shared model repository. This module is the client the UI
//! talks to: it letterboxes the image, sends one FP32 tensor with the KServe
//! binary extension, and maps the detector's rows back to image pixels. The
//! frontend never sees which server answered.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::services;

/// Windows-side port of the CV service; both server implementations publish here.
pub const PORT: u16 = 8900;

/// Detectors the store knows how to run. Every entry is an ONNX file in the
/// model repository (`repo/<name>/1/model.onnx`) with the same tensor contract:
/// input `images` FP32 [1,3,S,S] in 0..1, output `output0` FP32 [1,N,6] rows of
/// (x1, y1, x2, y2, score, class) in letterboxed pixels, NMS already applied.
#[derive(Debug, Clone, Copy)]
pub struct Detector {
    pub name: &'static str,
    pub display_name: &'static str,
    pub url: &'static str,
    pub size: u32,
}

pub const DETECTORS: &[Detector] = &[
    Detector {
        name: "yolov10n",
        display_name: "YOLOv10n (COCO, 9 MB)",
        url: "https://huggingface.co/onnx-community/yolov10n/resolve/main/onnx/model.onnx",
        size: 640,
    },
    Detector {
        name: "yolov10s",
        display_name: "YOLOv10s (COCO, 31 MB)",
        url: "https://huggingface.co/onnx-community/yolov10s/resolve/main/onnx/model.onnx",
        size: 640,
    },
];

const INPUT_NAME: &str = "images";
const OUTPUT_NAME: &str = "output0";
/// Letterbox fill, the value Ultralytics trains with.
const PAD: u8 = 114;

pub fn detector(name: &str) -> Option<&'static Detector> {
    DETECTORS.iter().find(|d| d.name == name)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Detection {
    pub label: String,
    pub score: f32,
    /// x1, y1, x2, y2 in pixels of the image the UI sent.
    #[serde(rename = "box")]
    pub bbox: [f32; 4],
}

#[derive(Debug, Clone, Serialize)]
pub struct DetectResult {
    pub model: String,
    /// which server answered, e.g. `triton` or `openvino`
    pub backend: String,
    pub detections: Vec<Detection>,
    /// wall time of the HTTP round trip to the server
    pub inference_ms: u32,
    /// decode + letterbox + inference + post-processing
    pub total_ms: u32,
}

/// Where a letterboxed image landed inside the square input.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Letterbox {
    scale: f32,
    src_w: u32,
    src_h: u32,
}

impl Letterbox {
    fn new(src_w: u32, src_h: u32, size: u32) -> Self {
        let scale = size as f32 / src_w.max(src_h).max(1) as f32;
        Self {
            scale,
            src_w,
            src_h,
        }
    }

    fn scaled(&self) -> (u32, u32) {
        (
            ((self.src_w as f32 * self.scale).round() as u32).max(1),
            ((self.src_h as f32 * self.scale).round() as u32).max(1),
        )
    }

    /// Map a box in letterboxed pixels back to source pixels, clamped to the image.
    fn unmap(&self, b: [f32; 4]) -> [f32; 4] {
        let w = self.src_w as f32;
        let h = self.src_h as f32;
        [
            (b[0] / self.scale).clamp(0.0, w),
            (b[1] / self.scale).clamp(0.0, h),
            (b[2] / self.scale).clamp(0.0, w),
            (b[3] / self.scale).clamp(0.0, h),
        ]
    }
}

/// Decode, resize so the long side is `size`, paste top-left on a PAD square,
/// and lay out as CHW float32 in 0..1. Returns the tensor and the mapping.
fn preprocess(image_bytes: &[u8], size: u32) -> Result<(Vec<f32>, Letterbox)> {
    let img = image::load_from_memory(image_bytes)
        .map_err(|e| Error::Other(format!("cannot decode image: {e}")))?
        .into_rgb8();
    let lb = Letterbox::new(img.width(), img.height(), size);
    let (w, h) = lb.scaled();
    let resized = image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle);
    let plane = (size * size) as usize;
    let pad = PAD as f32 / 255.0;
    let mut chw = vec![pad; plane * 3];
    for (x, y, px) in resized.enumerate_pixels() {
        let i = (y * size + x) as usize;
        chw[i] = px[0] as f32 / 255.0;
        chw[plane + i] = px[1] as f32 / 255.0;
        chw[2 * plane + i] = px[2] as f32 / 255.0;
    }
    Ok((chw, lb))
}

#[derive(Serialize)]
struct InferInput<'a> {
    name: &'a str,
    shape: [usize; 4],
    datatype: &'a str,
    parameters: InferParams,
}

#[derive(Serialize)]
struct InferParams {
    binary_data_size: usize,
}

#[derive(Serialize)]
struct InferOutputReq<'a> {
    name: &'a str,
}

#[derive(Serialize)]
struct InferRequest<'a> {
    inputs: [InferInput<'a>; 1],
    outputs: [InferOutputReq<'a>; 1],
}

#[derive(Deserialize)]
struct InferResponse {
    #[serde(default)]
    outputs: Vec<InferOutput>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
struct InferOutput {
    name: String,
    #[serde(default)]
    shape: Vec<i64>,
    #[serde(default)]
    data: Vec<f32>,
}

/// Body of a KServe v2 request with the binary tensor extension: the JSON
/// header followed by the raw little-endian floats. Returns (header length, body).
fn build_body(tensor: &[f32], size: u32) -> (usize, Vec<u8>) {
    let bytes = tensor.len() * 4;
    let header = serde_json::to_vec(&InferRequest {
        inputs: [InferInput {
            name: INPUT_NAME,
            shape: [1, 3, size as usize, size as usize],
            datatype: "FP32",
            parameters: InferParams {
                binary_data_size: bytes,
            },
        }],
        outputs: [InferOutputReq { name: OUTPUT_NAME }],
    })
    .expect("static request shape");
    let mut body = Vec::with_capacity(header.len() + bytes);
    body.extend_from_slice(&header);
    for v in tensor {
        body.extend_from_slice(&v.to_le_bytes());
    }
    (header.len(), body)
}

/// Turn `[N,6]` rows into detections in source pixels, dropping low scores and
/// the zero-padded rows NMS-free detectors emit.
fn postprocess(rows: &[f32], lb: Letterbox, min_score: f32) -> Vec<Detection> {
    rows.as_chunks::<6>()
        .0
        .iter()
        .filter(|r| r[4] >= min_score && r[2] > r[0] && r[3] > r[1])
        .map(|r| Detection {
            label: coco_label(r[5].round() as i64).to_string(),
            score: r[4],
            bbox: lb.unmap([r[0], r[1], r[2], r[3]]),
        })
        .collect()
}

/// Run `model` on `image_bytes` through the CV service.
pub async fn detect(
    app: &tauri::AppHandle,
    model: &str,
    image_bytes: &[u8],
    min_score: f32,
) -> Result<DetectResult> {
    let started = Instant::now();
    let det = detector(model).ok_or_else(|| Error::Other(format!("unknown detector `{model}`")))?;
    let backend = services::backend_name(app, services::ServiceId::Cv);

    let (tensor, lb) = preprocess(image_bytes, det.size)?;
    let (header_len, body) = build_body(&tensor, det.size);

    let http = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(60))
        .build()?;
    let url = format!("http://localhost:{PORT}/v2/models/{}/infer", det.name);
    let t0 = Instant::now();
    let resp = http
        .post(&url)
        .header("Content-Type", "application/octet-stream")
        .header("Inference-Header-Content-Length", header_len.to_string())
        .body(body)
        .send()
        .await
        .map_err(|e| Error::Other(format!("CV service unreachable on :{PORT}: {e}")))?;
    let status = resp.status();
    let text = resp.text().await?;
    let inference_ms = t0.elapsed().as_millis() as u32;
    if !status.is_success() {
        let msg = serde_json::from_str::<InferResponse>(&text)
            .ok()
            .and_then(|r| r.error)
            .unwrap_or_else(|| text.chars().take(300).collect());
        return Err(Error::Other(format!(
            "CV service returned {status} for `{}`: {msg}",
            det.name
        )));
    }
    let parsed: InferResponse = serde_json::from_str(&text)
        .map_err(|e| Error::Other(format!("CV service sent unreadable JSON: {e}")))?;
    let out = parsed
        .outputs
        .into_iter()
        .find(|o| o.name == OUTPUT_NAME)
        .ok_or_else(|| Error::Other(format!("response has no `{OUTPUT_NAME}` tensor")))?;
    if out.shape.last() != Some(&6) {
        return Err(Error::Other(format!(
            "unexpected output shape {:?}; expected [1, N, 6]",
            out.shape
        )));
    }
    let detections = postprocess(&out.data, lb, min_score);
    Ok(DetectResult {
        model: det.name.to_string(),
        backend,
        detections,
        inference_ms,
        total_ms: started.elapsed().as_millis() as u32,
    })
}

/// COCO 80 classes, the label set every starter detector is trained on.
const COCO: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

fn coco_label(class: i64) -> &'static str {
    usize::try_from(class)
        .ok()
        .and_then(|i| COCO.get(i))
        .copied()
        .unwrap_or("object")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_scales_the_long_side_and_unmaps() {
        let lb = Letterbox::new(1280, 720, 640);
        assert_eq!(lb.scaled(), (640, 360));
        // A box in the letterboxed frame maps back by the inverse scale and clamps to the image.
        assert_eq!(lb.unmap([0.0, 0.0, 320.0, 180.0]), [0.0, 0.0, 640.0, 360.0]);
        assert_eq!(
            lb.unmap([-5.0, 0.0, 700.0, 400.0]),
            [0.0, 0.0, 1280.0, 720.0]
        );
    }

    #[test]
    fn preprocess_lays_out_chw_with_padding() {
        // 4x2 red image -> scaled to 8x4 inside an 8x8 square; rows 4..8 are PAD.
        let img = image::RgbImage::from_pixel(4, 2, image::Rgb([255, 0, 0]));
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let (t, lb) = preprocess(png.get_ref(), 8).unwrap();
        assert_eq!(t.len(), 3 * 64);
        assert_eq!(lb.scaled(), (8, 4));
        assert_eq!(t[0], 1.0); // R of pixel (0,0)
        assert_eq!(t[64], 0.0); // G of pixel (0,0)
        let pad = PAD as f32 / 255.0;
        assert_eq!(t[8 * 7], pad); // R of a padded row
        assert_eq!(t[64 + 8 * 7], pad);
    }

    #[test]
    fn body_is_header_then_raw_floats() {
        let (n, body) = build_body(&[1.0, 2.0], 640);
        let header: serde_json::Value = serde_json::from_slice(&body[..n]).unwrap();
        assert_eq!(header["inputs"][0]["name"], "images");
        assert_eq!(header["inputs"][0]["parameters"]["binary_data_size"], 8);
        assert_eq!(
            header["inputs"][0]["shape"],
            serde_json::json!([1, 3, 640, 640])
        );
        assert_eq!(&body[n..n + 4], &1.0f32.to_le_bytes());
        assert_eq!(body.len(), n + 8);
    }

    #[test]
    fn postprocess_filters_and_labels() {
        let lb = Letterbox::new(1280, 720, 640);
        let rows = [
            10.0, 10.0, 110.0, 60.0, 0.9, 0.0, // person
            0.0, 0.0, 50.0, 50.0, 0.1, 2.0, // below threshold
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // padded row
            5.0, 5.0, 20.0, 20.0, 0.5, 999.0, // unknown class
        ];
        let d = postprocess(&rows, lb, 0.25);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].label, "person");
        assert_eq!(d[0].bbox, [20.0, 20.0, 220.0, 120.0]);
        assert_eq!(d[1].label, "object");
    }

    #[test]
    fn detectors_are_unique_and_known() {
        let mut names: Vec<_> = DETECTORS.iter().map(|d| d.name).collect();
        names.dedup();
        assert_eq!(names.len(), DETECTORS.len());
        assert!(detector("yolov10n").is_some());
        assert!(detector("nope").is_none());
    }
}
