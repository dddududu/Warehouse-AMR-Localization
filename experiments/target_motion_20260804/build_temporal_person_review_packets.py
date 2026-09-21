from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from dataset_io.depth_loader import load_depth_png
from dataset_io.lidar_loader import load_pcd_xyz
from dataset_io.sequence_dataset import WarehouseSequenceDataset
from experiments.target_motion_20260803.evaluate_pretrained_person_detector import _box_iou, _build_model
from geometry.se3 import transform_points


def _load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Expected a mapping.")
    return payload


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = ["C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _link_box(previous_box: np.ndarray, boxes: np.ndarray, scores: np.ndarray, minimum_score: float = 0.20) -> tuple[np.ndarray | None, float, float]:
    valid = scores >= float(minimum_score)
    if not np.any(valid):
        return None, 0.0, 0.0
    candidate_boxes, candidate_scores = boxes[valid], scores[valid]
    overlaps = _box_iou(previous_box.reshape(1, 4), candidate_boxes).reshape(-1)
    ranking = overlaps * candidate_scores
    best = int(np.argmax(ranking))
    if float(overlaps[best]) <= 0.0:
        return None, 0.0, 0.0
    return candidate_boxes[best], float(candidate_scores[best]), float(overlaps[best])


def _motion_hint(world_centers: list[np.ndarray | None]) -> tuple[str, float | None, float | None]:
    observed = [center for center in world_centers if center is not None]
    if len(observed) < 3:
        return "insufficient_3d_evidence", None, None
    path_length = float(sum(np.linalg.norm(right - left) for left, right in zip(observed[:-1], observed[1:], strict=True)))
    net_displacement = float(np.linalg.norm(observed[-1] - observed[0]))
    if net_displacement < 0.20 and path_length < 0.35:
        return "likely_static_relative_to_map", path_length, net_displacement
    if net_displacement >= 0.50 or path_length >= 0.80:
        return "potentially_moving", path_length, net_displacement
    return "ambiguous_motion", path_length, net_displacement


def _person_detections(model: torch.nn.Module, image_path: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read {image_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255.0).to(device)
    output = model([tensor])[0]
    person = output["labels"].detach().cpu().numpy() == 1
    return (
        output["boxes"].detach().cpu().numpy()[person].astype(np.float32, copy=False),
        output["scores"].detach().cpu().numpy()[person].astype(np.float32, copy=False),
    )


def _lidar_front_cluster_center(dataset: WarehouseSequenceDataset, frame_idx: int, box: np.ndarray) -> np.ndarray | None:
    points_sensor = load_pcd_xyz(dataset.frame_index[frame_idx].lidar_path).astype(np.float64)
    points_camera = transform_points(dataset.calibration.T_cam1_os.matrix, points_sensor).astype(np.float64)
    projected, _ = cv2.projectPoints(points_camera, np.zeros(3), np.zeros(3), dataset.camera_left.build_K(), dataset.camera_left.distortion)
    pixels = projected.reshape(-1, 2)
    x1, y1, x2, y2 = box.tolist()
    inner_x1, inner_x2 = x1 + 0.20 * (x2 - x1), x1 + 0.80 * (x2 - x1)
    inner_y1, inner_y2 = y1 + 0.10 * (y2 - y1), y1 + 0.90 * (y2 - y1)
    inside = (
        np.isfinite(points_camera).all(axis=1)
        & (points_camera[:, 2] > 0.2)
        & (pixels[:, 0] >= inner_x1)
        & (pixels[:, 0] <= inner_x2)
        & (pixels[:, 1] >= inner_y1)
        & (pixels[:, 1] <= inner_y2)
    )
    selected = points_sensor[inside]
    if selected.shape[0] < 3:
        return None
    selected_camera = points_camera[inside]
    front_limit = np.quantile(selected_camera[:, 2], 0.30)
    front_cluster = selected[selected_camera[:, 2] <= front_limit]
    if front_cluster.shape[0] < 3:
        front_cluster = selected
    points_world = transform_points(dataset.ground_truth.poses_4x4[frame_idx], front_cluster)
    return np.median(points_world, axis=0).astype(np.float64)


def _world_center(dataset: WarehouseSequenceDataset, frame_idx: int, box: np.ndarray | None, depth_scale_m: float) -> tuple[np.ndarray | None, str | None]:
    if box is None:
        return None, None
    record = dataset.frame_index[frame_idx]
    depth = load_depth_png(record.depth_left_path).astype(np.float64) * float(depth_scale_m)
    height, width = depth.shape
    x1, y1, x2, y2 = box.tolist()
    inner_x1 = int(np.clip(round(x1 + 0.35 * (x2 - x1)), 0, width - 1))
    inner_x2 = int(np.clip(round(x1 + 0.65 * (x2 - x1)), inner_x1 + 1, width))
    inner_y1 = int(np.clip(round(y1 + 0.25 * (y2 - y1)), 0, height - 1))
    inner_y2 = int(np.clip(round(y1 + 0.75 * (y2 - y1)), inner_y1 + 1, height))
    values = depth[inner_y1:inner_y2, inner_x1:inner_x2]
    valid = values[np.isfinite(values) & (values >= 0.2) & (values <= 12.0)]
    if valid.size > 0:
        depth_m = float(np.median(valid))
        center = np.asarray([[(x1 + x2) * 0.5, (y1 + y2) * 0.5]], dtype=np.float64)
        normalized = cv2.undistortPoints(center.reshape(-1, 1, 2), dataset.camera_left.build_K(), dataset.camera_left.distortion).reshape(2)
        point_camera = np.asarray([[normalized[0] * depth_m, normalized[1] * depth_m, depth_m]], dtype=np.float64)
        point_sensor = transform_points(dataset.calibration.T_os_cam_left, point_camera)
        return transform_points(dataset.ground_truth.poses_4x4[frame_idx], point_sensor)[0].astype(np.float64), "depth_box_interior"
    lidar_center = _lidar_front_cluster_center(dataset, frame_idx, box)
    if lidar_center is not None:
        return lidar_center, "lidar_front_cluster"
    return None, None


def _track_window(
    model: torch.nn.Module,
    dataset: WarehouseSequenceDataset,
    anchor_frame: int,
    anchor_box: np.ndarray,
    window_radius: int,
    depth_scale_m: float,
    device: torch.device,
) -> tuple[list[int], list[np.ndarray | None], list[float], list[float], list[np.ndarray | None], list[str | None]]:
    frames = list(range(max(0, anchor_frame - window_radius), min(len(dataset), anchor_frame + window_radius + 1)))
    anchor_position = frames.index(anchor_frame)
    boxes: list[np.ndarray | None] = [None] * len(frames)
    scores = [0.0] * len(frames)
    overlaps = [0.0] * len(frames)
    boxes[anchor_position] = anchor_box
    scores[anchor_position] = 1.0
    for direction in (-1, 1):
        previous = anchor_box
        for position in range(anchor_position + direction, -1 if direction < 0 else len(frames), direction):
            detections, detection_scores = _person_detections(model, dataset.frame_index[frames[position]].image_left_path, device)
            linked, score, overlap = _link_box(previous, detections, detection_scores)
            boxes[position], scores[position], overlaps[position] = linked, score, overlap
            if linked is None:
                break
            previous = linked
    centers_with_source = [_world_center(dataset, frame_idx, box, depth_scale_m) for frame_idx, box in zip(frames, boxes, strict=True)]
    world_centers = [item[0] for item in centers_with_source]
    sources = [item[1] for item in centers_with_source]
    return frames, boxes, scores, overlaps, world_centers, sources


def _draw_packet_page(path: Path, rows: list[dict[str, Any]]) -> None:
    panel_width, panel_height = 250, 141
    canvas = Image.new("RGB", (5 * panel_width + 80, max(len(rows), 1) * 230 + 95), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 20), "五帧人员核验包：红框为时序关联建议，动静提示仅供人工判断", fill="#172B4D", font=_font(24, True))
    for row_index, row in enumerate(rows):
        y = 70 + row_index * 230
        for column, (image_path, box, frame_idx) in enumerate(zip(row["window_image_paths"], row["track_boxes"], row["window_frames"], strict=True)):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            view = Image.fromarray(image).resize((panel_width, panel_height), resample=Image.Resampling.BILINEAR)
            if box is not None:
                scale = np.asarray((panel_width / image.shape[1], panel_height / image.shape[0], panel_width / image.shape[1], panel_height / image.shape[0]))
                ImageDraw.Draw(view).rectangle(tuple(np.asarray(box) * scale), outline="#D64545", width=3)
            x = 25 + column * panel_width
            canvas.paste(view, (x, y))
            draw.text((x, y + panel_height + 3), f"帧 {frame_idx}", fill="#52606D", font=_font(13))
        evidence = row["motion_hint"]
        if row["net_world_displacement_m"] is not None:
            evidence += f" | 位移 {row['net_world_displacement_m']:.2f} m"
        draw.text((25, y + panel_height + 24), f"{row['split']} | {row['sequence']} | 中心帧 {row['anchor_frame']} | {evidence}", fill="#334E68", font=_font(15, True))
    canvas.save(path)


def run(manifest_path: str | Path, quality_config_path: str | Path, output_dir: str | Path, window_radius: int) -> dict[str, Any]:
    config = _load_yaml(quality_config_path)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = list(csv.DictReader(Path(manifest_path).open(encoding="utf-8-sig")))
    entries = {
        (split, Path(entry["sequence_root"]).name.lower()): entry
        for split, values in config["splits"].items()
        for entry in values
    }
    datasets = {
        key: WarehouseSequenceDataset(entry["sequence_root"], entry["calibration_path"], {"load_lidar": False})
        for key, entry in entries.items()
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name = _build_model({"name": "fasterrcnn_resnet50_fpn_v2"}, device)
    model.eval()
    reviewed: list[dict[str, Any]] = []
    depth_scale_m = float(config["analysis"]["depth_scale_m"])
    with torch.no_grad():
        for index, row in enumerate(records, start=1):
            key = (row["split"], row["sequence"])
            dataset = datasets[key]
            anchor = int(row["anchor_frame"])
            anchor_box = np.asarray(json.loads(row["suggested_box_json"]), dtype=np.float32)
            frames, boxes, scores, overlaps, centers, center_sources = _track_window(model, dataset, anchor, anchor_box, window_radius, depth_scale_m, device)
            hint, path_length, displacement = _motion_hint(centers)
            reviewed.append(
                {
                    "split": row["split"],
                    "sequence": row["sequence"],
                    "anchor_frame": anchor,
                    "window_start_frame": frames[0],
                    "window_end_frame": frames[-1],
                    "window_frames": frames,
                    "window_image_paths": [str(dataset.frame_index[frame_idx].image_left_path) for frame_idx in frames],
                    "track_boxes": [box.tolist() if box is not None else None for box in boxes],
                    "track_scores": scores,
                    "track_link_iou": overlaps,
                    "world_centers_xyz": [center.tolist() if center is not None else None for center in centers],
                    "world_center_sources": center_sources,
                    "track_observation_count": sum(center is not None for center in centers),
                    "world_path_length_m": path_length,
                    "net_world_displacement_m": displacement,
                    "motion_hint": hint,
                    "manual_person_label": "pending",
                    "manual_motion_label": "pending",
                    "manual_instance_id": "pending",
                    "manual_box_json": json.dumps(anchor_box.tolist()),
                    "annotation_status": "pending_manual_verification",
                }
            )
            if index % 10 == 0 or index == len(records):
                print(f"built review packets {index}/{len(records)}")
    nested_keys = {"window_frames", "window_image_paths", "track_boxes", "track_scores", "track_link_iou", "world_centers_xyz", "world_center_sources"}
    fields = [key for key in reviewed[0] if key not in nested_keys] + [f"{key}_json" for key in sorted(nested_keys)] if reviewed else []
    with (root / "temporal_person_review_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in reviewed:
            serialized = {key: value for key, value in row.items() if key not in nested_keys}
            for key in nested_keys:
                serialized[f"{key}_json"] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(serialized)
    for page_index, start in enumerate(range(0, len(reviewed), 4), start=1):
        _draw_packet_page(root / f"review_packet_{page_index:02d}.png", reviewed[start : start + 4])
    hint_counts: dict[str, int] = defaultdict(int)
    for row in reviewed:
        hint_counts[row["motion_hint"]] += 1
    instructions = """# 五帧人员与动静标注说明

每条记录展示中心帧前后各两帧。红框是 COCO 检测器的时序关联建议；世界坐标位移优先由框内深度计算，深度缺失时才回退到投影入框的前方点云簇。两者都只是帮助判断相对地图的运动证据。请先填写 `manual_person_label`（person / non_person）；`manual_box_json` 已预填建议框，真实人员应确认或修订它，并填写五帧一致的 `manual_instance_id` 和 `manual_motion_label`（static / moving / uncertain）。任何自动框、自动轨迹或 motion_hint 都不能直接转为训练真值。
"""
    (root / "annotation_instructions.md").write_text(instructions, encoding="utf-8")
    summary = {
        "model": model_name,
        "review_packets": len(reviewed),
        "motion_hint_counts": dict(hint_counts),
        "policy": "运动提示只服务人工核验；人工确认前不训练、不调阈值、不进入定位门控。",
    }
    (root / "temporal_person_review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal person review packets with non-label motion evidence.")
    parser.add_argument("--manifest", default="outputs/target_motion_20260804/detector_proposal_annotation_manifest/detector_proposal_annotation_manifest.csv")
    parser.add_argument("--quality-config", default="experiments/target_motion_20260804/person_component_quality_audit.yaml")
    parser.add_argument("--output-dir", default="outputs/target_motion_20260804/temporal_person_review_packets")
    parser.add_argument("--window-radius", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.quality_config, args.output_dir, args.window_radius), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
