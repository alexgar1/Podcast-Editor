#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gzip
import io
import json
import re
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from speaker_diarization import (
    SpeakerDiarizationConfig,
    add_speaker_diarization_args,
    append_speaker_diarization_command_args,
    speaker_diarization_config_from_args,
)

TICKS_PER_SECOND = 254_016_000_000
TIMECODE_PATTERN = re.compile(
    r"(?:Timecode:\s*)?((?:\d{2}:){2,3}\d{2})\s*[–—-]\s*((?:\d{2}:){2,3}\d{2})"
)
GENERATED_SEQUENCE_PATTERN = re.compile(r"\s*-\s*(?:selects|concision)(?:\s+\d+)?$", re.IGNORECASE)


class PodcastEditorError(Exception):
    pass


def emit_status(message: str) -> None:
    print(f"[status] {message}", file=sys.stderr, flush=True)


def is_generated_selects_sequence_name(name: str | None) -> bool:
    if not name:
        return False
    return GENERATED_SEQUENCE_PATTERN.search(name) is not None


def choose_default_sequence(sequences: list[ET.Element]) -> ET.Element | None:
    if len(sequences) == 1:
        return sequences[0]

    non_generated_sequences = [
        sequence for sequence in sequences if not is_generated_selects_sequence_name(sequence.findtext("Name"))
    ]
    if len(non_generated_sequences) == 1:
        return non_generated_sequences[0]
    return None


@dataclass(frozen=True)
class TimecodeRange:
    start: str
    end: str


@dataclass(frozen=True)
class TickRange:
    start: int
    end: int


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    selected: bool


@dataclass(frozen=True)
class SourceTrackItemTemplate:
    original_track_item_id: int
    media_kind: str
    track_index: int
    track: ET.Element
    track_item: ET.Element
    component: ET.Element
    subclip: ET.Element
    clip: ET.Element
    timeline_start: int
    timeline_end: int
    source_start: int
    source_end: int
    secondary_templates: tuple[ET.Element, ...]

    def overlap(self, range_start: int, range_end: int) -> tuple[int, int] | None:
        overlap_start = max(self.timeline_start, range_start)
        overlap_end = min(self.timeline_end, range_end)
        if overlap_end <= overlap_start:
            return None
        return overlap_start, overlap_end

    def source_bounds_for_overlap(self, overlap_start: int, overlap_end: int) -> tuple[int, int]:
        offset = overlap_start - self.timeline_start
        source_start = self.source_start + offset
        source_end = source_start + (overlap_end - overlap_start)
        return source_start, source_end


@dataclass(frozen=True)
class SequenceLinkTemplate:
    original_link_id: int
    track_item_ids: tuple[int, ...]


@dataclass
class TemplateContext:
    root_bin: ET.Element
    sequence_project_item: ET.Element
    sequence_master_clip: ET.Element
    sequence_logging_info: ET.Element
    sequence_master_audio_chain: ET.Element
    sequence_audio_clip: ET.Element
    sequence_video_clip: ET.Element
    sequence_audio_source: ET.Element
    sequence_video_source: ET.Element
    sequence_audio_channel_group: ET.Element
    sequence_audio_secondary_templates: list[ET.Element]
    sequence: ET.Element
    video_group: ET.Element
    audio_group: ET.Element
    data_group: ET.Element | None
    video_group_component_template: ET.Element
    audio_master_track: ET.Element
    video_tracks: list[ET.Element]
    audio_tracks: list[ET.Element]
    primary_video_track: ET.Element
    primary_audio_track: ET.Element
    video_track_item_template: ET.Element
    audio_track_item_template: ET.Element
    video_component_template: ET.Element
    audio_component_template: ET.Element
    video_subclip_template: ET.Element
    audio_subclip_template: ET.Element
    video_clip_template: ET.Element
    audio_clip_template: ET.Element
    audio_secondary_templates: list[ET.Element]
    link_template: ET.Element | None
    duration_ticks: int
    frame_ticks: int
    fps: int
    source_video_items: list[SourceTrackItemTemplate]
    source_audio_items: list[SourceTrackItemTemplate]
    link_groups: list[SequenceLinkTemplate]


@dataclass
class ClonedSequenceContext:
    sequence: ET.Element
    project_item: ET.Element
    master_clip: ET.Element
    video_group: ET.Element
    audio_group: ET.Element
    project_audio_clip: ET.Element
    project_video_clip: ET.Element
    audio_source: ET.Element
    video_source: ET.Element
    audio_master_track: ET.Element
    video_tracks: list[ET.Element]
    audio_tracks: list[ET.Element]
    primary_video_track: ET.Element
    primary_audio_track: ET.Element


class IdAllocator:
    def __init__(self, root: ET.Element) -> None:
        existing_ids = [
            int(elem.attrib["ObjectID"])
            for elem in root
            if "ObjectID" in elem.attrib and elem.attrib["ObjectID"].isdigit()
        ]
        self.next_id = max(existing_ids, default=0) + 1

    def object_id(self) -> int:
        value = self.next_id
        self.next_id += 1
        return value

    @staticmethod
    def guid() -> str:
        return str(uuid.uuid4())


def normalize_timecode(timecode: str) -> str:
    parts = timecode.split(":")
    if len(parts) == 4:
        return timecode
    if len(parts) == 3:
        return f"00:{timecode}"
    raise PodcastEditorError(f"Unsupported timecode format: {timecode}")


def extract_timecode_ranges(text: str) -> list[TimecodeRange]:
    ranges = extract_timecode_ranges_or_empty(text)
    if not ranges:
        raise PodcastEditorError("No timecode ranges found in the GPT output.")
    return ranges


def extract_timecode_ranges_or_empty(text: str) -> list[TimecodeRange]:
    seen: set[tuple[str, str]] = set()
    ranges: list[TimecodeRange] = []
    for match in TIMECODE_PATTERN.finditer(text):
        pair = (normalize_timecode(match.group(1)), normalize_timecode(match.group(2)))
        if pair in seen:
            continue
        seen.add(pair)
        ranges.append(TimecodeRange(*pair))
    return ranges


def nominal_fps(frame_ticks: int) -> int:
    ratio = Fraction(TICKS_PER_SECOND, frame_ticks)
    rounded = round(float(ratio))
    if abs(float(ratio) - rounded) > 0.1:
        raise PodcastEditorError(
            f"Unsupported non-integer nominal frame rate derived from frame ticks {frame_ticks}: {float(ratio)}"
        )
    return rounded


def timecode_to_ticks(timecode: str, frame_ticks: int) -> int:
    timecode = normalize_timecode(timecode)
    hours, minutes, seconds, frames = (int(part) for part in timecode.split(":"))
    fps = nominal_fps(frame_ticks)
    if frames >= fps:
        raise PodcastEditorError(
            f"Invalid timecode {timecode}: frame number {frames} is >= sequence fps {fps}."
        )
    total_frames = (((hours * 60) + minutes) * 60 + seconds) * fps + frames
    return total_frames * frame_ticks


def normalize_ranges(ranges: Iterable[TickRange], duration_ticks: int) -> list[TickRange]:
    cleaned: list[TickRange] = []
    for time_range in ranges:
        start = max(0, min(duration_ticks, time_range.start))
        end = max(0, min(duration_ticks, time_range.end))
        if end <= start:
            continue
        cleaned.append(TickRange(start, end))

    unique_cleaned = sorted(set(cleaned), key=lambda item: (item.start, item.end))
    if not unique_cleaned:
        raise PodcastEditorError("All extracted timecode ranges were empty after clipping to the sequence duration.")
    return unique_cleaned


def merge_touching_ranges(ranges: Iterable[TickRange]) -> list[TickRange]:
    ordered = sorted(ranges, key=lambda item: (item.start, item.end))
    if not ordered:
        return []

    merged = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start <= previous.end:
            merged[-1] = TickRange(previous.start, max(previous.end, current.end))
            continue
        merged.append(current)
    return merged


def invert_ranges(ranges: list[TickRange], duration_ticks: int) -> list[TickRange]:
    merged_ranges = merge_touching_ranges(ranges)
    return [
        TickRange(segment.start, segment.end)
        for segment in build_segments(merged_ranges, duration_ticks)
        if not segment.selected
    ]


def build_segments(selected_ranges: list[TickRange], duration_ticks: int) -> list[Segment]:
    segments: list[Segment] = []
    cursor = 0
    for selected in selected_ranges:
        if selected.start > cursor:
            segments.append(Segment(cursor, selected.start, False))
        segments.append(Segment(selected.start, selected.end, True))
        cursor = selected.end
    if cursor < duration_ticks:
        segments.append(Segment(cursor, duration_ticks, False))
    return [segment for segment in segments if segment.end > segment.start]


def load_prproj(path: Path) -> ET.Element:
    with gzip.open(path, "rb") as compressed_file:
        data = compressed_file.read()
    return ET.fromstring(data)


def save_prproj(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    buffer = io.BytesIO()
    tree.write(buffer, encoding="UTF-8", xml_declaration=True)
    with gzip.open(path, "wb") as compressed_file:
        compressed_file.write(buffer.getvalue())


def build_object_maps(root: ET.Element) -> tuple[dict[int, ET.Element], dict[str, ET.Element]]:
    id_map: dict[int, ET.Element] = {}
    uid_map: dict[str, ET.Element] = {}
    for elem in root:
        object_id = elem.attrib.get("ObjectID")
        object_uid = elem.attrib.get("ObjectUID")
        if object_id and object_id.isdigit():
            id_map[int(object_id)] = elem
        if object_uid:
            uid_map[object_uid] = elem
    return id_map, uid_map


def project_root_bin(root: ET.Element) -> ET.Element:
    root_bin = next((elem for elem in root if elem.tag == "RootProjectItem"), None)
    if root_bin is None:
        raise PodcastEditorError("Project is missing the root bin.")
    return root_bin


def collect_object_ref_closure(id_map: dict[int, ET.Element], roots: Iterable[ET.Element]) -> list[ET.Element]:
    result: list[ET.Element] = []
    seen_ids: set[int] = set()
    seen_identities: set[int] = set()
    queue: list[ET.Element] = list(roots)

    while queue:
        elem = queue.pop(0)
        if id(elem) in seen_identities:
            continue
        seen_identities.add(id(elem))
        object_id = elem.attrib.get("ObjectID")
        if object_id and object_id.isdigit():
            numeric_id = int(object_id)
            if numeric_id not in seen_ids:
                seen_ids.add(numeric_id)
                result.append(elem)
        for node in elem.iter():
            object_ref = node.attrib.get("ObjectRef")
            if object_ref and object_ref.isdigit():
                referenced = id_map.get(int(object_ref))
                if referenced is not None:
                    queue.append(referenced)
    return result


def collect_object_closure(
    id_map: dict[int, ET.Element],
    uid_map: dict[str, ET.Element],
    roots: Iterable[ET.Element],
) -> list[ET.Element]:
    result: list[ET.Element] = []
    seen_ids: set[int] = set()
    seen_identities: set[int] = set()
    queue: list[ET.Element] = list(roots)

    while queue:
        elem = queue.pop(0)
        identity = id(elem)
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        object_id = elem.attrib.get("ObjectID")
        if object_id and object_id.isdigit():
            numeric_id = int(object_id)
            if numeric_id not in seen_ids:
                seen_ids.add(numeric_id)
                result.append(elem)
        elif elem.attrib.get("ObjectUID"):
            result.append(elem)

        for node in elem.iter():
            object_ref = node.attrib.get("ObjectRef")
            if object_ref and object_ref.isdigit():
                referenced = id_map.get(int(object_ref))
                if referenced is not None:
                    queue.append(referenced)
            object_uref = node.attrib.get("ObjectURef")
            if object_uref:
                referenced_uid = uid_map.get(object_uref)
                if referenced_uid is not None:
                    queue.append(referenced_uid)
    return result


def unique_root_children(root: ET.Element, elements: Iterable[ET.Element]) -> list[ET.Element]:
    wanted = {id(elem) for elem in elements}
    return [elem for elem in root if id(elem) in wanted]


def unique_elements(elements: Iterable[ET.Element]) -> list[ET.Element]:
    ordered: list[ET.Element] = []
    seen_identities: set[int] = set()
    for elem in elements:
        identity = id(elem)
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        ordered.append(elem)
    return ordered


def remap_clone_refs(
    elem: ET.Element,
    old_to_new_ids: dict[int, int],
    old_to_new_uids: dict[str, str],
) -> None:
    for node in elem.iter():
        object_ref = node.attrib.get("ObjectRef")
        if object_ref and object_ref.isdigit():
            mapped_id = old_to_new_ids.get(int(object_ref))
            if mapped_id is not None:
                node.attrib["ObjectRef"] = str(mapped_id)
        object_uref = node.attrib.get("ObjectURef")
        if object_uref:
            mapped_uid = old_to_new_uids.get(object_uref)
            if mapped_uid is not None:
                node.attrib["ObjectURef"] = mapped_uid


def clone_root_objects(
    root: ET.Element,
    allocator: IdAllocator,
    elements: Iterable[ET.Element],
) -> tuple[list[ET.Element], dict[int, int], dict[str, str], dict[int, ET.Element]]:
    ordered = unique_root_children(root, elements)
    old_to_new_ids: dict[int, int] = {}
    old_to_new_uids: dict[str, str] = {}
    clone_by_old_identity: dict[int, ET.Element] = {}
    clones: list[ET.Element] = []

    for old in ordered:
        clone = copy.deepcopy(old)
        object_id = old.attrib.get("ObjectID")
        if object_id and object_id.isdigit():
            new_id = allocator.object_id()
            old_to_new_ids[int(object_id)] = new_id
            clone.attrib["ObjectID"] = str(new_id)
        object_uid = old.attrib.get("ObjectUID")
        if object_uid:
            new_uid = allocator.guid()
            old_to_new_uids[object_uid] = new_uid
            clone.attrib["ObjectUID"] = new_uid
        clone_by_old_identity[id(old)] = clone
        clones.append(clone)

    for clone in clones:
        remap_clone_refs(clone, old_to_new_ids, old_to_new_uids)
    return clones, old_to_new_ids, old_to_new_uids, clone_by_old_identity


def clone_detached_objects(
    allocator: IdAllocator,
    elements: Iterable[ET.Element],
) -> tuple[list[ET.Element], dict[int, int], dict[str, str], dict[int, ET.Element]]:
    ordered = unique_elements(elements)
    old_to_new_ids: dict[int, int] = {}
    old_to_new_uids: dict[str, str] = {}
    clone_by_old_identity: dict[int, ET.Element] = {}
    clones: list[ET.Element] = []

    for old in ordered:
        clone = copy.deepcopy(old)
        object_id = old.attrib.get("ObjectID")
        if object_id and object_id.isdigit():
            new_id = allocator.object_id()
            old_to_new_ids[int(object_id)] = new_id
            clone.attrib["ObjectID"] = str(new_id)
        object_uid = old.attrib.get("ObjectUID")
        if object_uid:
            new_uid = allocator.guid()
            old_to_new_uids[object_uid] = new_uid
            clone.attrib["ObjectUID"] = new_uid
        clone_by_old_identity[id(old)] = clone
        clones.append(clone)

    for clone in clones:
        remap_clone_refs(clone, old_to_new_ids, old_to_new_uids)
    return clones, old_to_new_ids, old_to_new_uids, clone_by_old_identity


def require_text(parent: ET.Element, tag: str) -> str:
    child = parent.find(tag)
    if child is None or child.text is None:
        raise PodcastEditorError(f"Missing required tag <{tag}>.")
    return child.text


def get_object_ref(parent: ET.Element, path: str) -> int:
    child = parent.find(path)
    if child is None:
        raise PodcastEditorError(f"Missing required object ref at {path}.")
    object_ref = child.attrib.get("ObjectRef")
    if not object_ref or not object_ref.isdigit():
        raise PodcastEditorError(f"Invalid ObjectRef at {path}.")
    return int(object_ref)


def get_object_uref(parent: ET.Element, path: str) -> str:
    child = parent.find(path)
    if child is None:
        raise PodcastEditorError(f"Missing required object uref at {path}.")
    object_uref = child.attrib.get("ObjectURef")
    if not object_uref:
        raise PodcastEditorError(f"Invalid ObjectURef at {path}.")
    return object_uref


def track_item_refs(track: ET.Element) -> list[int]:
    refs: list[int] = []
    track_items = track.find("./ClipTrack/ClipItems/TrackItems")
    if track_items is None:
        return refs
    for item in track_items.findall("TrackItem"):
        object_ref = item.attrib.get("ObjectRef")
        if object_ref and object_ref.isdigit():
            refs.append(int(object_ref))
    return refs


def track_item_timeline_bounds(track_item_element: ET.Element) -> tuple[int, int]:
    track_item = track_item_element.find("./ClipTrackItem/TrackItem")
    if track_item is None:
        raise PodcastEditorError("Track item template is missing its TrackItem node.")
    start = int(track_item.findtext("Start", "0"))
    end = int(track_item.findtext("End", "0"))
    if end <= start:
        raise PodcastEditorError(f"Track item has an invalid span: start={start} end={end}")
    return start, end


def clip_source_bounds(clip_element: ET.Element, timeline_duration: int) -> tuple[int, int]:
    clip = clip_element.find("Clip")
    if clip is None:
        raise PodcastEditorError("Clip template is missing its <Clip> child.")
    source_start = int(clip.findtext("InPoint", "0"))
    source_end = int(clip.findtext("OutPoint", str(source_start + timeline_duration)))
    if source_end <= source_start:
        source_end = source_start + timeline_duration
    return source_start, source_end


def secondary_content_templates(clip_element: ET.Element, id_map: dict[int, ET.Element]) -> tuple[ET.Element, ...]:
    secondary_contents = clip_element.find("SecondaryContents")
    if secondary_contents is None:
        return ()

    templates: list[ET.Element] = []
    for item in secondary_contents.findall("SecondaryContentItem"):
        object_ref = item.attrib.get("ObjectRef")
        if object_ref and object_ref.isdigit():
            templates.append(id_map[int(object_ref)])
    return tuple(templates)


def secondary_content_object_refs(clip_element: ET.Element) -> list[int]:
    refs: list[int] = []
    secondary_contents = clip_element.find("SecondaryContents")
    if secondary_contents is None:
        return refs
    for item in secondary_contents.findall("SecondaryContentItem"):
        object_ref = item.attrib.get("ObjectRef")
        if object_ref and object_ref.isdigit():
            refs.append(int(object_ref))
    return refs


def build_source_track_item_template(
    media_kind: str,
    track_index: int,
    track: ET.Element,
    track_item_object_id: int,
    id_map: dict[int, ET.Element],
) -> SourceTrackItemTemplate:
    track_item = id_map[track_item_object_id]
    component = id_map[get_object_ref(track_item, "./ClipTrackItem/ComponentOwner/Components")]
    subclip = id_map[get_object_ref(track_item, "./ClipTrackItem/SubClip")]
    clip = id_map[get_object_ref(subclip, "Clip")]
    timeline_start, timeline_end = track_item_timeline_bounds(track_item)
    source_start, source_end = clip_source_bounds(clip, timeline_end - timeline_start)
    secondary_templates = secondary_content_templates(clip, id_map) if media_kind == "audio" else ()
    return SourceTrackItemTemplate(
        original_track_item_id=track_item_object_id,
        media_kind=media_kind,
        track_index=track_index,
        track=track,
        track_item=track_item,
        component=component,
        subclip=subclip,
        clip=clip,
        timeline_start=timeline_start,
        timeline_end=timeline_end,
        source_start=source_start,
        source_end=source_end,
        secondary_templates=secondary_templates,
    )


def build_source_track_item_templates(
    media_kind: str,
    tracks: list[ET.Element],
    id_map: dict[int, ET.Element],
) -> list[SourceTrackItemTemplate]:
    templates: list[SourceTrackItemTemplate] = []
    for track_index, track in enumerate(tracks):
        for track_item_object_id in track_item_refs(track):
            templates.append(
                build_source_track_item_template(
                    media_kind,
                    track_index,
                    track,
                    track_item_object_id,
                    id_map,
                )
            )
    return templates


def build_sequence_link_templates(sequence: ET.Element, id_map: dict[int, ET.Element]) -> list[SequenceLinkTemplate]:
    links: list[SequenceLinkTemplate] = []
    for link_ref in sequence.findall("./PersistentGroupContainer/LinkContainer/Links/Link"):
        object_ref = link_ref.attrib.get("ObjectRef")
        if not object_ref or not object_ref.isdigit():
            continue
        link = id_map[int(object_ref)]
        track_item_ids: list[int] = []
        for item in link.findall("./TrackItemGroup/TrackItems/TrackItem"):
            item_ref = item.attrib.get("ObjectRef")
            if item_ref and item_ref.isdigit():
                track_item_ids.append(int(item_ref))
        if track_item_ids:
            links.append(
                SequenceLinkTemplate(
                    original_link_id=int(object_ref),
                    track_item_ids=tuple(track_item_ids),
                )
            )
    return links


def clear_track_items(track: ET.Element) -> None:
    clip_items = track.find("./ClipTrack/ClipItems")
    if clip_items is None:
        raise PodcastEditorError("Clip track is missing a ClipItems container.")
    track_items = clip_items.find("TrackItems")
    if track_items is not None:
        clip_items.remove(track_items)


def replace_track_items(track: ET.Element, object_refs: list[int]) -> None:
    clip_items = track.find("./ClipTrack/ClipItems")
    if clip_items is None:
        raise PodcastEditorError("Clip track is missing a ClipItems container.")
    clear_track_items(track)
    if not object_refs:
        return
    track_items = ET.Element("TrackItems", {"Version": "1"})
    for index, object_ref in enumerate(object_refs):
        ET.SubElement(track_items, "TrackItem", {"Index": str(index), "ObjectRef": str(object_ref)})
    clip_items.insert(0, track_items)


def replace_links(sequence: ET.Element, link_refs: list[int]) -> None:
    link_container = sequence.find("./PersistentGroupContainer/LinkContainer")
    if link_container is None:
        raise PodcastEditorError("Sequence is missing a link container.")
    links = link_container.find("Links")
    if links is None:
        if not link_refs:
            return
        links = ET.SubElement(link_container, "Links", {"Version": "1"})
    for child in list(links):
        links.remove(child)
    for index, object_ref in enumerate(link_refs):
        ET.SubElement(links, "Link", {"Index": str(index), "ObjectRef": str(object_ref)})


def set_track_item_times(track_item: ET.Element, start: int, end: int) -> None:
    if end <= start:
        raise PodcastEditorError(f"Invalid track item span: start={start} end={end}")
    for child in list(track_item):
        if child.tag in {"Start", "End"}:
            track_item.remove(child)
    insert_at = 1 if track_item.find("Node") is not None else 0
    if start > 0:
        start_elem = ET.Element("Start")
        start_elem.text = str(start)
        track_item.insert(insert_at, start_elem)
        insert_at += 1
    end_elem = ET.Element("End")
    end_elem.text = str(end)
    track_item.insert(insert_at, end_elem)


def replace_or_insert_after(parent: ET.Element, tag: str, text: str, after_tags: tuple[str, ...]) -> None:
    existing = parent.find(tag)
    if existing is not None:
        existing.text = text
        return

    new_elem = ET.Element(tag)
    new_elem.text = text
    insert_at = len(parent)
    for index, child in enumerate(list(parent)):
        if child.tag in after_tags:
            insert_at = index + 1
    parent.insert(insert_at, new_elem)


def remove_child(parent: ET.Element, tag: str) -> None:
    child = parent.find(tag)
    if child is not None:
        parent.remove(child)


def update_video_clip(clip_element: ET.Element, start: int, end: int, clip_id: str) -> None:
    clip = clip_element.find("Clip")
    if clip is None:
        raise PodcastEditorError("Video clip template is missing its <Clip> child.")
    replace_or_insert_after(clip, "InPoint", str(start), ("Source",))
    replace_or_insert_after(clip, "OutPoint", str(end), ("InPoint",))
    replace_or_insert_after(clip, "ClipID", clip_id, ("OutPoint",))
    remove_child(clip, "InUse")


def update_audio_clip(
    clip_element: ET.Element,
    start: int,
    end: int,
    clip_id: str,
    secondary_content_ids: list[int],
) -> None:
    clip = clip_element.find("Clip")
    if clip is None:
        raise PodcastEditorError("Audio clip template is missing its <Clip> child.")
    replace_or_insert_after(clip, "InPoint", str(start), ("Source",))
    replace_or_insert_after(clip, "OutPoint", str(end), ("InPoint",))
    replace_or_insert_after(clip, "ClipID", clip_id, ("OutPoint",))
    remove_child(clip, "InUse")

    secondary_contents = clip_element.find("SecondaryContents")
    if secondary_contents is not None:
        for child in list(secondary_contents):
            secondary_contents.remove(child)
        for index, object_id in enumerate(secondary_content_ids):
            ET.SubElement(
                secondary_contents,
                "SecondaryContentItem",
                {"Index": str(index), "ObjectRef": str(object_id)},
            )


def update_subclip(subclip: ET.Element, clip_object_id: int) -> None:
    clip_ref = subclip.find("Clip")
    if clip_ref is None:
        raise PodcastEditorError("Subclip template is missing its clip ref.")
    clip_ref.attrib["ObjectRef"] = str(clip_object_id)
    master_clip_ref = subclip.find("MasterClip")
    if master_clip_ref is not None and not master_clip_ref.attrib.get("ObjectURef"):
        raise PodcastEditorError("Subclip template is missing its master clip reference.")


def update_subclip_metadata(
    subclip: ET.Element,
    clip_object_id: int,
    master_clip_uid: str,
    name: str,
) -> None:
    update_subclip(subclip, clip_object_id)
    master_clip_ref = subclip.find("MasterClip")
    if master_clip_ref is None:
        raise PodcastEditorError("Subclip template is missing its master clip reference.")
    master_clip_ref.attrib["ObjectURef"] = master_clip_uid
    set_optional_child_text(subclip, "Name", name)


def update_component_ref(track_item_element: ET.Element, component_object_id: int) -> None:
    component_ref = track_item_element.find("./ClipTrackItem/ComponentOwner/Components")
    if component_ref is None:
        raise PodcastEditorError("Track item template is missing a component reference.")
    component_ref.attrib["ObjectRef"] = str(component_object_id)


def update_subclip_ref(track_item_element: ET.Element, subclip_object_id: int) -> None:
    subclip_ref = track_item_element.find("./ClipTrackItem/SubClip")
    if subclip_ref is None:
        raise PodcastEditorError("Track item template is missing a subclip reference.")
    subclip_ref.attrib["ObjectRef"] = str(subclip_object_id)


def update_link(link: ET.Element, track_item_ids: list[int]) -> None:
    track_items = link.find("./TrackItemGroup/TrackItems")
    if track_items is None:
        raise PodcastEditorError("Link template is missing its TrackItems collection.")
    for child in list(track_items):
        track_items.remove(child)
    for index, track_item_id in enumerate(track_item_ids):
        ET.SubElement(track_items, "TrackItem", {"Index": str(index), "ObjectRef": str(track_item_id)})


def find_sequence_project_item_bundle(
    root: ET.Element,
    sequence: ET.Element,
    id_map: dict[int, ET.Element],
    uid_map: dict[str, ET.Element],
) -> tuple[ET.Element, ET.Element, ET.Element, ET.Element, ET.Element, ET.Element, ET.Element, ET.Element, list[ET.Element]]:
    sequence_uid = sequence.attrib.get("ObjectUID")
    if not sequence_uid:
        raise PodcastEditorError("Sequence is missing an ObjectUID.")

    for project_item in [elem for elem in root if elem.tag == "ClipProjectItem"]:
        master_clip_uid = get_object_uref(project_item, "MasterClip")
        master_clip = uid_map.get(master_clip_uid)
        if master_clip is None:
            continue

        audio_clip = None
        video_clip = None
        audio_source = None
        video_source = None

        for clip_ref in master_clip.findall("./Clips/Clip"):
            object_ref = clip_ref.attrib.get("ObjectRef")
            if not object_ref or not object_ref.isdigit():
                continue
            clip = id_map[int(object_ref)]
            try:
                source = id_map[get_object_ref(clip, "./Clip/Source")]
            except PodcastEditorError:
                continue
            try:
                source_sequence_uid = get_object_uref(source, "./SequenceSource/Sequence")
            except PodcastEditorError:
                continue
            if source.tag == "AudioSequenceSource" and source_sequence_uid == sequence_uid:
                audio_clip = clip
                audio_source = source
            elif source.tag == "VideoSequenceSource" and source_sequence_uid == sequence_uid:
                video_clip = clip
                video_source = source

        if audio_clip is None or video_clip is None or audio_source is None or video_source is None:
            continue

        logging_info = id_map[get_object_ref(master_clip, "LoggingInfo")]
        master_audio_chain = id_map[get_object_ref(master_clip, "./AudioComponentChains/AudioComponentChain")]
        audio_channel_group = id_map[get_object_ref(master_clip, "AudioClipChannelGroups")]
        audio_secondary_templates: list[ET.Element] = []
        secondary_contents = audio_clip.find("SecondaryContents")
        if secondary_contents is not None:
            for item in secondary_contents.findall("SecondaryContentItem"):
                object_ref = item.attrib.get("ObjectRef")
                if object_ref and object_ref.isdigit():
                    audio_secondary_templates.append(id_map[int(object_ref)])

        return (
            project_item,
            master_clip,
            logging_info,
            master_audio_chain,
            audio_clip,
            video_clip,
            audio_source,
            video_source,
            [audio_channel_group, *audio_secondary_templates],
        )

    raise PodcastEditorError("Could not locate the project-panel item for the selected sequence.")


def build_template_context(root: ET.Element, sequence_name: str | None) -> TemplateContext:
    id_map, uid_map = build_object_maps(root)
    root_bin = next((elem for elem in root if elem.tag == "RootProjectItem"), None)
    if root_bin is None:
        raise PodcastEditorError("Project is missing the root bin.")

    sequences = [elem for elem in root if elem.tag == "Sequence"]
    if not sequences:
        raise PodcastEditorError("Project does not contain any sequences.")

    if sequence_name:
        candidates = [sequence for sequence in sequences if sequence.findtext("Name") == sequence_name]
        if not candidates:
            raise PodcastEditorError(f"Sequence '{sequence_name}' was not found in the project.")
        sequence = candidates[0]
    else:
        sequence = choose_default_sequence(sequences)
        if sequence is None:
            names = ", ".join(sequence.findtext("Name", "<unnamed>") for sequence in sequences)
            raise PodcastEditorError(
                f"Project contains multiple sequences; choose one with --sequence. Available: {names}"
            )

    group_refs = [get_object_ref(track_group, "Second") for track_group in sequence.findall("./TrackGroups/TrackGroup")]
    groups = [id_map[ref] for ref in group_refs]
    video_group = next((group for group in groups if group.tag == "VideoTrackGroup"), None)
    audio_group = next((group for group in groups if group.tag == "AudioTrackGroup"), None)
    data_group = next((group for group in groups if group.tag == "DataTrackGroup"), None)
    if video_group is None or audio_group is None:
        raise PodcastEditorError("Sequence is missing video or audio track groups.")

    video_track_urefs = [track_ref.attrib["ObjectURef"] for track_ref in video_group.findall("./TrackGroup/Tracks/Track")]
    audio_track_urefs = [track_ref.attrib["ObjectURef"] for track_ref in audio_group.findall("./TrackGroup/Tracks/Track")]
    video_tracks = [uid_map[uref] for uref in video_track_urefs]
    audio_tracks = [uid_map[uref] for uref in audio_track_urefs]

    source_video_items = build_source_track_item_templates("video", video_tracks, id_map)
    source_audio_items = build_source_track_item_templates("audio", audio_tracks, id_map)
    if not source_video_items or not source_audio_items:
        raise PodcastEditorError("Sequence must contain at least one populated video track and one populated audio track.")

    primary_video_track = next((track for track in video_tracks if track_item_refs(track)), video_tracks[0])
    primary_audio_track = next((track for track in audio_tracks if track_item_refs(track)), audio_tracks[0])
    primary_video_item = source_video_items[0]
    primary_audio_item = source_audio_items[0]
    video_track_item = primary_video_item.track_item
    audio_track_item = primary_audio_item.track_item
    video_component = primary_video_item.component
    audio_component = primary_audio_item.component
    video_subclip = primary_video_item.subclip
    audio_subclip = primary_audio_item.subclip
    video_clip = primary_video_item.clip
    audio_clip = primary_audio_item.clip
    audio_secondary_templates = list(primary_audio_item.secondary_templates)
    link_groups = build_sequence_link_templates(sequence, id_map)

    (
        sequence_project_item,
        sequence_master_clip,
        sequence_logging_info,
        sequence_master_audio_chain,
        sequence_audio_clip,
        sequence_video_clip,
        sequence_audio_source,
        sequence_video_source,
        sequence_bundle_extras,
    ) = find_sequence_project_item_bundle(root, sequence, id_map, uid_map)

    video_group_component_template = id_map[get_object_ref(video_group, "./ComponentOwner/Components")]
    audio_master_track = id_map[get_object_ref(audio_group, "MasterTrack")]

    duration_ticks = max(item.timeline_end for item in [*source_video_items, *source_audio_items])
    frame_ticks = int(require_text(video_group.find("./TrackGroup"), "FrameRate"))  # type: ignore[arg-type]
    fps = nominal_fps(frame_ticks)

    link_template = None
    first_link = sequence.find("./PersistentGroupContainer/LinkContainer/Links/Link")
    if first_link is not None:
        link_ref = first_link.attrib.get("ObjectRef")
        if link_ref and link_ref.isdigit():
            link_template = id_map[int(link_ref)]

    return TemplateContext(
        root_bin=root_bin,
        sequence_project_item=sequence_project_item,
        sequence_master_clip=sequence_master_clip,
        sequence_logging_info=sequence_logging_info,
        sequence_master_audio_chain=sequence_master_audio_chain,
        sequence_audio_clip=sequence_audio_clip,
        sequence_video_clip=sequence_video_clip,
        sequence_audio_source=sequence_audio_source,
        sequence_video_source=sequence_video_source,
        sequence_audio_channel_group=sequence_bundle_extras[0],
        sequence_audio_secondary_templates=sequence_bundle_extras[1:],
        sequence=sequence,
        video_group=video_group,
        audio_group=audio_group,
        data_group=data_group,
        video_group_component_template=video_group_component_template,
        audio_master_track=audio_master_track,
        video_tracks=video_tracks,
        audio_tracks=audio_tracks,
        primary_video_track=primary_video_track,
        primary_audio_track=primary_audio_track,
        video_track_item_template=video_track_item,
        audio_track_item_template=audio_track_item,
        video_component_template=video_component,
        audio_component_template=audio_component,
        video_subclip_template=video_subclip,
        audio_subclip_template=audio_subclip,
        video_clip_template=video_clip,
        audio_clip_template=audio_clip,
        audio_secondary_templates=audio_secondary_templates,
        link_template=link_template,
        duration_ticks=duration_ticks,
        frame_ticks=frame_ticks,
        fps=fps,
        source_video_items=source_video_items,
        source_audio_items=source_audio_items,
        link_groups=link_groups,
    )

def make_link_template() -> ET.Element:
    link = ET.Element(
        "Link",
        {"ObjectID": "0", "ClassID": "149d4ea5-a7d4-4b34-9bb7-16d783904bf2", "Version": "1"},
    )
    track_item_group = ET.SubElement(link, "TrackItemGroup", {"Version": "1"})
    ET.SubElement(track_item_group, "TrackItems", {"Version": "1"})
    return link


def build_sequence_shell_clone_elements(
    root: ET.Element,
    context: TemplateContext,
    id_map: dict[int, ET.Element],
) -> list[ET.Element]:
    support_roots = [context.video_group_component_template]

    for audio_track in context.audio_tracks:
        support_roots.append(id_map[get_object_ref(audio_track, "./AudioTrack/ComponentOwner/Components")])
        support_roots.append(id_map[get_object_ref(audio_track, "./AudioTrack/Panner")])

    support_roots.append(id_map[get_object_ref(context.audio_master_track, "./AudioTrack/ComponentOwner/Components")])
    support_roots.append(id_map[get_object_ref(context.audio_master_track, "./AudioTrack/Panner")])
    support_roots.append(id_map[get_object_ref(context.audio_master_track, "Inlet")])

    elements: list[ET.Element] = [
        context.sequence,
        context.video_group,
        context.audio_group,
        *context.video_tracks,
        *context.audio_tracks,
        context.audio_master_track,
    ]
    if context.data_group is not None:
        elements.append(context.data_group)
    elements.extend(collect_object_ref_closure(id_map, support_roots))
    return unique_root_children(root, elements)


def build_sequence_project_item_clone_elements(
    root: ET.Element,
    context: TemplateContext,
    id_map: dict[int, ET.Element],
) -> list[ET.Element]:
    elements = [context.sequence_project_item, context.sequence_master_clip]
    elements.extend(collect_object_ref_closure(id_map, [context.sequence_master_clip]))
    return unique_root_children(root, elements)


def unique_sequence_name(root: ET.Element, base_name: str) -> str:
    existing = {elem.findtext("Name") for elem in root if elem.tag == "Sequence"}
    if base_name not in existing:
        return base_name
    index = 2
    while f"{base_name} {index}" in existing:
        index += 1
    return f"{base_name} {index}"


def set_optional_child_text(parent: ET.Element, tag: str, text: str) -> None:
    child = parent.find(tag)
    if child is not None:
        child.text = text


def project_item_grid_order(root_bin: ET.Element, uid_map: dict[str, ET.Element]) -> int:
    items = root_bin.find("./ProjectItemContainer/Items")
    if items is None:
        return 0
    max_order = -1
    for item in items.findall("Item"):
        item_uid = item.attrib.get("ObjectURef")
        if not item_uid:
            continue
        project_item = uid_map.get(item_uid)
        if project_item is None:
            continue
        order_text = project_item.findtext("./ProjectItem/Node/Properties/project.icon.view.grid.order")
        if order_text is not None and order_text.isdigit():
            max_order = max(max_order, int(order_text))
    return max_order + 1


def append_item_to_root_bin(root_bin: ET.Element, object_uid: str) -> None:
    items = root_bin.find("./ProjectItemContainer/Items")
    if items is None:
        container = root_bin.find("ProjectItemContainer")
        if container is None:
            raise PodcastEditorError("Root bin is missing its project item container.")
        items = ET.SubElement(container, "Items", {"Version": "1"})
    next_index = len(items.findall("Item"))
    ET.SubElement(items, "Item", {"Index": str(next_index), "ObjectURef": object_uid})


def append_root_bin_coordinate(root_bin: ET.Element, object_uid: str) -> None:
    properties = root_bin.find("./ProjectItem/Node/Properties")
    if properties is None:
        return
    coord_elem = properties.find("project.freeform.view.bin.coordinate")
    if coord_elem is None or not coord_elem.text:
        return
    try:
        payload = json.loads(coord_elem.text)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict) or not payload:
        return
    key = next(iter(payload))
    entries = payload.get(key)
    if not isinstance(entries, list):
        return
    y_values = [
        item[1][1]
        for item in entries
        if isinstance(item, list)
        and len(item) >= 2
        and isinstance(item[1], list)
        and len(item[1]) >= 2
        and isinstance(item[1][1], (int, float))
    ]
    next_y = int(max(y_values, default=10) + 168)
    entries.append([object_uid, [10, next_y, 1]])
    coord_elem.text = json.dumps(payload, separators=(",", ":"))


def increment_sequence_index(root: ET.Element) -> None:
    project = next((elem for elem in root if elem.tag == "Project" and elem.attrib.get("ObjectID")), None)
    if project is None:
        return
    properties = project.find("./Node/Properties")
    if properties is None:
        return
    next_index = properties.find("MZ.NextSequenceIndex")
    if next_index is None or next_index.text is None or not next_index.text.isdigit():
        return
    next_index.text = str(int(next_index.text) + 1)


def refresh_audio_track_ids(audio_tracks: list[ET.Element], audio_master_track: ET.Element) -> None:
    for track in [*audio_tracks, audio_master_track]:
        audio_track = track.find("AudioTrack")
        if audio_track is not None:
            set_optional_child_text(audio_track, "ID", str(uuid.uuid4()))


def sequence_source_duration(context: TemplateContext, timeline_duration: int) -> int:
    original_duration_text = context.sequence_audio_source.findtext("OriginalDuration")
    if original_duration_text and original_duration_text.isdigit() and context.duration_ticks > 0:
        scaled = round(int(original_duration_text) * timeline_duration / context.duration_ticks)
        return max(1, scaled)
    return max(1, timeline_duration)


def build_cloned_sequence_context(
    clone_by_old_identity: dict[int, ET.Element],
    context: TemplateContext,
) -> ClonedSequenceContext:
    sequence = clone_by_old_identity[id(context.sequence)]
    project_item = clone_by_old_identity[id(context.sequence_project_item)]
    master_clip = clone_by_old_identity[id(context.sequence_master_clip)]
    project_audio_clip = clone_by_old_identity[id(context.sequence_audio_clip)]
    project_video_clip = clone_by_old_identity[id(context.sequence_video_clip)]
    audio_source = clone_by_old_identity[id(context.sequence_audio_source)]
    video_source = clone_by_old_identity[id(context.sequence_video_source)]
    video_tracks = [clone_by_old_identity[id(track)] for track in context.video_tracks]
    audio_tracks = [clone_by_old_identity[id(track)] for track in context.audio_tracks]
    return ClonedSequenceContext(
        sequence=sequence,
        project_item=project_item,
        master_clip=master_clip,
        video_group=clone_by_old_identity[id(context.video_group)],
        audio_group=clone_by_old_identity[id(context.audio_group)],
        project_audio_clip=project_audio_clip,
        project_video_clip=project_video_clip,
        audio_source=audio_source,
        video_source=video_source,
        audio_master_track=clone_by_old_identity[id(context.audio_master_track)],
        video_tracks=video_tracks,
        audio_tracks=audio_tracks,
        primary_video_track=clone_by_old_identity[id(context.primary_video_track)],
        primary_audio_track=clone_by_old_identity[id(context.primary_audio_track)],
    )


def update_cloned_sequence_metadata(
    root: ET.Element,
    context: TemplateContext,
    cloned: ClonedSequenceContext,
    sequence_name: str,
    timeline_duration: int,
) -> None:
    cloned.sequence.find("Name").text = sequence_name  # type: ignore[union-attr]
    cloned.project_item.find("./ProjectItem/Name").text = sequence_name  # type: ignore[union-attr]
    cloned.master_clip.find("Name").text = sequence_name  # type: ignore[union-attr]

    set_optional_child_text(cloned.project_audio_clip.find("Clip"), "ClipID", str(uuid.uuid4()))  # type: ignore[arg-type]
    set_optional_child_text(cloned.project_video_clip.find("Clip"), "ClipID", str(uuid.uuid4()))  # type: ignore[arg-type]

    duration_text = str(sequence_source_duration(context, timeline_duration))
    set_optional_child_text(cloned.audio_source, "OriginalDuration", duration_text)
    set_optional_child_text(cloned.video_source, "OriginalDuration", duration_text)

    sequence_properties = cloned.sequence.find("./Node/Properties")
    if sequence_properties is not None:
        set_optional_child_text(sequence_properties, "MZ.WorkInPoint", "0")
        set_optional_child_text(sequence_properties, "MZ.EditLine", "0")
        set_optional_child_text(sequence_properties, "Monitor.ProgramZoomOut", str(max(timeline_duration, context.frame_ticks)))
        set_optional_child_text(sequence_properties, "MZ.WorkOutPoint", str(min(timeline_duration, 60 * TICKS_PER_SECOND)))

    refresh_audio_track_ids(cloned.audio_tracks, cloned.audio_master_track)

    uid_map = build_object_maps(root)[1]
    order = project_item_grid_order(context.root_bin, uid_map)
    project_item_properties = cloned.project_item.find("./ProjectItem/Node/Properties")
    if project_item_properties is not None:
        set_optional_child_text(project_item_properties, "project.icon.view.grid.order", str(order))


def clear_cloned_sequence_shell(cloned: ClonedSequenceContext) -> None:
    for track in [*cloned.video_tracks, *cloned.audio_tracks]:
        clear_track_items(track)
    replace_links(cloned.sequence, [])


def next_track_numeric_id(tracks: list[ET.Element]) -> int:
    existing_ids = [
        int(track.findtext("./ClipTrack/Track/ID", "0"))
        for track in tracks
        if track.findtext("./ClipTrack/Track/ID", "0").isdigit()
    ]
    return max(existing_ids, default=0) + 1


def set_track_identity(track: ET.Element, track_index: int, track_id: int) -> None:
    track_body = track.find("./ClipTrack/Track")
    if track_body is None:
        raise PodcastEditorError("Track template is missing its <Track> body.")

    set_optional_child_text(track_body, "ID", str(track_id))
    set_optional_child_text(track_body, "Index", str(track_index))

    clip_items = track.find("./ClipTrack/ClipItems")
    if clip_items is not None:
        set_optional_child_text(clip_items, "Index", str(track_index))

    transition_items = track.find("./ClipTrack/TransitionItems")
    if transition_items is not None:
        set_optional_child_text(transition_items, "Index", str(track_index))

    properties = track.find("./ClipTrack/Track/Node/Properties")
    if properties is not None:
        set_optional_child_text(properties, "MZ.TrackTargeted", "0")
        set_optional_child_text(properties, "MZ.SourceTrackState", "0")
        set_optional_child_text(properties, "MZ.SourceTrackNumber", "-1")

    audio_track = track.find("AudioTrack")
    if audio_track is not None:
        set_optional_child_text(audio_track, "ID", str(uuid.uuid4()))


def append_track_reference(track_group: ET.Element, track_uid: str) -> None:
    tracks = track_group.find("./TrackGroup/Tracks")
    if tracks is None:
        raise PodcastEditorError("Track group is missing its Tracks collection.")
    next_index = len(tracks.findall("Track"))
    ET.SubElement(tracks, "Track", {"Index": str(next_index), "ObjectURef": track_uid})


def refresh_track_group_next_track_id(track_group: ET.Element, tracks: list[ET.Element]) -> None:
    set_optional_child_text(track_group.find("TrackGroup"), "NextTrackID", str(next_track_numeric_id(tracks)))  # type: ignore[arg-type]


def clone_extra_video_track(
    root: ET.Element,
    allocator: IdAllocator,
    template_track: ET.Element,
    track_index: int,
    track_id: int,
) -> ET.Element:
    cloned_objects, _, _, clone_by_old_identity = clone_root_objects(root, allocator, [template_track])
    for cloned_object in cloned_objects:
        root.append(cloned_object)

    cloned_track = clone_by_old_identity[id(template_track)]
    clear_track_items(cloned_track)
    set_track_identity(cloned_track, track_index, track_id)
    return cloned_track


def clone_extra_audio_track(
    root: ET.Element,
    allocator: IdAllocator,
    template_track: ET.Element,
    id_map: dict[int, ET.Element],
    track_index: int,
    track_id: int,
) -> ET.Element:
    support_roots = [
        id_map[get_object_ref(template_track, "./AudioTrack/ComponentOwner/Components")],
        id_map[get_object_ref(template_track, "./AudioTrack/Panner")],
    ]
    elements = [template_track, *collect_object_ref_closure(id_map, support_roots)]
    cloned_objects, _, _, clone_by_old_identity = clone_root_objects(root, allocator, elements)
    for cloned_object in cloned_objects:
        root.append(cloned_object)

    cloned_track = clone_by_old_identity[id(template_track)]
    clear_track_items(cloned_track)
    set_track_identity(cloned_track, track_index, track_id)
    return cloned_track


def active_track_count_for_range(
    items: list[SourceTrackItemTemplate],
    range_start: int,
    range_end: int,
) -> int:
    return len(
        {
            item.track_index
            for item in items
            if item.overlap(range_start, range_end) is not None
        }
    )


def ensure_video_track_capacity(
    root: ET.Element,
    allocator: IdAllocator,
    context: TemplateContext,
    cloned: ClonedSequenceContext,
    target_index: int,
) -> None:
    next_video_id = next_track_numeric_id(cloned.video_tracks)
    while len(cloned.video_tracks) <= target_index:
        template_track = context.video_tracks[min(len(context.video_tracks) - 1, len(cloned.video_tracks) - 1)]
        cloned_track = clone_extra_video_track(
            root,
            allocator,
            template_track,
            len(cloned.video_tracks),
            next_video_id,
        )
        next_video_id += 1
        cloned.video_tracks.append(cloned_track)
        append_track_reference(cloned.video_group, cloned_track.attrib["ObjectUID"])
    refresh_track_group_next_track_id(cloned.video_group, cloned.video_tracks)


def ensure_audio_track_capacity(
    root: ET.Element,
    allocator: IdAllocator,
    context: TemplateContext,
    cloned: ClonedSequenceContext,
    target_index: int,
) -> None:
    id_map, _ = build_object_maps(root)
    next_audio_id = next_track_numeric_id(cloned.audio_tracks)
    while len(cloned.audio_tracks) <= target_index:
        template_track = context.audio_tracks[min(len(context.audio_tracks) - 1, len(cloned.audio_tracks) - 1)]
        cloned_track = clone_extra_audio_track(
            root,
            allocator,
            template_track,
            id_map,
            len(cloned.audio_tracks),
            next_audio_id,
        )
        next_audio_id += 1
        cloned.audio_tracks.append(cloned_track)
        append_track_reference(cloned.audio_group, cloned_track.attrib["ObjectUID"])
    refresh_track_group_next_track_id(cloned.audio_group, cloned.audio_tracks)


def create_track_item_objects(
    template: SourceTrackItemTemplate,
    source_start: int,
    source_end: int,
    timeline_start: int,
    timeline_end: int,
    allocator: IdAllocator,
) -> tuple[list[ET.Element], int]:
    objects: list[ET.Element] = []

    component = copy.deepcopy(template.component)
    component_id = allocator.object_id()
    component.attrib["ObjectID"] = str(component_id)
    objects.append(component)

    secondary_content_ids: list[int] = []
    for secondary_template in template.secondary_templates:
        secondary_content = copy.deepcopy(secondary_template)
        secondary_id = allocator.object_id()
        secondary_content.attrib["ObjectID"] = str(secondary_id)
        secondary_content_ids.append(secondary_id)
        objects.append(secondary_content)

    clip = copy.deepcopy(template.clip)
    clip_id = allocator.object_id()
    clip.attrib["ObjectID"] = str(clip_id)
    if template.media_kind == "video":
        update_video_clip(clip, source_start, source_end, allocator.guid())
    else:
        update_audio_clip(clip, source_start, source_end, allocator.guid(), secondary_content_ids)
    objects.append(clip)

    subclip = copy.deepcopy(template.subclip)
    subclip_id = allocator.object_id()
    subclip.attrib["ObjectID"] = str(subclip_id)
    update_subclip(subclip, clip_id)
    objects.append(subclip)

    track_item = copy.deepcopy(template.track_item)
    track_item_id = allocator.object_id()
    track_item.attrib["ObjectID"] = str(track_item_id)
    update_component_ref(track_item, component_id)
    update_subclip_ref(track_item, subclip_id)
    track_item_body = track_item.find("./ClipTrackItem/TrackItem")
    if track_item_body is None:
        raise PodcastEditorError(f"{template.media_kind.title()} track item template is missing its TrackItem node.")
    set_track_item_times(track_item_body, timeline_start, timeline_end)
    if template.media_kind == "audio":
        audio_id = track_item.find("ID")
        if audio_id is not None:
            audio_id.text = allocator.guid()
    objects.append(track_item)

    return objects, track_item_id


def create_link_object(
    track_item_ids: list[int],
    allocator: IdAllocator,
    link_template: ET.Element | None,
) -> tuple[ET.Element, int]:
    link = copy.deepcopy(link_template) if link_template is not None else make_link_template()
    link_id = allocator.object_id()
    link.attrib["ObjectID"] = str(link_id)
    update_link(link, track_item_ids)
    return link, link_id


def prepare_cloned_sequence(
    root: ET.Element,
    context: TemplateContext,
    sequence_name: str,
    timeline_duration: int,
) -> tuple[IdAllocator, ClonedSequenceContext]:
    allocator = IdAllocator(root)
    id_map, _ = build_object_maps(root)

    shell_elements = build_sequence_shell_clone_elements(root, context, id_map)
    project_item_elements = build_sequence_project_item_clone_elements(root, context, id_map)
    cloned_objects, _, _, clone_by_old_identity = clone_root_objects(
        root,
        allocator,
        [*shell_elements, *project_item_elements],
    )

    cloned = build_cloned_sequence_context(clone_by_old_identity, context)
    clear_cloned_sequence_shell(cloned)
    update_cloned_sequence_metadata(root, context, cloned, sequence_name, timeline_duration)

    for cloned_object in cloned_objects:
        root.append(cloned_object)

    append_item_to_root_bin(context.root_bin, cloned.project_item.attrib["ObjectUID"])
    append_root_bin_coordinate(context.root_bin, cloned.project_item.attrib["ObjectUID"])
    increment_sequence_index(root)
    return allocator, cloned


def create_selects_sequence(root: ET.Element, context: TemplateContext, ranges: list[TickRange]) -> dict[str, int | str]:
    sequence_name = unique_sequence_name(root, f"{context.sequence.findtext('Name', 'Sequence')} - Selects")
    total_duration = sum(time_range.end - time_range.start for time_range in ranges)
    allocator, cloned = prepare_cloned_sequence(root, context, sequence_name, total_duration)

    video_track_refs: list[list[int]] = [[] for _ in cloned.video_tracks]
    audio_track_refs: list[list[int]] = [[] for _ in cloned.audio_tracks]
    link_refs: list[int] = []
    appended_clip_objects: list[ET.Element] = []
    source_items = [*context.source_video_items, *context.source_audio_items]

    cursor = 0
    for time_range in ranges:
        range_track_items: dict[int, int] = {}
        for source_item in source_items:
            overlap = source_item.overlap(time_range.start, time_range.end)
            if overlap is None:
                continue
            overlap_start, overlap_end = overlap
            source_start, source_end = source_item.source_bounds_for_overlap(overlap_start, overlap_end)
            timeline_start = cursor + (overlap_start - time_range.start)
            timeline_end = cursor + (overlap_end - time_range.start)
            objects, new_track_item_id = create_track_item_objects(
                source_item,
                source_start,
                source_end,
                timeline_start,
                timeline_end,
                allocator,
            )
            appended_clip_objects.extend(objects)
            range_track_items[source_item.original_track_item_id] = new_track_item_id
            if source_item.media_kind == "video":
                video_track_refs[source_item.track_index].append(new_track_item_id)
            else:
                audio_track_refs[source_item.track_index].append(new_track_item_id)

        for link_group in context.link_groups:
            new_link_items = [
                range_track_items[track_item_id]
                for track_item_id in link_group.track_item_ids
                if track_item_id in range_track_items
            ]
            if len(new_link_items) < 2:
                continue
            link_object, link_id = create_link_object(new_link_items, allocator, context.link_template)
            appended_clip_objects.append(link_object)
            link_refs.append(link_id)

        cursor += time_range.end - time_range.start

    for track, track_refs in zip(cloned.video_tracks, video_track_refs):
        replace_track_items(track, track_refs)
    for track, track_refs in zip(cloned.audio_tracks, audio_track_refs):
        replace_track_items(track, track_refs)
    replace_links(cloned.sequence, link_refs)

    for obj in appended_clip_objects:
        root.append(obj)

    return {
        "new_sequence": sequence_name,
        "selected_ranges": len(ranges),
        "video_segments": sum(len(track_refs) for track_refs in video_track_refs),
        "audio_segments": sum(len(track_refs) for track_refs in audio_track_refs),
        "links": len(link_refs),
        "assembled_duration_ticks": total_duration,
    }


def create_concision_sequence(
    root: ET.Element,
    context: TemplateContext,
    removal_ranges: list[TickRange],
) -> dict[str, int | str]:
    merged_removals = merge_touching_ranges(removal_ranges)
    keep_ranges = invert_ranges(merged_removals, context.duration_ticks)
    sequence_name = unique_sequence_name(root, f"{context.sequence.findtext('Name', 'Sequence')} - Concision")
    allocator, cloned = prepare_cloned_sequence(root, context, sequence_name, context.duration_ticks)

    video_track_refs: list[list[int]] = [[] for _ in cloned.video_tracks]
    audio_track_refs: list[list[int]] = [[] for _ in cloned.audio_tracks]
    link_refs: list[int] = []
    appended_clip_objects: list[ET.Element] = []
    source_items = [*context.source_video_items, *context.source_audio_items]

    for keep_range in keep_ranges:
        range_track_items: dict[int, int] = {}
        for source_item in source_items:
            overlap = source_item.overlap(keep_range.start, keep_range.end)
            if overlap is None:
                continue
            overlap_start, overlap_end = overlap
            source_start, source_end = source_item.source_bounds_for_overlap(overlap_start, overlap_end)
            objects, new_track_item_id = create_track_item_objects(
                source_item,
                source_start,
                source_end,
                overlap_start,
                overlap_end,
                allocator,
            )
            appended_clip_objects.extend(objects)
            range_track_items[source_item.original_track_item_id] = new_track_item_id
            if source_item.media_kind == "video":
                video_track_refs[source_item.track_index].append(new_track_item_id)
            else:
                audio_track_refs[source_item.track_index].append(new_track_item_id)

        for link_group in context.link_groups:
            new_link_items = [
                range_track_items[track_item_id]
                for track_item_id in link_group.track_item_ids
                if track_item_id in range_track_items
            ]
            if len(new_link_items) < 2:
                continue
            link_object, link_id = create_link_object(new_link_items, allocator, context.link_template)
            appended_clip_objects.append(link_object)
            link_refs.append(link_id)

    for removal_range in merged_removals:
        active_video_track_count = active_track_count_for_range(
            context.source_video_items,
            removal_range.start,
            removal_range.end,
        )
        active_audio_track_count = active_track_count_for_range(
            context.source_audio_items,
            removal_range.start,
            removal_range.end,
        )
        range_track_items: dict[int, int] = {}
        for source_item in source_items:
            overlap = source_item.overlap(removal_range.start, removal_range.end)
            if overlap is None:
                continue
            overlap_start, overlap_end = overlap
            source_start, source_end = source_item.source_bounds_for_overlap(overlap_start, overlap_end)
            objects, new_track_item_id = create_track_item_objects(
                source_item,
                source_start,
                source_end,
                overlap_start,
                overlap_end,
                allocator,
            )
            appended_clip_objects.extend(objects)
            range_track_items[source_item.original_track_item_id] = new_track_item_id
            if source_item.media_kind == "video":
                target_track_index = source_item.track_index + active_video_track_count
                ensure_video_track_capacity(root, allocator, context, cloned, target_track_index)
                while len(video_track_refs) < len(cloned.video_tracks):
                    video_track_refs.append([])
                video_track_refs[target_track_index].append(new_track_item_id)
            else:
                target_track_index = source_item.track_index + active_audio_track_count
                ensure_audio_track_capacity(root, allocator, context, cloned, target_track_index)
                while len(audio_track_refs) < len(cloned.audio_tracks):
                    audio_track_refs.append([])
                audio_track_refs[target_track_index].append(new_track_item_id)

        for link_group in context.link_groups:
            new_link_items = [
                range_track_items[track_item_id]
                for track_item_id in link_group.track_item_ids
                if track_item_id in range_track_items
            ]
            if len(new_link_items) < 2:
                continue
            link_object, link_id = create_link_object(new_link_items, allocator, context.link_template)
            appended_clip_objects.append(link_object)
            link_refs.append(link_id)

    for track, track_refs in zip(cloned.video_tracks, video_track_refs):
        replace_track_items(track, track_refs)
    for track, track_refs in zip(cloned.audio_tracks, audio_track_refs):
        replace_track_items(track, track_refs)
    replace_links(cloned.sequence, link_refs)

    for obj in appended_clip_objects:
        root.append(obj)

    return {
        "new_sequence": sequence_name,
        "selected_ranges": len(merged_removals),
        "video_segments": sum(len(track_refs) for track_refs in video_track_refs),
        "audio_segments": sum(len(track_refs) for track_refs in audio_track_refs),
        "links": len(link_refs),
        "assembled_duration_ticks": context.duration_ticks,
        "lifted_duration_ticks": sum(time_range.end - time_range.start for time_range in merged_removals),
    }


def latest_autosave(project_path: Path) -> Path | None:
    autosave_dir = project_path.parent / "Adobe Premiere Pro Auto-Save"
    if not autosave_dir.exists():
        return None
    candidates = sorted(
        autosave_dir.glob("*.prproj"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def resolve_source_project(project_path: Path, sequence_name: str | None) -> tuple[Path, ET.Element, TemplateContext, bool]:
    if not project_path.exists():
        autosave_path = latest_autosave(project_path)
        if autosave_path is None:
            raise PodcastEditorError(f"Project file was not found: {project_path}")
        autosave_root = load_prproj(autosave_path)
        autosave_context = build_template_context(autosave_root, sequence_name)
        return autosave_path, autosave_root, autosave_context, True

    root = load_prproj(project_path)
    try:
        context = build_template_context(root, sequence_name)
        return project_path, root, context, False
    except PodcastEditorError as primary_error:
        autosave_path = latest_autosave(project_path)
        if autosave_path is None:
            raise primary_error
        autosave_root = load_prproj(autosave_path)
        autosave_context = build_template_context(autosave_root, sequence_name)
        return autosave_path, autosave_root, autosave_context, True


def default_output_path(project_path: Path) -> Path:
    return project_path


def default_transcript_dir(output_project: Path) -> Path:
    return output_project.parent / f"{output_project.stem}.transcript"


def resolve_working_project(
    project_path: Path,
    output_path: Path,
    sequence_name: str | None,
) -> tuple[Path, ET.Element, TemplateContext, bool]:
    if output_path.exists():
        try:
            root = load_prproj(output_path)
            context = build_template_context(root, sequence_name)
            return output_path, root, context, False
        except PodcastEditorError:
            pass
    return resolve_source_project(project_path, sequence_name)


def parse_key_value_lines(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            values[key] = value
    return values


def build_transcription_command(
    project_path: Path,
    sequence_name: str,
    output_dir: Path,
    whisper_model: str,
    whisper_language: str,
    whisper_conda_env: str | None,
    whisper_python: Path | None,
    speaker_config: SpeakerDiarizationConfig,
) -> list[str]:
    script_path = Path(__file__).with_name("transcribe_sequence.py")
    if whisper_python is not None:
        command = [str(whisper_python)]
    elif whisper_conda_env:
        conda = shutil.which("conda")
        if conda is None:
            raise PodcastEditorError("`conda` is required for transcription but was not found on PATH.")
        command = [conda, "run", "-n", whisper_conda_env, "python"]
    else:
        command = [sys.executable]

    command.extend(
        [
            str(script_path),
            "--project",
            str(project_path),
            "--sequence",
            sequence_name,
            "--output-dir",
            str(output_dir),
            "--whisper-model",
            whisper_model,
            "--language",
            whisper_language,
        ]
    )
    append_speaker_diarization_command_args(command, speaker_config)
    return command


def transcribe_selects_sequence(
    project_path: Path,
    sequence_name: str,
    output_dir: Path,
    whisper_model: str,
    whisper_language: str,
    whisper_conda_env: str | None,
    whisper_python: Path | None,
    speaker_config: SpeakerDiarizationConfig,
) -> dict[str, str]:
    command = build_transcription_command(
        project_path,
        sequence_name,
        output_dir,
        whisper_model,
        whisper_language,
        whisper_conda_env,
        whisper_python,
        speaker_config,
    )
    emit_status(f"Launching transcription helper for selects sequence '{sequence_name}'")
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise PodcastEditorError(f"Selects-sequence transcription failed with exit code {exc.returncode}.") from exc

    summary = parse_key_value_lines(completed.stdout or "")
    if not summary:
        raise PodcastEditorError("Transcription helper completed but did not return a summary.")
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse GPT podcast highlights and assemble the selected Premiere segments into a new sequence."
    )
    parser.add_argument("--project", required=True, type=Path, help="Path to the Premiere .prproj file.")
    parser.add_argument(
        "--notes-file",
        required=True,
        type=Path,
        help="Path to the GPT output text file containing the timecode ranges.",
    )
    parser.add_argument(
        "--output-project",
        type=Path,
        help="Where to write the updated .prproj. Defaults to the original project path.",
    )
    parser.add_argument(
        "--sequence",
        help=(
            "Optional sequence name. If omitted, the editor uses the only sequence in the project or the only "
            "sequence whose name does not look like a generated selects sequence."
        ),
    )
    parser.add_argument(
        "--transcribe-selects",
        action="store_true",
        help="After creating the selects sequence, render its audio and transcribe it with the Whisper helper.",
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        help="Directory for the rendered audio and transcript artifacts. Defaults to <output-project-stem>.transcript/.",
    )
    parser.add_argument(
        "--whisper-model",
        default="base.en",
        help="Whisper model name for the selects transcription step.",
    )
    parser.add_argument(
        "--whisper-language",
        default="en",
        help="Language hint for the selects transcription step.",
    )
    parser.add_argument(
        "--whisper-conda-env",
        default="agent",
        help="Conda environment used to run transcribe_sequence.py. Set to an empty string to use the current Python.",
    )
    parser.add_argument(
        "--whisper-python",
        type=Path,
        help="Optional Python executable for the selects transcription step. Overrides --whisper-conda-env.",
    )
    add_speaker_diarization_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        speaker_config = speaker_diarization_config_from_args(args)
    except ValueError as exc:
        raise PodcastEditorError(str(exc)) from exc

    output_path = args.output_project or default_output_path(args.project)
    emit_status(f"Opening Premiere project {args.project}")
    source_project, root, context, used_autosave = resolve_working_project(args.project, output_path, args.sequence)
    if used_autosave:
        emit_status(f"Using autosave fallback {source_project}")
    else:
        emit_status(f"Using project file {source_project}")

    emit_status(f"Reading timecode notes from {args.notes_file}")
    gpt_text = read_text(args.notes_file)
    emit_status("Parsing timecode ranges")
    timecode_ranges = extract_timecode_ranges(gpt_text)
    tick_ranges = normalize_ranges(
        [
            TickRange(
                timecode_to_ticks(time_range.start, context.frame_ticks),
                timecode_to_ticks(time_range.end, context.frame_ticks),
            )
            for time_range in timecode_ranges
        ],
        context.duration_ticks,
    )
    emit_status("Building selects sequence")
    summary = create_selects_sequence(root, context, tick_ranges)
    emit_status(f"Saving updated project to {output_path}")
    save_prproj(root, output_path)

    transcript_summary: dict[str, str] | None = None
    if args.transcribe_selects:
        transcript_dir = args.transcript_dir or default_transcript_dir(output_path)
        whisper_conda_env = args.whisper_conda_env or None
        emit_status("Starting selects-sequence transcription")
        transcript_summary = transcribe_selects_sequence(
            output_path,
            str(summary["new_sequence"]),
            transcript_dir,
            args.whisper_model,
            args.whisper_language,
            whisper_conda_env,
            args.whisper_python,
            speaker_config,
        )
    emit_status("Finished")

    print(f"source_project={source_project}")
    print(f"used_autosave_fallback={'true' if used_autosave else 'false'}")
    print(f"source_sequence={context.sequence.findtext('Name', '<unnamed>')}")
    print(f"new_sequence={summary['new_sequence']}")
    print(f"fps={context.fps}")
    print(f"extracted_ranges={len(timecode_ranges)}")
    print(f"normalized_ranges={summary['selected_ranges']}")
    print(f"segments_v1={summary['video_segments']}")
    print(f"segments_a1={summary['audio_segments']}")
    print(f"links={summary['links']}")
    print(f"assembled_duration_ticks={summary['assembled_duration_ticks']}")
    print(f"output_project={output_path}")
    if transcript_summary is not None:
        print(f"transcript_sequence={transcript_summary.get('sequence', summary['new_sequence'])}")
        print(f"transcript_segments={transcript_summary.get('segments', '')}")
        print(f"rendered_audio={transcript_summary.get('rendered_audio', '')}")
        print(f"segment_manifest={transcript_summary.get('segment_manifest', '')}")
        if "transcript_json" in transcript_summary:
            print(f"transcript_json={transcript_summary['transcript_json']}")
        if "base_transcript_json" in transcript_summary:
            print(f"base_transcript_json={transcript_summary['base_transcript_json']}")
        if "speaker_blocks_txt" in transcript_summary:
            print(f"speaker_blocks_txt={transcript_summary['speaker_blocks_txt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
