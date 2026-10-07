"""证据门禁：Agent 结论里的每条判断都要能追溯到工具结果、预检索证据或代码位置。

由代码而不是模型来检查：
- 引用格式：[E3] 指第 3 条预检索证据，[T:call_x] 指 tool_use_id 为 call_x 的成功工具调用，
  `path/to/file.py:42` 指代码位置（路径必须在本轮工具参数、证据元数据或观察结果里出现过）。
- 判断句没有有效引用 -> 标为"推测·待人工确认"，并计入无依据结论比例。
- 引用了不存在的编号（模型编的）也算无依据，单独记录 invalid_citations。
- 自己声明为推测/建议/下一步的句子、问句、标题、代码块不计入判断。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

UNSUPPORTED_TAG = "〔推测·待人工确认〕"

CITATION_RE = re.compile(r"\[(E\d+|T:[A-Za-z0-9_.:\-]+)\]")
LEADING_CITATION_RE = re.compile(r"\s*\[(?:E\d+|T:[A-Za-z0-9_.:\-]+)\]")
CODE_LOCATION_RE = re.compile(r"`?((?:[\w.\-]+/)*[\w\-]+\.[A-Za-z][A-Za-z0-9]{0,7}):(\d+)(?:-\d+)?`?")
PATH_TOKEN_RE = re.compile(r"(?:[\w.\-]+/)*[\w\-]+\.[A-Za-z][A-Za-z0-9]{0,7}")
TERMINATOR_RE = re.compile(r"[。！？!?；;]|\.(?=\s|$)")
HEADING_RE = re.compile(r"^\s*(#{1,6}\s+.+|\*\*[^*]{1,40}\*\*[:：]?\s*|[^。！？!?]{1,24}[:：])\s*$")
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)、])\s+")

HEDGE_MARKERS = (
    "推测", "推断", "可能", "也许", "或许", "大概", "估计", "不确定", "待确认", "需要确认", "无法确认",
    "建议", "下一步", "可以考虑", "如果", "假设",
    "maybe", "might", "possibly", "likely", "suggest", "recommend", "if ",
)
NON_CLAIM_SECTION_MARKERS = (
    "下一步", "建议", "不确定", "待确认", "推测", "推断", "open question", "next action", "next step", "tradeoff", "取舍",
)


@dataclass
class EvidenceRegistry:
    entries: list[dict[str, str]] = field(default_factory=list)
    ids: set[str] = field(default_factory=set)
    paths: set[str] = field(default_factory=set)

    def add(self, evidence_id: str, label: str) -> None:
        if evidence_id in self.ids:
            return
        self.ids.add(evidence_id)
        self.entries.append({"id": evidence_id, "label": label})

    def add_path(self, value: Any) -> None:
        for token in PATH_TOKEN_RE.findall(str(value or "")):
            self.paths.add(token.strip("./").lower())

    def has_location(self, path: str) -> bool:
        target = path.strip("`./").lower()
        return any(target == known or known.endswith("/" + target) or target.endswith("/" + known) for known in self.paths)

    def prompt_block(self, limit: int = 40) -> str:
        if not self.entries:
            return "（本轮还没有可引用的证据）"
        return "\n".join(f"[{item['id']}] {item['label']}" for item in self.entries[:limit])


def build_evidence_registry(state: dict[str, Any]) -> EvidenceRegistry:
    registry = EvidenceRegistry()
    context = state.get("context") or {}
    evidence = context.get("evidence") if isinstance(context.get("evidence"), list) else []
    for index, item in enumerate(evidence, start=1):
        if not isinstance(item, dict):
            continue
        registry.add(f"E{index}", f"{item.get('source_type') or 'evidence'} · {str(item.get('title') or '未命名证据')[:80]}")
        metadata = item.get("metadata") or {}
        registry.add_path(metadata.get("path"))
        registry.add_path(item.get("title"))

    for call in state.get("tool_calls") or []:
        if not isinstance(call, dict) or call.get("status") != "success":
            continue
        tool_use_id = str(call.get("tool_use_id") or "").strip()
        if not tool_use_id:
            continue
        arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
        brief = ", ".join(f"{key}={str(value)[:40]}" for key, value in list(arguments.items())[:3])
        registry.add(f"T:{tool_use_id}", f"{call.get('tool_name') or 'tool'}({brief})")
        registry.add_path(arguments.get("path"))

    for item in state.get("observations") or []:
        if not isinstance(item, dict):
            continue
        observation = item.get("observation") if isinstance(item.get("observation"), dict) else {}
        if observation.get("route") == "tool_error":
            continue
        for match in CODE_LOCATION_RE.finditer(str(observation.get("answer_preview") or "")):
            registry.add_path(match.group(1))

    for citation in state.get("citations") or []:
        if isinstance(citation, dict):
            registry.add_path(citation.get("id"))
            registry.add_path(citation.get("title"))
    return registry


def _sentence_spans(line: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    length = len(line)
    while start < length:
        match = TERMINATOR_RE.search(line, start)
        end = match.end() if match else length
        while True:
            trailing = LEADING_CITATION_RE.match(line, end)
            if not trailing:
                break
            end = trailing.end()
        spans.append((start, end))
        start = end
    return spans


def _core_text(segment: str) -> str:
    text = LIST_PREFIX_RE.sub("", segment)
    text = CITATION_RE.sub("", text)
    text = text.replace("**", "").replace("`", "").strip(" |\t-：:")
    return text.strip()


def _classify(segment: str, registry: EvidenceRegistry, in_non_claim_section: bool) -> tuple[str, list[str]]:
    core = _core_text(segment)
    if len(core) < 6 or core.endswith(("？", "?")):
        return "skip", []
    cited_ids = CITATION_RE.findall(segment)
    invalid = [item for item in cited_ids if item not in registry.ids]
    valid_ids = [item for item in cited_ids if item in registry.ids]
    valid_locations = [match.group(1) for match in CODE_LOCATION_RE.finditer(segment) if registry.has_location(match.group(1))]
    if valid_ids or valid_locations:
        return "supported", invalid
    lowered = core.lower()
    if in_non_claim_section or any(marker in lowered for marker in HEDGE_MARKERS) or UNSUPPORTED_TAG in segment:
        return "hedged", invalid
    return "unsupported", invalid


def apply_evidence_gate(answer: str, registry: EvidenceRegistry, *, annotate: bool = True) -> tuple[str, dict[str, Any]]:
    counts = {"supported": 0, "unsupported": 0, "hedged": 0}
    invalid_citations: list[str] = []
    unsupported_samples: list[str] = []
    output_lines: list[str] = []
    in_code = False
    in_non_claim_section = False
    previous_was_table = False

    for line in (answer or "").split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            output_lines.append(line)
            continue
        if in_code or not stripped:
            output_lines.append(line)
            previous_was_table = False
            continue
        if HEADING_RE.match(stripped) and not CITATION_RE.search(stripped):
            heading = stripped.lower()
            in_non_claim_section = any(marker in heading for marker in NON_CLAIM_SECTION_MARKERS)
            output_lines.append(line)
            previous_was_table = False
            continue
        if stripped.startswith("|"):
            is_separator = set(stripped.replace("|", "").strip()) <= set("-: ")
            is_header = not previous_was_table
            previous_was_table = True
            if is_separator or is_header:
                output_lines.append(line)
                continue
            spans = [(0, len(line))]
        else:
            previous_was_table = False
            spans = _sentence_spans(line)

        pieces: list[str] = []
        for start, end in spans:
            segment = line[start:end]
            kind, invalid = _classify(segment, registry, in_non_claim_section)
            invalid_citations.extend(invalid)
            if kind != "skip":
                counts[kind] += 1
            if kind == "unsupported":
                if len(unsupported_samples) < 5:
                    unsupported_samples.append(_core_text(segment)[:120])
                if annotate:
                    body = segment.rstrip()
                    if body.endswith("|"):
                        body = body[:-1].rstrip() + f" {UNSUPPORTED_TAG} |"
                    else:
                        body = f"{body}{UNSUPPORTED_TAG}"
                    segment = body + segment[len(segment.rstrip()):]
            pieces.append(segment)
        output_lines.append("".join(pieces))

    judged = counts["supported"] + counts["unsupported"]
    report = {
        "claims": judged,
        "supported": counts["supported"],
        "unsupported": counts["unsupported"],
        "hedged": counts["hedged"],
        "unsupported_ratio": round(counts["unsupported"] / judged, 4) if judged else 0.0,
        "invalid_citations": sorted(set(invalid_citations)),
        "unsupported_samples": unsupported_samples,
        "evidence_count": len(registry.entries),
        "status": "passed" if counts["unsupported"] == 0 and not invalid_citations else "needs_review",
        "annotated": annotate,
    }
    return "\n".join(output_lines), report


EVIDENCE_RULE = (
    "证据引用规则（代码会逐句校验）：每条事实或判断句末尾必须标注来源，"
    "预检索证据用 [E编号]，工具结果用 [T:tool_use_id]，代码位置写成 `路径:行号`；"
    "只能使用下方『可引用证据编号』里列出的编号，禁止编造编号。"
    "没有证据支撑的内容要明确写成推测、建议或待确认，否则会被自动标记为待人工确认。"
)
