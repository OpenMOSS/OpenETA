"""Exercise the actual queue renderer without a browser or UI dependencies."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest


def render_queue(requests):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to execute the console renderer")
    source = Path(__file__).resolve().parents[1] / "tools/manual_vlm_console.js"
    script = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = fs.readFileSync(input.source, 'utf8');
const prefix = source.slice(0, source.indexOf('\nfunction renderSection('));
const queue = {innerHTML: ''};
vm.runInNewContext(prefix + '\nstate.requests = input.requests; renderQueue();', {
  input, localStorage: {getItem: () => null},
  document: {querySelector: () => queue, querySelectorAll: () => []}
});
process.stdout.write(queue.innerHTML);
"""
    result = subprocess.run(
        [node, "-e", script], input=json.dumps({"source": str(source), "requests": requests}),
        text=True, capture_output=True, timeout=5, check=True,
    )
    return result.stdout


def request(id, session, *, parent="", status="pending", wait_reason=""):
    return {
        "id": id, "session_id": session, "parent_session_id": parent,
        "status": status, "wait_reason": wait_reason, "session_turn": 1,
        "request_label": "Grasp advisor", "message_count": 2, "image_count": 0,
    }


def test_pending_children_render_in_parent_group_with_explicit_wait():
    rendered = render_queue([
        request("child", "advisor-one", parent="main", wait_reason="等待人工 advisor 响应"),
        request("main-request", "main", status="responded"),
        request("legacy", "inferred-old", wait_reason="等待人工 advisor 响应"),
    ])
    assert rendered.count('<details class="session"') == 2
    assert "待人工响应：2 个请求（含 1 个关联子请求）" in rendered
    assert 'title="main"' in rendered
    assert 'title="advisor-one"' in rendered
    assert "↳ 子请求" in rendered
    assert rendered.count("等待人工 advisor 响应") == 2


def test_renderer_keeps_parents_separate_and_escapes_untrusted_labels():
    rendered = render_queue([
        request("one", "advisor-one", parent="main-one", wait_reason="<script>bad</script>"),
        request("two", "advisor-two", parent="main-two"),
    ])
    assert rendered.count('<details class="session"') == 2
    assert "<script>bad</script>" not in rendered
    assert "&lt;script&gt;bad&lt;/script&gt;" in rendered


def test_completed_queue_does_not_show_pending_banner():
    rendered = render_queue([request("done", "advisor-one", parent="main", status="responded")])
    assert "待人工响应" not in rendered
