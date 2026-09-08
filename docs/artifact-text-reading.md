# Artifact 文本搜索、续读与投影边界

状态：2026-09-06 个人分支候选，覆盖 I27-03/04/05 的证据链路局部修复。新增返回 schema、游标与 context projection 元数据需三人评审；不是已批准的新 `ls/read/grep` 工具集合，也未完成整个 issue #27。

## 现有 Python artifact API

保持 session-owned roots、路径校验和 Python worker 沙箱不变。新增 `artifacts.read_text_page`，`artifacts.grep_text` 增加可选 `cursor`。旧 `read_text` 的字符串返回不改，但仍只有前缀，续读请用新方法。

```python
# path 必须使用本 session 已返回的实际 artifact 路径。
result = artifacts.grep_text(path, "needle", max_matches=5)
result = artifacts.read_text_page(path, max_chars=1500)
# 后续调用复制对应响应的 next_cursor；grep 的 read_cursor 可直接定位到命中处。
result = artifacts.read_text_page(path, max_chars=1500, cursor=previous_cursor)
```

以上示例的 path/previous_cursor 是解释性变量，不会自动跨独立 python_exec 调用保存。Agent 应传入上一结果中的实际值。没有扩大文件访问范围；从别的会话取得游标不能绕过根目录校验。

文本页返回 `openeta.text_artifact_page.v1`，包含 `path`、`text`、`char_offset`、`line`、`column`、`chars_returned`、`offset_unit`、`truncated`、`next_cursor`。`max_chars` 为 1–1,000,000 的整数，默认 1500；bool/非法边界拒绝。

- 字符单位是 UTF-8 解码后 Unicode code point，不是字节或 UTF-16 code unit；非法 UTF-8 用 replacement character。
- 采用 universal newline：CRLF/CR 归一为 LF。`line` 从 1 起，`column` 和 `char_offset` 从 0 起；无需行边界，能从一条超长 JSONL 的中间继续读取。
- `next_cursor=null` 表示这次实际读取到 EOF；非空游标指向下一字符。恰好等于页长的文件不是截断。
- 游标包含版本和字符偏移；版本绑定规范路径、文件 metadata 与全文 SHA-256。实际测试发现同长度快速改写可有相同时间戳，因此不单独依赖 mtime。文件改变、替换、不同文件、非法偏移均拒绝；读取前后再次检查变化。
- 这是只读一致性检测，不是对恶意并发写入者的原子快照或执行凭证。每页为校验内容扫描文件两次，并扫描前缀定位字符；内存按块使用，但 I/O 成本与文件长度/偏移有关。后续可研究不可变 artifact 版本索引，不能用 metadata-only 缓存悄悄弱化当前一致性。

## grep 的数量、片段与续页

返回 `openeta.text_artifact_search.v1`。保持 `match_count` 为本页**匹配行数**，不是正则出现次数；一行返回第一个匹配。`max_matches` 为 1–200 的整数，默认 20，建议模型闭环使用较小页数。

- 片段围绕实际匹配截取，最多 500 个字符；包含 `snippet_start_column`、`match_start_column`、`match_end_column`，区间为零基、右开。
- `snippet_clipped` 表示行上下文被裁剪；正则匹配本身超过片段时另报 `match_clipped`。不能把整个匹配超长时的有限前缀冒充完整匹配。
- 用第 N+1 个匹配判断 `truncated`；恰好 N 个匹配且后面无命中时为 false。首次查询且未截断才能给出全文 `total_match_count`，其余为 null，不猜总数。
- `next_cursor` 指向首个未返回匹配行，绑定同一个 pattern/ignore_case 与文件版本；改查询必须重新开始。每个命中的 `read_cursor` 可传给 `read_text_page`，直接读命中处。即使零宽正则也按行推进，不重复同一位置。
- 搜索按行读取，不再一次性 read_text/splitlines 整个文件；但任意正则需要保留被搜索的一整行，极长行内存和病态 regex CPU 仍受调用方 worker 的内存/时间预算保护，不能称为通用流式 regex 引擎。
- 普通文件以 nonblocking FD 打开，拒绝 FIFO/device，避免等候管道 writer。

## 原始结果与最终模型可见内容

`_bounded_decision_value` 对字典投影添加 `__context_projection__`：

```json
{
  "model_visible_complete": false,
  "source_truncated": false,
  "omission_count": 1,
  "omissions": [
    {"json_pointer": "/matches", "kind": "list_items", "original_count": 100, "visible_count": 8}
  ]
}
```

`source_truncated` 仅反映该源字典的字段（没有则 null），不会覆盖原字段，更不会改 episode 的 `truncated` 语义。字符串、列表、dict entries、深度和 inline artifact 的省略分别计数；最多保留 16 条省略记录，`omission_count` 可更大，长 JSON pointer 也有显示上限。这不是全树的无限详细 diff。

叶节点的 ID、布尔值和数字不消耗额外结构深度；固定大小文本游标按整体保留。超大 latest outputs 的降级投影保留文本/搜索/Python result 和明确完整结果引用，并报告源字段数/选中字段数。最终 provider 层保留有类型的文本/JSON artifact 路径，不再把它们当成视觉传输路径删除；普通相机路径仍隐藏。

完整链路测试从真实受限 Python worker 搜索 100 行，经 memory action、主 planner、生产 provider prompt formatter 得到最终请求 body，再用该 body 中的游标成功续读原文件。测试使用假 provider transport，不发真实模型请求，也不是 LIBERO 实验。

若模型投影截断了一个源文本页，源 `next_cursor` 仍指向**整个源页**之后，不是模型可见前缀之后；不要直接跳过未见文本。应从保留的源路径使用较小 `max_chars` 重读，或使用保留的版本与原始字符偏移定位。当前没有伪造一个压缩层专属文件游标。

仍开放：列表/标量根为兼容保持原形状，没有独立 projection envelope；若需完整性信息，调用者应放入字典。其他历史/conversation 压缩路径、完整 artifact 原生工具入口、显式图片打开后进入模型、JSON 字段级读取、性能与所有关键 ID 的优先级均需后续验收，不能以本批测试关闭这些要求。
