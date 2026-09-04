# Progressive Trace Viewer

本目录只维护当前 append-only progressive agent viewer。旧的 single/group/final、identity-blind、
matched-prefetch 和历史 TDC trace 不再适配。Viewer 不是单文件离线页面；`viewer.html` 会从同一 HTTP
根目录动态读取每个 task 的 `none/predictions.jsonl`、`levels/level_N/predictions.jsonl`，并在选中样本后
按需读取 `queries/query_idxNNNNN/levels/level_N/{prepared,request,output}.json`。

`start_viewer.sh` 从 `current_conditioned_results.json` 解析 registry-pinned scaffold progressive roots，
在临时 serving root 中只链接这些 roots，并生成 `.trace_viewer_catalog.tsv`。不得根据目录名推断 current
lineage；registry status 和 retrieval-change receipt 必须显示在页面上。显式查看 rerun 时可以传入
`task=progressive-run-root`，此时页面明确标记为 `manual_override`。

每层结果必须从通用 `label`、`pred_label`、`correct`、`status`、`model_called` 和 `revision_action` 字段读取，
不得读取 task-specific prediction 字段。页面按 query 对齐 Prior、L1...LN，显示 always-correct、
always-wrong、rescued、harmed、oscillating 和 incomplete trajectory，并从相邻层 prediction rows 计算
accuracy、macro-F1、correctness transitions 和 prediction flips。

## Level 与 evidence 展示

Level 名称和说明只读取每个 query 的 `prepared.json.level_definition`；不得在 viewer 中复制 task family
配置。右侧每层只保留纵向的 `Prompt`、`Reasoning`、`Output` 三个折叠入口：Prompt 展示完整 request
messages，Reasoning 展示完整 model reasoning，Output 优先结构化展示 LLM response，并在其内部保留 raw
response、validation、provider attempts 和 usage。Prior 不是一次 model call，必须单独简洁标明。结构化对象
使用字段名在上、值在下的纵向布局；`*_card_ids` 使用可换行的紧凑 chips。LLM-visible request/response 中的
证据引用保持实际的 `C01`、`C02` 等短 alias，不在主阅读面重复 prepared、active-evidence 或稳定长 ID。
`user.content` 虽由 API 保存为 JSON string，viewer 必须先解析，再按 protocol、task definition、level
context、query、query prior、active evidence、required schema 和 prior state 分区；active evidence 进一步按
analog 和 evidence card 渲染。Evidence card 的短 metadata 使用自适应紧凑网格，support text 独占整行；
旧 card 的 prior-use 状态显示在 card 顶部。Analog 的 ID、similarity、bucket、relation、first-seen level 和
card count 集中在标题，额外短字段复用紧凑网格；只有 canonical SMILES 和 tool text 独占整行。完整原始
message 只放在该消息底部的审计折叠项中。

## 本地启动

从仓库根目录运行：

```bash
bash tools/trace_viewer/start_viewer.sh 8776
```

默认注册 current registry 中 BBB、Bioavailability 和 Skin 的 retained scaffold-valid progressive roots，并按
registry status 标明 current/stale；不存在或
不符合 progressive contract 的 root 会被跳过。也可以在端口后显式传入 task 与 run root：

```bash
bash tools/trace_viewer/start_viewer.sh 8776 \
  bbb_martins=outputs/paper/<progressive-run-root>
```

本地页面固定为：

```text
http://127.0.0.1:8776/.trace_viewer.html?v=paper-v3
```

`start_viewer.sh` 会以前台进程运行；终端停在 `Serving HTTP ...` 并持续打印 GET 日志是正常状态，
不是卡住。`GET /favicon.ico ... 404` 只是浏览器自动请求了未提供的图标，不影响 viewer。

## 可选临时分享

本地验证通过且 trace 内容允许公开时，可在另一终端临时转发：

```bash
cloudflared tunnel --url http://127.0.0.1:8776
```

把 viewer 路径追加到命令返回的临时域名：

```text
https://random-words.trycloudflare.com/.trace_viewer.html?v=paper-v3
```

Quick Tunnel 是公开、无鉴权、无 uptime guarantee 的临时入口。拿到 URL 的人可以读取已注册 trace
root 中的文件；只有在 prompt、SMILES、label 和 model/tool 返回允许公开时才能使用，结束后立即关闭。
不得暴露密钥、`.env`、工具服务端口或仓库根目录。
