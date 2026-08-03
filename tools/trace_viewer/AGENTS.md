# Trace Viewer 运行与临时公网分享

本目录维护最终 paper trace viewer。Viewer 不是单文件离线页面；`viewer.html` 会从同一 HTTP
根目录动态读取 condition、`predictions.jsonl`、sample trace 和 retrieval 文件。因此不要只上传
`viewer.html` 到单文件 HTML pastebin，也不要把一个本地文件 URL 当作完整 viewer 分享。

Viewer 通过启动时生成的 `.trace_viewer_sources.tsv` 注册 benchmark dataset，并用
`.trace_viewer_catalog.tsv` 注册已有 condition，避免公网页面逐目录扫描；缺少 catalog 时仍可回退为
browser-side discovery。每个 dataset 内只注册四个正式 paper root：identity-blind、matched-prefetch、agentic deployment-visible 和
deployment-visible parent-disjoint。Dataset path/label 不写死在 `viewer.html`；新增 benchmark 或 split
时只需由启动脚本注册新的 trace root。Parent-disjoint sample 必须同时读取 manifest 与 `reuse.json`，
明确展示 `neighbor_identity_policy`，并区分 retrieval 变化后的重跑与输入未变化时的 artifact reuse。

Condition results 必须从当前 `predictions.jsonl` 通用计算样本数、accuracy、macro-F1、失败数和可用时的
二分类混淆矩阵，不读取 task-specific prediction 字段，也不把不同 benchmark dataset 的样本合并。

## 展示名称与内部 ID

Viewer 可以为 paper-facing mechanism family 提供简洁、可读的展示名称，但不得改写 trace 或 retrieval
中的稳定 `group_id`。友好名称用于 stage、pill 和 retrieval group 标题；原始 ID 继续在 retrieval 标题 tag、
group metadata 和 raw JSON 中展示，以保留 provenance、input hash 和 branch reuse 的可审计性。

Bioavailability 的展示名称统一为 Direct oral bioavailability (F%)、Oral exposure proxies (AUC/Cmax)、
Fa、Fg 和 Fh 的机制描述。不要把 `Observed` 前缀展示成一个额外 reasoning branch。

## 本地启动

从仓库根目录运行：

```bash
bash tools/trace_viewer/start_viewer.sh 8776
```

默认会同时注册 Starling random test、Starling scaffold test 和历史 TDC test 三个结果根；不存在的根会
被跳过。也可以在端口后显式传入任意数量的 trace root：

```bash
bash tools/trace_viewer/start_viewer.sh 8776 \
  outputs/paper/molecular_evidence_agent_starling_random \
  outputs/paper/molecular_evidence_agent_starling_scaffold
```

本地页面固定为：

```text
http://127.0.0.1:8776/.trace_viewer.html?v=paper-v2
```

`start_viewer.sh` 会以前台进程运行；终端停在 `Serving HTTP ...` 并持续打印 GET 日志是正常状态，
不是卡住。`GET /favicon.ico ... 404` 只是浏览器自动请求了未提供的图标，不影响 viewer。

## 在 node002 安装 cloudflared

`node002` 是 Linux amd64，且 `$HOME/.local/bin` 已在 PATH 中。临时分享优先使用 Cloudflare
官方 standalone binary，安装到用户目录，不需要 sudo，也不修改系统 package repository。

下面的命令会从 Cloudflare 官方 GitHub release API 读取当前 latest release，并用 release asset
中公布的 SHA256 digest 校验后安装。不要在长期文档中写死某个历史版本号或 checksum。

```bash
(
  set -euo pipefail

  release_json="$(curl -fsSL https://api.github.com/repos/cloudflare/cloudflared/releases/latest)"
  asset_name="cloudflared-linux-amd64"
  download_url="$(printf '%s' "$release_json" | jq -r --arg name "$asset_name" '.assets[] | select(.name == $name) | .browser_download_url')"
  sha256="$(printf '%s' "$release_json" | jq -r --arg name "$asset_name" '.assets[] | select(.name == $name) | .digest' | sed 's/^sha256://')"
  tmp="$(mktemp)"

  test -n "$download_url"
  test -n "$sha256"
  test "$download_url" != "null"
  test "$sha256" != "null"

  curl -fL --retry 3 -o "$tmp" "$download_url"
  printf '%s  %s\n' "$sha256" "$tmp" | sha256sum -c -

  mkdir -p "$HOME/.local/bin"
  install -m 0755 "$tmp" "$HOME/.local/bin/cloudflared"
  rm -f "$tmp"
)

hash -r
cloudflared --version
```

如果 standalone binary 已安装，后续可用 `cloudflared update` 检查更新。

## 创建 Quick Tunnel

保持本地 viewer 终端运行，另开一个终端执行：

```bash
cloudflared tunnel --url http://127.0.0.1:8776
```

成功时日志会给出一个临时基础地址，例如：

```text
https://random-words.trycloudflare.com
```

Cloudflare 只打印公网基础地址，不知道 viewer 的具体页面路径。对外分享的完整 URL 需要把
本地 URL 中端口后的路径追加到公网基础地址：

```text
https://random-words.trycloudflare.com/.trace_viewer.html?v=paper-v2
```

通用拼接规则：

```text
公网基础地址 + /.trace_viewer.html?v=paper-v2
```

两个前台进程都必须保持运行：

```text
start_viewer.sh 负责提供 HTML 和 trace 文件。
cloudflared 负责把本地 8776 映射到临时公网域名。
```

分享结束后，在两个终端分别按 `Ctrl-C`。Quick Tunnel 重启后通常会得到新的随机域名，不要把
临时域名写进代码、文档或实验 manifest。

## 日志判断与故障处理

以下日志表示 tunnel 已正常建立：

```text
Your quick Tunnel has been created
Registered tunnel connection
SUMMARY: Environment is healthy
```

Quick Tunnel 不需要 `config.yml`，所以 `Cannot determine default configuration path` 是信息提示，
不是失败。QUIC 的 UDP receive-buffer warning 也可以在已经出现 `Registered tunnel connection`
且 connectivity pre-check 全部 PASS 时忽略。

如果 QUIC/UDP 无法建立，但 TCP/443 可用，改用 HTTP/2：

```bash
cloudflared tunnel --protocol http2 --url http://127.0.0.1:8776
```

分享前先验证本地页面；本地不通时，tunnel 也无法修复 origin：

```bash
curl -I 'http://127.0.0.1:8776/.trace_viewer.html?v=paper-v2'
```

## 安全边界

Quick Tunnel 是公开、无鉴权、无 uptime guarantee 的临时开发入口。`start_viewer.sh` 只在临时 serving
directory 中链接本次注册的 trace roots，不暴露仓库或整个 `outputs/paper/`；但任何拿到 URL 的人仍可能
请求这些已注册 root 中的其它文件，而不仅是浏览器中当前打开的样本。因此：

1. 只在确认 trace、prompt、SMILES、label、内部路径和 model/tool 返回内容可以分享时启动 tunnel。
2. 不要用 Quick Tunnel 暴露 API key、`.env`、工具服务端口、LLM endpoint 或仓库根目录。
3. 临时演示结束立即关闭两个进程；需要长期、固定域名或访问控制时改用 Cloudflare named tunnel
   和 Access policy，而不是继续依赖匿名 Quick Tunnel。
