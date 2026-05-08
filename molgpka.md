# MolGpKa import 很慢的原因和解决办法

## 现象

`import molgpka` 会等待很久，常见耗时在 20 秒左右。  
如果每次 LLM agent 都新开一个 Python 进程调用 MolGpKa，这个等待会反复发生。

## 主要原因

MolGpKa 的顶层导入链路太重：

```text
import molgpka
-> molgpka/__init__.py
-> molgpka/api.py
-> molgpka/predict_pka.py
-> torch
-> torch_geometric
```

慢的核心不是 `.pth` 权重文件。两个模型权重大约各 5.7MB，而且是在预测时加载，不是在 import 阶段加载。

真正的大头是 `torch_geometric`。当前环境里的 PyG 2.7.0 在导入时会 eager import 很多子模块，包括 `data`、`datasets`、`nn`、`llm`、`profile` 等，导致一次 `import molgpka` 触发 PyTorch/PyG 全家桶初始化。

实测大致耗时：

```text
import molgpka              约 21-26 秒
import torch                约 1.6-2.3 秒
import torch_geometric      约 8-17 秒
from torch_geometric.data import Data  约 8 秒
```

## 另一个性能问题

当前预测函数里还有重复初始化：

- `predict_base()` 每次都会重新构建 base 模型并 `torch.load()`
- `predict_acid()` 每次都会重新构建 acid 模型并 `torch.load()`
- `get_ionization_aid()` 每次都会重新读取 SMARTS 表
- 一些 SMARTS pattern 也会反复编译

这不解释 import 慢，但会拖慢大批量预测。

## 只做 lazy import 的效果

可以把 `molgpka` 的重依赖延迟到第一次 `predict()` 时再导入。  
这样能让普通 `import molgpka` 变快，但第一次预测仍然会慢，因为 PyTorch/PyG 还是要初始化。

所以 lazy import 只解决“import 包就卡住”的问题，不解决“第一次真正预测很慢”的问题。

## 推荐解决方案

最合适的方案是启动一个长期存活的 MolGpKa 服务。

结构如下：

```text
LLM agent
-> 调本地 HTTP / Unix socket / JSONL worker
   -> 常驻 Python 进程
      -> 启动时 import molgpka
      -> 启动时加载 acid/base 模型
      -> 启动时缓存 SMARTS 表和 pattern
      -> 后续请求直接 predict
```

这样慢的初始化只发生一次。后续 agent 调用服务时，只需要发送 SMILES，通常可以很快返回结果。

注意：初始化不能存在于 conda 环境本身，只能存在于某个长期运行的 Python 进程内。  
如果每次都执行 `python -c "import molgpka; ..."`，仍然会每次重新初始化。

## 大批量 multiprocessing 建议

不要一个 molecule 开一个进程，也不要每批数据都重新创建 pool。

建议：

- 使用固定的长期 worker pool
- 每个 worker 启动时只 import 一次 MolGpKa
- 每个 worker 只加载一次 acid/base 模型
- 每个 worker 缓存 SMARTS 表和编译后的 SMARTS pattern
- 输入按 chunk 分发，一个 worker 一次处理一批 SMILES
- 限制线程数，避免 PyTorch 线程和多进程抢资源：

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

并在 worker 内设置：

```python
torch.set_num_threads(1)
```

## 最小服务化方案

可以先做一个本地 FastAPI 服务：

```python
from fastapi import FastAPI
from pydantic import BaseModel
from molgpka import MolGpKa

app = FastAPI()
predictor = MolGpKa()

class Request(BaseModel):
    smiles: str

@app.post("/predict")
def predict(req: Request):
    pred = predictor.predict_smiles(req.smiles)
    return pred.as_dict()
```

然后 LLM agent 调用：

```bash
curl -s http://127.0.0.1:8765/predict \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CC(=O)O"}'
```

## 最终结论

MolGpKa 慢主要是因为顶层 import 过早触发了 `torch_geometric` 的重初始化。  
如果只是 lazy import，只能把慢从 import 阶段挪到第一次 predict 阶段。  
如果希望 LLM agent 后续快速调用，应该把 MolGpKa 做成常驻服务，并在服务或 worker 初始化时缓存模型和 SMARTS 资源。
