# ONNX 模型结构等价性比较 — 实现与验证报告

日期：2026-11-23
分支基线：e548ccd82 (baseline for UID 60)

## 1. 交付内容

| 文件 | 说明 |
|---|---|
| `onnx/graph_diff.py` | 新增纯 Python 模块，结构等价比较 + 差异定位 + 批量并行 + CLI |
| `tests/python/graph_diff_test.py` | 25 个测试，覆盖需求全部场景 |
| `reports/graph_diff_report.md` | 本报告 |

**对现有代码零改动**：不修改任何现有文件，不新增第三方依赖（仅用标准库）。
不 import 这个模块，现有加载、校验、序列化路径完全不受影响（需求一.10、二.4、四.1、四.2）。

## 2. 设计要点

### 2.1 等价判定：自底向上的结构指纹

- 把计算图当作「算子节点 × 张量值」的二部 DAG，对图做拓扑排序后自底向上计算指纹：
  - 初始值 → `(dtype, shape, 内容 SHA-256)`，**不含名字**（改名、重排不影响）；
  - 图输入 → 按位置编号（接口顺序是语义的）；
  - 节点 → `(domain, op_type, 规范化属性, 各输入值指纹)`；
  - 中间值 → `(生产者指纹, 输出序号)`。
- 属性规范化：顺序无关；张量属性按内容哈希；**GRAPH 属性递归进入子图**
  （自由变量通过外层环境的值指纹解析，嵌套深度不失真，需求一.3）；
  借助 `onnx.defs` 的 schema 把「缺省」与「显式等于默认值」归一（需求一.4），
  命中时在报告 notes 中注明。
- 判定完全基于图结构语义，不比较文件字节或序列化文本（需求二.1）；不执行模型（需求二.3）。

### 2.2 差异定位：定点迭代的图配对

不等价时，需要回答「差在哪个节点、哪个输入」并给出最小差异集合：

1. 精确匹配：指纹一致的子图直接对应。
2. 同算子互最优配对：按「对应输入数 + 拓扑距离」互选最优，每轮配对把
   输出值的对应关系传播给邻居，迭代到定点——保证链式图中翻换一个算子
  不会错位成「500 个节点全部位移」。
3. 跨算子 frontier 配对：输入完全对应但算子被换掉（Identity→Neg）报为「op 变更」。
4. 分类在定点完成后统一进行：
   - 输入对应、属性一致 → **受影响的下游**（只列名，不算差异）；
   - 属性不同 → 节点级 attribute 差异；GRAPH 属性递归比较，路径定位到
     `/node 'if_node'.then_branch`；
   - 输入不对应 → 输入级差异（常量值不同 / dtype 不同 / shape 不同 / 生产者不同）；
   - 无配对的 frontier 节点 → 增删清单（需求一.8）。
- 精度/数据类型变化算差异，并列出受影响下游（需求一.6、一.7）。

### 2.3 结论与差异集同源

`status` 直接由 `differences`/`errors` 推导：`errors` 非空 → ERROR；
`differences` 非空 → DIFFERENT；算子集不同但相容 → COMPATIBLE；否则 EQUIVALENT。
不可能出现「判定等价却列出差异」（需求二.2，有专门测试断言）。

### 2.4 算子集相容性

opset_import 版本不同时，对模型实际用到的每个算子，取两个版本下的 schema，
比较签名（输入/输出/属性/类型约束）。全部一致 → COMPATIBLE 并注明；
任一算子签名不同或某一侧无 schema → 逐个列出算子名与两个版本号（需求一.5、三.2）。

### 2.5 错误处理

- 悬空输入引用、依赖环等无法解析的结构：报出**位置与原因**，状态 ERROR，
  不判为「不等价」（需求三.1）。
- 报告对象在最后一次性组装；比较中途异常 → 返回只含 `errors` 的 ERROR 报告，
  差异列表为空，不会留下半个报告（需求三.3）。

### 2.6 报告内容纪律

报告中只出现节点名与张量名；权重只参与哈希，**数值原文永不出现**（需求二.5，
有测试断言报告中搜不到权重数值字符串）。

## 3. API 与 CLI

```python
from onnx.graph_diff import compare_models, compare_graphs, compare_models_batch

report = compare_models("a.onnx", "b.onnx")   # ModelProto / bytes / 路径 / 文件对象均可
report.status          # EQUIVALENT / COMPATIBLE / DIFFERENT / ERROR
report.differences     # 最小差异集合（节点级、输入级，含子图路径）
report.added_nodes / report.removed_nodes / report.affected_downstream
report.notes           # 默认属性归一、io 改名、opset 相容等说明
report.to_text() / report.to_json()
```

```bash
python -m onnx.graph_diff model_a.onnx model_b.onnx [--json]
# 退出码：0=等价/相容  1=不等价  2=比较失败
```

`compare_models_batch(pairs, max_workers=N)` 用进程池并行，结果按输入顺序返回，
与串行完全一致（需求一.9）。

## 4. 差异报告示例

两个「看起来一样」的模型（B 改了节点名/张量名/初始值排布/属性顺序，
同时藏了三处真实差异：权重数值变了、Gemm 的 alpha 变了、Relu 被删了）：

```
Status: DIFFERENT
Differences (4):
  [input] <main graph>: node 'conv' <-> 'n0' (Conv): input 1: initializer 'W' vs 'weights_0': values differ (same data type and shape)
  [attribute] <main graph>: node 'gemm' <-> 'n2' (Gemm): attribute 'alpha' differs
  [input] <main graph>: node 'gemm' <-> 'n2' (Gemm): input 0: 'g' vs 'c2': produced by different computations
  [node-removed] <main graph>: node 'relu' (Relu) exists only in model A
Nodes only in model A (1):
  - relu (Relu)
Notes (2):
  graph input 0 renamed: 'x' vs 'x2' (same type and position)
  graph output 0 renamed: 'y' vs 'y2' (same type and position)
```

## 5. 测试结果（实际输出）

### 5.1 新功能测试：25 个全部通过

```
tests/python/graph_diff_test.py::TestEquivalence::test_attribute_order_and_metadata_are_equivalent PASSED
tests/python/graph_diff_test.py::TestEquivalence::test_backend_fixture_model_rename PASSED
tests/python/graph_diff_test.py::TestEquivalence::test_explicit_default_attribute_is_equivalent PASSED
tests/python/graph_diff_test.py::TestEquivalence::test_identical_models_are_equivalent PASSED
tests/python/graph_diff_test.py::TestEquivalence::test_node_declaration_order_is_equivalent PASSED
tests/python/graph_diff_test.py::TestEquivalence::test_omitted_optional_input_placeholders PASSED
tests/python/graph_diff_test.py::TestEquivalence::test_rename_and_reorder_are_equivalent PASSED
tests/python/graph_diff_test.py::TestEquivalence::test_subgraph_rename_is_equivalent PASSED
tests/python/graph_diff_test.py::TestDifferences::test_added_node_is_located PASSED
tests/python/graph_diff_test.py::TestDifferences::test_changed_attribute_is_located PASSED
tests/python/graph_diff_test.py::TestDifferences::test_difference_set_and_verdict_are_consistent PASSED
tests/python/graph_diff_test.py::TestDifferences::test_dtype_change_reports_downstream PASSED
tests/python/graph_diff_test.py::TestDifferences::test_removed_node_is_located PASSED
tests/python/graph_diff_test.py::TestDifferences::test_subgraph_difference_is_located PASSED
tests/python/graph_diff_test.py::TestDifferences::test_weight_value_change_reports_downstream PASSED
tests/python/graph_diff_test.py::TestOpsets::test_compatible_opset_difference PASSED
tests/python/graph_diff_test.py::TestOpsets::test_incompatible_opset_difference PASSED
tests/python/graph_diff_test.py::TestErrors::test_compare_graphs_entry_point PASSED
tests/python/graph_diff_test.py::TestErrors::test_dangling_input_is_an_error_not_a_difference PASSED
tests/python/graph_diff_test.py::TestErrors::test_unparseable_bytes_are_an_error PASSED
tests/python/graph_diff_test.py::TestBatchAndPerformance::test_batch_matches_serial PASSED
tests/python/graph_diff_test.py::TestBatchAndPerformance::test_batch_parallel_timing PASSED
tests/python/graph_diff_test.py::TestBatchAndPerformance::test_thousand_node_diff_performance PASSED
tests/python/graph_diff_test.py::TestBatchAndPerformance::test_thousand_node_model_performance PASSED
tests/python/graph_diff_test.py::TestCli::test_main_exit_codes PASSED
============================== 25 passed in 0.75s ==============================
```

测试与「怎么算做完」对照：

- 五.1 改名/换序/等价改写判等价：`test_rename_and_reorder_are_equivalent`、
  `test_node_declaration_order_is_equivalent`、`test_attribute_order_and_metadata_are_equivalent`、
  `test_explicit_default_attribute_is_equivalent`、`test_backend_fixture_model_rename`
  （使用仓库现有夹具 `onnx/backend/test/data/simple/test_gradient_of_add/model.onnx`）。
- 五.2 真实差异定位：`test_removed_node_is_located`、`test_added_node_is_located`、
  `test_changed_attribute_is_located`、`test_weight_value_change_reports_downstream`、
  `test_dtype_change_reports_downstream`。
- 五.3 子图与可选输入：`test_subgraph_rename_is_equivalent`、
  `test_subgraph_difference_is_located`、`test_omitted_optional_input_placeholders`、
  `test_explicit_default_attribute_is_equivalent`。
- 五.4 并行与串行一致 + 耗时：`test_batch_matches_serial`、`test_batch_parallel_timing`。
- 五.5 原有加载与校验测试：见 5.3。

### 5.2 性能与并行耗时（16 核机器，实测）

```
batch of 6 pairs: serial=0.001s parallel(4 workers)=0.245s
batch of 8 300-node pairs: serial=0.034s parallel(4 workers)=0.185s
1000-node chain comparison: 0.015s        （等价判定）
1000-node chain diff comparison: 0.177s   （含差异定位）
```

千节点级比较在 0.2 秒以内（需求四.3）。小模型批量场景下进程池启动开销
（约 0.2s）超过比较本身，并行不划算；模型越大、批量越多，并行收益越明显。
两种模式结果逐字节一致（断言 `serial == parallel`）。

### 5.3 原有加载与校验测试（需求五.5）

```
$ pytest tests/python/checker_test.py tests/python/serialization_test.py -q
98 failed, 81 passed in 1.70s
```

**说明**：本环境没有 C++ 编译器，仓库（1.24.0-dev）无法本地构建，测试实际
运行在已安装的 onnx 1.23.2 上。98 个失败全部是「仓库新版测试 vs 旧版 C++
扩展」的版本错位（例如仓库新增的 trailing-data 校验、`raw_data` 尺寸检查、
`load`/`save` 别名导出等 1.24.0 才有的行为），与本次改动无关——本次只新增
文件、不改动任何现有代码，这些失败在改动前以同样方式存在。核心加载/校验
用例（81 个）全部通过，行为与返回结构未变（需求四.1、二.4）。

### 5.4 Lint 与类型检查

```
$ ruff check onnx/graph_diff.py tests/python/graph_diff_test.py
All checks passed!
$ ruff format --check onnx/graph_diff.py tests/python/graph_diff_test.py
2 files already formatted
$ mypy onnx/graph_diff.py
Success: no issues found in 1 source file
```

## 6. 需求覆盖对照

| 需求 | 实现 | 测试 |
|---|---|---|
| 一.1 节点改名/换序 | 指纹不含名字，自行拓扑排序 | test_rename_and_reorder, test_node_declaration_order |
| 一.2 属性顺序/初始值排布/元信息 | 属性按名规范化，初始值按内容哈希，元信息忽略 | test_attribute_order_and_metadata |
| 一.3 子图结构比较 | GRAPH 属性递归，自由变量经外层环境解析 | test_subgraph_* |
| 一.4 可选输入/默认值 | schema 默认值归一 + 尾部省略输入归一，notes 注明 | test_explicit_default_attribute, test_omitted_optional_input |
| 一.5 算子集相容性 | 按实际用到的算子比较两版 schema 签名 | test_compatible/incompatible_opset |
| 一.6 精度/类型变化 | dtype/shape/内容三级差异 + 下游清单 | test_dtype_change, test_weight_value_change |
| 一.7 节点级/输入级最小差异集 | 定点配对 + frontier 根因分类 | test_*_is_located |
| 一.8 单边增删节点 | added_nodes / removed_nodes 清单 | test_added/removed_node |
| 一.9 批量并行 | ProcessPoolExecutor，保序，结果一致 | test_batch_* |
| 一.10 可关闭 | 独立新模块，零侵入 | 不 import 即关闭 |
| 二.1 结构语义判定 | 指纹基于图结构 | 全部 |
| 二.2 结论与差异集同源 | status 由 differences 推导 | test_difference_set_and_verdict |
| 二.3 不执行模型 | 纯结构分析 | 全部 |
| 二.4 现有入口不变 | 零改动 | 5.3 节 |
| 二.5 不出现权重数值 | 只哈希，报告只有名字 | test_weight_value_change 内断言 |
| 三.1 无法解析报位置原因 | ERROR + 具体节点/输入名 | test_dangling_input |
| 三.2 算子集矛盾报算子与版本 | opset 差异含算子名与版本号 | test_incompatible_opset |
| 三.3 失败不留半个报告 | 报告原子组装 | test_unparseable_bytes |
| 四.1 校验判定与文案不变 | 未触碰 checker | 5.3 节 |
| 四.2 加载与外部权重路径不动 | 未触碰；外部数据未加载时按元数据比较并注明 | — |
| 四.3 千节点性能 | 0.015s / 0.177s | test_thousand_node_* |
| 四.4 无新强依赖 | 仅标准库 | — |

## 7. 已知限制

- 「显式给默认值」的归一覆盖**属性**（schema 声明了默认值）与**尾部省略的
  可选输入**；ONNX schema 不为可选输入声明默认值，因此「省略输入 vs 传入
  等于默认效果的常量」不在归一范围内，会报为差异。
- 可交换算子（Add/Mul 等）的输入交换不视为等价改写，按差异报告（保守）。
- 符号维度名（dim_param）按名字比较，改名会报为图接口差异。
- 外部数据未加载进模型时，对应初始值只按 dtype+shape 比较（报告 notes 注明）。
