# compile_time_bench

**中文 | [English](README.md)**

对两类参数化算子进行 `torch.compile()` 编译阶段耗时 benchmark。
使用 `TORCH_LOGS` 和 `torch._dynamo` 内部指标采集各编译阶段的详细耗时，
同时测量硬件 kernel 执行时间，并对每个 case 进行 compile 模式与 eager 模式的精度对比。

## 项目结构

```
benchmark.py          – 主入口
samples/
  elementwise.py      – 参数化逐元素融合压测（1,408 个变体）
  gemm.py             – matmul / batch_matmul 维度遍历（512 个变体）
logs/                 – 各 case 的 TORCH_LOGS 输出（运行时生成）
results/              – CSV 结果文件的建议存放目录
```

## 输出列说明

### 耗时（单位：秒，另有说明除外）

| 列名 | 说明 |
|---|---|
| `repeats` | 每个 case 采集的编译样本数量 |
| `first_call_s` | **第 1 次**前向推理的 median wall-clock 耗时（触发完整编译） |
| `first_call_min_s` / `first_call_max_s` | 多次采样中的 first-call 最小/最大 wall-clock 耗时 |
| `second_call_s` | **第 2 次**前向推理的 median wall-clock 耗时（复用已编译产物） |
| `dynamo_s` | Dynamo 阶段 median：Python 字节码 tracing + guard 构建 |
| `aot_s` | AOT Autograd 阶段 median：联合图 lowering 和 metadata 收集 |
| `backend_s` | Inductor backend median：完整 codegen + kernel 编译 |
| `total_compile_s` | 编译总耗时 median（`_compile.compile_inner` wall-clock） |
| `total_compile_min_s` / `total_compile_max_s` | 多次采样中的编译总耗时最小/最大值 |
| `inductor_codegen_s` | Inductor 子阶段：`GraphLowering.codegen`（IR → C++/Triton） |
| `inductor_compile_s` | Inductor 子阶段：`compile_file`（C++/Triton → `.so`） |
| `inductor_load_s` | Inductor 子阶段：`PyCodeCache.load_by_key_path`（加载编译后的 kernel） |
| `pre_grad_passes_s` | 前向图变换 pass 耗时 |
| `post_grad_passes_s` | 后向图变换 pass 耗时 |
| `joint_graph_passes_s` | 前后向联合图变换 pass 耗时 |
| `kernel_time_ms` | 硬件 kernel 执行时间，单位**毫秒**（设备侧 event 时钟） |

### 计数器

| 列名 | 说明 |
|---|---|
| `cache_hit` | `1` 表示命中 AOT/FX 缓存，Inductor 编译被跳过 |
| `graph_breaks` | Dynamo 检测到的图断裂次数 |
| `frames_compiled` | 编译的帧/图数量 |

### 精度（compile 模式 vs. eager 模式）

在第一阶段（编译 benchmark）内对每个 case 完成。
输出在比较前统一转为 fp32，以避免 fp16 精度问题。

| 列名 | 说明 |
|---|---|
| `max_abs_err` | 最大逐元素绝对误差 |
| `mean_abs_err` | 平均逐元素绝对误差 |
| `max_rel_err` | 最大逐元素相对误差（以 eager 输出幅值为基准） |
| `cosine_sim` | 展平输出向量的余弦相似度（`1.0` = 完全一致） |

## Sample 模块

| `--case_type` | 说明 | 变体数 |
|---|---|---|
| `elementwise` | 参数化逐元素融合：算术运算（+−×÷）与激活算子 | 1,408 |
| `gemm` | `matmul` 和 `batch_matmul` 维度遍历 | 512 |

### elementwise 变体命名规则

```
elementwise_ni{N}_no{M}_sz{S}[_{bcast_mode}][_perm]_{dtype}

N          – 输入数量 n_inputs  ∈ {1, 2, 3, 4}
M          – 输出数量 n_outputs ∈ {1, 2, 3, 4}
S          – 最低维大小 ∈ {16, 256, 8192, 32768}
bcast_mode – （仅 n_inputs ≥ 2）
             no_bcast | 2d_high | 2d_low | 3d_high | 3d_mid | 3d_low | 3d_hl
_perm      – inputs[0] 最低两维被 permute（非连续 view）
dtype      – fp32 | fp16
```

每个 case 的激活算子数量随 `max(n_inputs, n_outputs)` 线性增加，上限为 4 个。
使用的激活算子：`sigmoid`、`tanh`、`relu`、`sqrt`。

### gemm 变体命名规则

```
matmul_m{M}_n{N}_k{K}_{dtype}
batch_matmul_b{B}_m{M}_n{N}_k{K}_{dtype}

M, N, K ∈ {64, 256, 1024, 4096}
B       ∈ {1, 8, 32}
dtype   ∈ {fp32, fp16}
```

## 执行阶段

每个 case 在第一阶段可重复采集多次编译样本（由 `--repeats` 控制，默认 `1`），
并可选择执行第二阶段 kernel 计时。

**第一阶段 — 编译 benchmark**（支持 `--workers N` 并行）

| 次数 | 用途 |
|---|---|
| 第 1 次 | 触发 `torch.compile` — 采集一次 `first_call_s` 及所有编译阶段指标 |
| 第 2 次 | 复用已编译产物 — 记录 `second_call_s` |
| 第 3 次 | eager 前向推理 — 精度对比的参考值 |
| 第 4 次 | compiled 前向推理 — 精度对比 → `max_abs_err`、`cosine_sim` 等 |

并行模式下，每个 worker 通过 `os.sched_setaffinity` 绑定到独占的 CPU 核组，
以降低调度竞争和编译耗时的抖动。

**第二阶段 — kernel 耗时采集**（始终在主进程串行执行）

| 次数 | 用途 |
|---|---|
| 第 5 次 | 在主进程重新编译 / 预热 compiled callable |
| 第 6 次 | 用 `torch.cuda.Event.elapsed_time()` 计时 — 记录 `kernel_time_ms` |

第二阶段强制串行，确保每次只有一个 kernel 在设备上执行，保证硬件计时的准确性。
CPU 设备使用 wall-clock 计时。
如果某个 case 的 kernel 计时失败，benchmark 会把它记录为错误，而不是写成
`0.0 ms` 测量值。

## 统计汇总

写完每个 case 的 CSV 数据行后，benchmark 会在文件末尾追加三行汇总
（`[max]`、`[min]`、`[avg]`），同时在标准输出打印相同的统计表格。

## 使用方法

```bash
# 运行所有 case（自动检测设备：优先 CUDA，否则 CPU）
python benchmark.py

# 运行指定类别
python benchmark.py --case_type gemm
python benchmark.py --case_type elementwise

# 运行指定名称的 case
python benchmark.py --case_name matmul_m1024_n1024_k1024_fp32
python benchmark.py --case_name elementwise_ni2_no1_sz256_no_bcast_fp16

# 限定类别后再按名称过滤
python benchmark.py --case_type elementwise \
    --case_name elementwise_ni4_no4_sz8192_3d_hl_perm_fp16

# 并行编译（kernel 耗时采集始终串行）
python benchmark.py --workers 4

# 更稳定地测 GEMM 编译耗时：串行、多次采样、隔离 cache、跳过 kernel 计时
python benchmark.py --case_type gemm --workers 1 --repeats 5 \
    --skip-kernel-timing --isolate-cache

# 自定义输出路径
python benchmark.py --output results/my_run.csv

# 显式指定设备
python benchmark.py --device cpu

# 指定 backend（默认：inductor）
python benchmark.py --backend aot_eager
```

结果写入 `compile_times.csv`（可通过 `--output` 修改），
文件末尾自动追加 `[max]`/`[min]`/`[avg]` 汇总行。
各 case 的 `TORCH_LOGS` 输出保存在 `logs/<case_name>.log`。

## TORCH_LOGS

脚本在导入 torch 前设置 `TORCH_LOGS="dynamo"`。
预热阶段和第二阶段 kernel 计时期间的日志输出会被抑制，保持终端整洁。

如需更详细的日志，在运行前覆盖该环境变量：

```bash
TORCH_LOGS="+dynamo" python benchmark.py
```

## 依赖

```
torch >= 2.1.0
```
