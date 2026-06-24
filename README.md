# WCAS

WCAS（WRF-CMAQ Auto Script）是用于自动化驱动 WPS、WRF、CMAQ、MEGAN 与 MEIAT-CMAQ 的区域空气质量模拟流程脚本。本仓库当前以 `WCAS_codex.py` 为主版本，`namelist.WCAS` 为统一控制文件，`namelists_cshfiles/` 中的 namelist 与 csh 模板用于在运行时生成或更新各模型的实际执行参数。

脚本适用于已经能够手动完成 WRF-CMAQ 模拟的环境。使用 WCAS 前，应先确认 WPS、WRF、CMAQ、MCIP、ICON、BCON、CCTM、MEGAN、MEIAT-CMAQ 等组件在目标机器上可以独立正常运行。

## 主要功能

- 根据 `namelist.WCAS` 自动整理模拟目录、读取路径、时间、网格、并行核数、排放与后处理开关。
- 自动获取或整理 FNL/GFS 再分析资料，并按模拟名称归档到 `simulations/<GridName>/fnlfiles/`。
- 根据经纬度范围自动计算 WRF 嵌套网格，或使用 `&manualgrid` 中的固定网格参数复现实验。
- 自动生成 WPS `namelist.wps` 和 WRF `namelist.input`，并运行 geogrid、ungrib、metgrid、real、wrf。
- 支持 `profile` 与 `regrid` 两种 CMAQ 模拟流程。
- 自动运行 MCIP、BCON、ICON、MEIAT、可选 MEGAN、可选 ISAM 与 CCTM。
- 支持 CMAQ 输出结果 combine，并整理 WRF 输出。
- 使用 flag 文件记录步骤状态，便于断点续跑。

## 运行环境

### 外部模型与工具

建议准备以下模型或工具，并确保路径与版本和 `namelist.WCAS` 保持一致：

- WPS，例如 WPS v4.1
- WRF，例如 WRF v4.1.2、v4.3.3 或 v4.5.1
- CMAQ，例如 CMAQ v5.3 或 v5.4
- MCIP、ICON、BCON、CCTM 与 CMAQ POST/combine
- MEGAN v2.1，可选但建议用于生物源排放
- prepmegan4cmaq，可选，随 MEGAN 流程使用
- MEIAT-CMAQ 或 MEIAT-CMAQ-Linux
- MPI、csh、netCDF、I/O API、m3tools、mmv 等模型运行依赖
- 可选：SLURM，用于 `ScNet_env = .true.` 的集群提交场景

### Python 依赖

`WCAS_codex.py` 直接使用的主要 Python 包包括：

- `f90nml`
- `wrf-python`
- `pyproj`
- `numpy`
- `netCDF4`
- `gdal` / `osgeo`
- `matplotlib`
- `cartopy`
- `requests`
- `urllib3`
- `paramiko`
- `tqdm`

含 Windows ArcGIS Pro 空间处理的 MEIAT 流程还需要 `MEIAT_python_dir` 指向可运行对应脚本的 Python 解释器。Linux 版 MEIAT 流程由 `MEIAT_Linux` 与 `MEIAT_Linux_dir` 控制。

## 快速开始

1. 配置 `namelist.WCAS`。
2. 确认 `namelists_cshfiles/` 中的模板脚本与当前 CMAQ、WRF、MEGAN、MEIAT 环境匹配。
3. 在 WCAS 仓库目录运行：

```bash
python3 WCAS_codex.py
```

如果使用打包后的可执行文件，应按打包版本要求传入 WCAS 所在目录；当前源码版本通过 `__file__` 自动识别仓库目录。

运行日志写入 `WCAS.log`，模拟输出与中间文件按 `GridName` 归档到：

```text
simulations/<GridName>/
```

## 主流程

`WCAS_codex.py` 的入口流程如下：

1. 初始化：读取 `namelist.WCAS`，创建 `WCAS.log`、`simulations/<GridName>/` 与 `flagfiles/`。
2. 时间计算：以 `&date` 中的 `start_date` 和 `end_date` 为 CMAQ 目标时段，WRF 前后各扩展 2 天，MCIP 前后各扩展 1 天。
3. 再分析资料：根据 `direct_linkreanalysis`、`remote_server_download`、`ForceGFS` 等参数整理 FNL/GFS 文件。
4. 网格设置：根据 `ManualGrid` 选择自动计算嵌套网格或读取 `&manualgrid` 固定网格，并写入 WPS/WRF 参数。
5. WPS：运行 geogrid、ungrib、metgrid。若 `testGeogrid = .true.`，会输出 geogrid 检查图并允许重新调整网格。
6. WRF：生成 `namelist.input`，运行 `real.exe` 和 `wrf.exe`，并检查 `wrfout` 是否正常输出。
7. CMAQ：若 `runCMAQ = .true.`，进入 MCIP、BCON、ICON、排放、CCTM 与后处理流程。
8. 后处理：若 `runPostprocess = .true.`，进行 CMAQ combine，并将 WRF 输出归档。
9. 完成：写入 `WCAS.ok` flag 文件。

## CMAQ 模拟模式

### profile

`CMAQsimtype = 'profile'` 时，脚本直接对 `regrid_dom` 指定的目标层进行 MCIP、BCON、ICON、排放与 CCTM 计算。边界条件采用 profile 方式，流程相对简单，计算时间较短。

典型流程：

```text
WRF -> MCIP(d0<regrid_dom>) -> BCON(profile) -> ICON(profile)
    -> MEGAN/MEIAT -> CCTM(profile) -> combine
```

### regrid

`CMAQsimtype = 'regrid'` 时，脚本先计算 d01，再使用外层 CCTM 浓度场为目标层生成 regrid 边界条件。该模式耗时更长，但目标层边界场通常更一致。

当 `regrid_D02andD03 = .false.` 时：

```text
d01 profile CCTM -> target domain regrid BCON/ICON -> target domain CCTM
```

当 `regrid_D02andD03 = .true.` 时：

```text
d01 -> d02 -> d03
```

此时脚本会逐层运行 MCIP、BCON、ICON、排放与 CCTM。若 `max_dom < 3`，不会运行 d03。

## 断点续跑

WCAS 在 `simulations/<GridName>/flagfiles/` 中写入每个关键步骤的 `.ok` 文件。例如：

- `Reanalysis.ok`
- `WPS.ok`
- `WRF.ok`
- `MCIP_d01.ok`
- `BCON_profile_d01.ok`
- `ICON_regrid_d03.ok`
- `MEIAT_d03.ok`
- `CCTM_regrid_d03.ok`
- `Post_combine.ok`
- `WCAS.ok`

再次运行同一 `GridName` 时，已完成步骤会被跳过。若需要重新运行某一步，应在确认对应输出可以被覆盖或重新生成后，手动删除相应 flag 文件以及相关中间结果。

## `namelist.WCAS` 参数说明

### `&dirs`

模型和辅助工具路径。路径末尾建议保留 `/`。

| 参数 | 说明 |
| --- | --- |
| `WPS_dir` | WPS 安装目录。脚本会在其中写入 `namelist.wps` 并运行 WPS 程序。 |
| `WRF_dir` | WRF 安装目录。脚本默认使用 `WRF_dir/run/` 运行 real 和 wrf。 |
| `CMAQ_dir` | CMAQ 安装目录。用于调用 MCIP、BCON、ICON、CCTM 与 POST。 |
| `premegan_dir` | `prepmegan4cmaq` 程序目录。 |
| `premeganinput_dir` | prepMEGAN 输入数据目录，例如 EF、PFT 等。 |
| `premeganoutput_dir` | prepMEGAN 输出目录，供 MEGAN 后续读取。 |
| `MEGAN_dir` | MEGAN v2.1 目录。 |
| `MEIAT_dir` | Windows/ArcGIS Pro 版 MEIAT-CMAQ 目录，WSL 下通常使用 `/mnt/...` 路径。 |
| `MEIAT_python_dir` | 可运行 MEIAT Windows 脚本的 Python 解释器路径。 |

### `&grid`

当 `ManualGrid = .false.` 时使用，用经纬度范围自动计算 WRF 嵌套网格。

| 参数 | 说明 |
| --- | --- |
| `ratio` | 嵌套分辨率比例。当前主流程按 3 倍嵌套处理。 |
| `res_d01` | d01 水平分辨率，单位为米。 |
| `std_lat1_d01` | Lambert 投影第一标准纬线。 |
| `std_lat2_d01` | Lambert 投影第二标准纬线。 |
| `lat_min_d01` / `lat_max_d01` | d01 纬度范围。 |
| `lon_min_d01` / `lon_max_d01` | d01 经度范围。 |
| `lat_min_d02` / `lat_max_d02` | d02 纬度范围。 |
| `lon_min_d02` / `lon_max_d02` | d02 经度范围。 |
| `lat_min_d03` / `lat_max_d03` | d03 纬度范围。 |
| `lon_min_d03` / `lon_max_d03` | d03 经度范围。 |

`max_dom` 决定实际使用几层。即使 `&grid` 中写有 d03，`max_dom = 2` 时也只使用 d01 和 d02。

### `&date`

| 参数 | 说明 |
| --- | --- |
| `start_date` | CMAQ 目标模拟开始日期，格式为 `YYYY-MM-DD`。 |
| `end_date` | CMAQ 目标模拟结束日期，格式为 `YYYY-MM-DD`。 |

脚本内部会自动扩展时间：

- WRF：`start_date - 2 天` 到 `end_date + 2 天`
- MCIP：`start_date - 1 天` 到 `end_date + 1 天`
- MEGAN、MEIAT、CCTM：通常使用 MCIP 起始日到 `end_date`

### `&control`

全局运行控制。

| 参数 | 说明 |
| --- | --- |
| `cores_win` | MEIAT 或 MEIAT-Linux 使用的并行核数。 |
| `cores_WRF` | WRF 使用的 MPI 核数；小区域模拟不宜盲目增大。 |
| `cores_CMAQ_col` | CMAQ 并行列方向划分数。 |
| `cores_CMAQ_row` | CMAQ 并行行方向划分数。CMAQ 总核数为二者乘积。 |
| `GridName` | 本次模拟名称，也是 `simulations/<GridName>/` 的目录名。每个实验建议使用唯一名称。 |
| `CMAQsimtype` | CMAQ 模式，可选 `'profile'` 或 `'regrid'`。 |
| `regrid_dom` | 最终输出或重点模拟的目标层编号。 |
| `regrid_D02andD03` | 是否按 d01 -> d02 -> d03 连续 regrid。 |
| `CCTMversion` | CCTM 可执行文件版本号，例如 `532` 对应 `CCTM_v532.exe`。 |
| `BCICversion` | BCON/ICON 相关版本标识。 |
| `compiler_str` | CMAQ 编译器标识，用于定位编译目录。 |
| `max_dom` | WRF/CMAQ 最大嵌套层数，只支持 1、2 或 3。 |
| `ScNet_env` | 是否使用曙光/集群环境逻辑。为 `.true.` 时部分步骤使用 SLURM。 |
| `direct_linkreanalysis` | 是否直接整理已有再分析资料，而不自动下载。 |
| `direct_linkreanalysis_dir` | 已有 FNL 资料所在目录，供直接复制到模拟目录。 |
| `aria2c_download` | 是否使用 aria2c 下载。当前主流程中保留该开关，具体下载逻辑需与工具脚本匹配。 |
| `remote_server_download` | 是否通过远程服务器下载 FNL/GFS。 |
| `remoteServer_hostname` | 远程服务器地址。 |
| `remoteServer_port` | 远程服务器 SSH 端口。 |
| `remoteServer_username` | 远程服务器用户名。 |
| `remoteServer_password` | 远程服务器密码。建议避免将真实密码提交到版本库。 |
| `ForceGFS` | 是否强制使用 GFS 预报场。为 `.false.` 时优先使用 FNL，缺失后切换 GFS。 |
| `MannualExtent` | 是否手动指定后处理绘图范围。变量名沿用历史拼写。 |
| `Extent` | 后处理绘图范围，顺序为 `lon_min, lon_max, lat_min, lat_max`。 |
| `testGeogrid` | 是否在 WPS geogrid 后输出嵌套范围检查图，并允许交互式修改网格后重跑 geogrid。 |
| `dom_now_wind` | 后处理风场绘图时是否使用当前层风场。 |
| `ManualGrid` | 是否使用 `&manualgrid` 中的固定网格参数。 |
| `runCMAQ` | 是否运行 CMAQ。为 `.false.` 时只执行 WRF 相关流程。 |
| `runMEGAN` | 是否运行 MEGAN 生物源排放。 |
| `runISAM` | 是否打开 ISAM 源解析流程。需要额外准备 ISAM 输入文件。 |
| `runPostprocess` | 是否运行后处理与结果整理。 |
| `CMAQcombine` | 是否调用 CMAQ 官方 POST/combine。为 `.false.` 时使用脚本内置 O3/PM2.5 combine 流程。 |
| `Mannualterminal` | 历史终端控制参数。当前 `WCAS_codex.py` 主流程已主要改为 subprocess 执行。 |
| `control_terminal` | 历史手动 tty 参数。 |
| `MEIAT_Linux` | 是否使用 Linux 版 MEIAT。为 `.false.` 时使用 Windows/ArcGIS Pro 版 MEIAT。 |
| `MEIAT_Linux_dir` | Linux 版 MEIAT-CMAQ 目录。 |

### `&manualgrid`

当 `ManualGrid = .true.` 时使用，用于复现已有 WPS/WRF 嵌套网格。

| 参数 | 说明 |
| --- | --- |
| `i_parent_start` | 各层在父网格中的 x 方向起始索引。 |
| `j_parent_start` | 各层在父网格中的 y 方向起始索引。 |
| `e_we` | 各层 west-east 网格点数。 |
| `e_sn` | 各层 south-north 网格点数。 |
| `dx` | d01 x 方向分辨率，单位为米。 |
| `dy` | d01 y 方向分辨率，单位为米。当前脚本主要读取 `dx` 并按 3 倍嵌套推导各层分辨率。 |
| `ref_lat` | 投影中心纬度。 |
| `ref_lon` | 投影中心经度。 |
| `truelat1` | 第一标准纬线。 |
| `truelat2` | 第二标准纬线。 |
| `stand_lon` | 标准经线。 |

`i_parent_start`、`j_parent_start`、`e_we`、`e_sn` 至少应提供 `max_dom` 个值。

### `&ISAMcontrol`

仅当 `runISAM = .true.` 时使用。

| 参数 | 说明 |
| --- | --- |
| `if_makemask` | 是否由脚本自动根据区域 shp 生成 ISAM mask。 |
| `mask_shpfilenames` | 区域 shp 文件名，不含扩展名。 |
| `mask_varnames` | ISAM mask 变量名，通常不超过 3 个字符；多个区域时需与 shp 名称对应。 |

自动 ISAM 需要准备 `input/ISAM/ISAM_CONTROL.txt`、`input/ISAM/EmissCtrl_cb6r3_ae6_aq.nml` 以及区域 shp 文件。若不自动生成 mask，应手动提供符合 CMAQ-ISAM 要求的 `ISAM_REGION.nc`。

### `&ScNetcontrol`

仅当 `ScNet_env = .true.` 时使用。

| 参数 | 说明 |
| --- | --- |
| `WPS_WRF_env` | 集群环境下 WPS/WRF 运行前需要加载的环境脚本。 |

## 目录结构

```text
WCAS/
├── WCAS_codex.py                 # 当前主版本脚本
├── WCAS.py                       # 历史/对照版本
├── namelist.WCAS                 # 主控制参数
├── namelists_cshfiles/           # WPS、WRF、CMAQ、MEGAN、MEIAT 模板文件
├── input/                        # 后处理 shp、字体、ISAM 输入文件
├── tools/                        # 下载、ISAM mask 等辅助脚本
└── simulations/<GridName>/       # 运行后生成的实验目录
```

典型输出目录：

```text
simulations/<GridName>/
├── flagfiles/
├── fnlfiles/
├── WPSoutput/
├── geogridinfo.txt
├── <GridName>_d01/
├── <GridName>_d02/
├── <GridName>_d03/
└── WRFoutput/
```

每个 CMAQ domain 目录下通常包含：

```text
mcip/
bcon/
icon/
emis/
cctm/
cctmCombine/ 或 POST/
```

## 模板文件说明

`namelists_cshfiles/` 中的文件会被脚本读取、修改并复制到对应模型目录运行。常用模板包括：

- `namelist.wps.temp`
- `namelist.input.temp`
- `prepmegan4cmaq.inp.temp`
- `namelist.MEIAT.temp`
- `namelist.MEIAT.Linux.temp`
- `run_mcip_daybyday_profile_WCAS.csh`
- `run_bcon_daybyday_profile_WCAS.csh`
- `run_icon_daybyday_profile_WCAS.csh`
- `run_cctm_daybyday_profile_WCAS.csh`
- `run_bcon_daybyday_regrid_WCAS.csh`
- `run_icon_daybyday_regrid_WCAS.csh`
- `run_cctm_daybyday_regrid_WCAS.csh`
- `run_combine_WCAS.csh`
- `run_prepmegan4cmaq_WCAS.csh`
- `run_megan_WCAS.csh`

修改这些模板前，应先确认相同参数在手动模型运行中可用。WCAS 只负责自动替换关键变量，物理方案、化学机制、排放清单路径和编译目录仍需与本机模型环境一致。

## 使用注意事项

- `GridName` 决定输出目录与断点续跑状态。重复使用同一个 `GridName` 会复用已有 flag 文件。
- 修改 `namelist.WCAS` 后，如果希望重新运行已经完成的步骤，需要同步删除对应 flag 文件。
- `max_dom`、`regrid_dom`、`regrid_D02andD03` 必须互相一致。例如 `regrid_dom` 不能大于 `max_dom`。
- `ManualGrid = .true.` 时，脚本不会使用 `&grid` 的经纬度范围，而是直接使用 `&manualgrid`。
- `direct_linkreanalysis = .true.` 时，脚本默认从 `direct_linkreanalysis_dir` 复制 FNL 文件；需确保文件名与下载逻辑生成的列表一致。
- `runISAM = .true.` 需要额外准备 ISAM 控制文件、区域文件与对应 CMAQ-ISAM 编译版本。
- 当前脚本会在模型目录中清理部分临时文件，例如 WPS 的 `met_em*`、`FILE*`、`GRIBFILE*`，以及 WRF `run/` 中的 `wrfinput*`、`wrfbdy*`、`rsl.*` 等。运行前请确认这些目录没有需要保留的同名文件。
- `namelist.WCAS` 中如包含服务器密码或私有路径，应避免提交到公共版本库。

## 常见问题

### WRF 运行后没有输出到目标日期

检查 `WRF_dir/run/rsl.*`、`namelist.input.temp`、FNL/GFS 文件完整性和 `num_metgrid_levels`。不同年份或资料源的 metgrid 垂直层数可能不同，必要时需要调整 WRF namelist 模板。

### geogrid 范围不合适

设置 `testGeogrid = .true.` 后重新运行。脚本会输出 geogrid 检查图，可根据图像修改 `&grid` 或 `&manualgrid` 后选择重新运行 geogrid。

### CMAQ regrid 输出缺失

确认外层 CCTM 是否输出了目标 regrid 所需的 `CONC` 文件。连续 regrid 时，脚本会在外层使用 `CONC_level = ALL`，但仍需检查 CCTM 模板和磁盘空间是否正常。

### 只想运行 WRF

设置：

```fortran
runCMAQ = .false.
```

此时脚本不会进入 MCIP、CMAQ、排放与后处理流程。

## 版本说明

本文档按 `WCAS_codex.py` 的当前流程整理。仓库中保留的 `WCAS.py`、`WCAS_subprocess.0812.py`、`WCAS_terminal.py` 等文件可作为历史实现参考，但日常使用建议以 `WCAS_codex.py` 与当前 `namelist.WCAS` 为准。
