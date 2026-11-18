# 第三阶段模型与输出数据字典

## 回测预测 `predictions.parquet`

每行是一个曼哈顿站点 UTC 小时。`hour_utc` 是唯一键；`hour_local` 仅用于解释。`ride_out`、`ride_in` 为实际订单量，`net_outflow = ride_out - ride_in`。`<model>_ride_out`、`<model>_ride_in` 是相应模型预测；`<model>_net_outflow` 为两者之差。`days_from_thanksgiving` 及节假日标记来自日历。天气列由固定 48 小时提前量的 Open-Meteo 历史预报覆盖。重复的夏令时回拨小时未进入评价。

## 指标文件

- `model_metrics.csv`：`model`、`scenario`、`target`、`rows`、`mae`、`rmse`、`wape`。
- `station_metrics.csv`：模型和站点级骑出/骑入 MAE、WAPE 与净流量 MAE。低活动站点 WAPE 可能极大。
- `topk_metrics.csv`：每个 UTC 小时按预测绝对净流出排序，和实际 Top K 的平均交集比例。
- `ablation_results.csv`：每个模型相对历史轮廓的 WAPE 改善，正值代表改善。
- `panel_*_coefficients.csv`：目标、变量、系数、按本地日期聚类稳健标准误、p 值、样本数和残差 R²。

## 模型权重和元数据

- `ann_full.pt`：PyTorch `state_dict`；站点嵌入 16，隐藏层 128/64，两个 Softplus 计数输出。
- `gru_exogenous.pt`：PyTorch `state_dict`；24 小时外生序列，隐藏维 64，两个 Softplus 输出。
- `metadata.json`：训练边界、样本数、站点映射、输入字段、标准化均值/尺度、训练轮次、损失轨迹、PyTorch/CUDA 版本。
- `profile_*.parquet`：只由训练期形成的站点季节平均和回退轮廓。
- `panel_weather_holiday_coefficients.csv`：最终面板修正参数。

## 固定日期预测 CSV

主键是 `station_id + hour_utc`。

| 字段 | 含义 |
|---|---|
| `station_id`, `station_name` | 官方站点编号和当前名称 |
| `latitude`, `longitude`, `capacity` | 当前站点属性；容量未知时保留缺失 |
| `hour_utc` | 唯一 UTC 小时键 |
| `hour_local`, `utc_offset_minutes` | America/New_York 本地小时及偏移 |
| `predicted_ride_out`, `predicted_ride_in` | ANN full 的非负小时需求预测 |
| `predicted_net_outflow` | 骑出预测减骑入预测 |
| `*_lower`, `*_upper` | 由 2025 感恩节窗口绝对误差 90% 分位构造的经验带，下限截为零；不是正式 conformal 保证 |
| 天气列 | 文件清单中指定版本的 Open-Meteo 预报 |
| `model_version`, `forecast_status` | 模型版本与 preliminary/final 状态 |

`manifest.json` 记录生成时间、训练范围、天气源及取得时间、模型选择规则、站点数、行数和不确定性方法。`top_pressure.csv` 按每站最大绝对预测净流出整理，只是提前需求压力，不是基于实时库存的风险。

## 库存风险

`risk_predictions.parquet` 每行是一个可评价的一小时预测起点和下一小时标签。`origin_*` 字段来自前一完整小时；`target_empty_risk`/`target_full_risk` 表示下一小时至少一个合格分钟为空/满。质量规则要求起点和目标均至少 45 个观测分钟、45 个运行中分钟且容量为正。

`risk_metrics.csv` 中 precision、recall、F1、TP、FP、FN、TN 是 2026-10 严格留出集结果；Logistic 另有 ROC AUC。`*.joblib` 保存标准化与 Logistic 模型。

实时风险 CSV 与固定日期预测严格分开：它包含最新可得库存、基于历史需求轮廓的一小时净流出压力和空/满概率。其来源时间必须从相邻 JSON 清单读取；过期快照不能称为当前状态。

## 缺失与适用范围

- 预测评价只含正式面板中运营期有效站点小时；新站在训练期无记录时必须走未知站点/轮廓回退，不得补造历史。
- 固定日期输出使用 2026-10 当前站点清单，因此是 686 个站点；训练元数据的 828 个 ID 包含历史停用、迁移标识及系统/实验 ID，并不等于目标日运营站数。
- 预测天气可以在预报版本中真实缺失；生成器对模型必需字段执行完整性检查，不填造未来真实天气。
- 2026 感恩节实际订单尚不可得；任何 `preliminary` 文件都不是实际验证结果。

