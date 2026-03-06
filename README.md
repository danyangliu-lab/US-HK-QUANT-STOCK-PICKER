# 成长+趋势量化选股系统（美股/港股）

日更量化选股系统，适配**激进成长+趋势动量、高弹性偏好、中线持有**的投资风格。系统优先选择趋势强劲、成长性好的高弹性标的，估值容忍度高，追求进攻性收益。

---

## 1. 项目结构

```text
.
├── .env                          # 密钥配置（自动加载，不进git）
├── run_daily.py                  # 主入口（控制台输出 + 参数解析）
├── requirements.txt
├── cache/                        # 当日数据缓存（自动生成，按 ticker 列表 MD5 + 日期命名）
├── outputs/                      # 每日输出（CSV 信号表 / 权重表 / 差异报告 / 持仓建议 / 持仓历史）
└── quant_system/
    ├── config.py                 # 策略参数、自选观察池、小盘投机池、持仓池、行业映射、杠杆ETF映射
    ├── data.py                   # 行情 + 基本面 + VIX恐慌指数下载（yfinance + 缓存 + 逐只重试）
    ├── scoring.py                # 三引擎打分（成长+趋势/小盘投机/杠杆）+ 15个技术因子 + 线性插值 + 引擎内独立幂次拉伸
    ├── sentiment.py              # 分析师评级量化（yfinance 评级共识 + 目标价上涨空间 + 近期升降级 → 个股调节）
    ├── institutional.py          # 机构持仓+做空数据（yfinance 机构占比 + 做空比例 + 内部人交易 → 个股调节）
    ├── news.py                   # 新闻抓取（Google News RSS → 全局/个股新闻 → 注入LLM prompt）
    ├── llm.py                    # 三模型AI辅助评分（Kimi K2.5 + DeepSeek V3.2 + GLM-5 交叉验证）+ 持仓投资建议 + 持仓整体组合分析
    ├── dcf.py                    # DCF 折现现金流估值（CAPM→WACC→5年FCF预测→终值→三场景→敏感性矩阵→评分融合）
    ├── report.py                 # 格式化研报生成（投资摘要→个股分析→DCF汇总→Top推荐→组合分析→风险提示，支持LLM增强）
    ├── moomoo_sync.py            # moomoo (富途) 持仓/自选股同步（通过 futu-api 连接 OpenD 网关）
    ├── tracker.py                # 组合收益跟踪（持仓快照 + 区间收益 + 累计收益 + 基准指数对比）
    └── engine.py                 # 主流水线（数据→打分→分析师评级→机构/做空→DCF估值→新闻→LLM融合→选股→权重→持仓建议→整体分析→研报→收益跟踪(含基准)→输出）
```

---

## 2. 股票池

系统维护两大类股票池：**自选观察池**（用于筛选/排名）和**持仓股票池**（当前实际持仓，额外生成投资建议）。

### 2.1 自选观察池

| 池子 | 引擎 | 标的 |
|------|------|------|
| 成长池 | 成长+趋势引擎 | 所有个股统一使用成长+趋势逻辑评估（具体列表见 `config.py`） |
| 小盘投机池 | 小盘投机引擎 | 市值<10亿美金的小盘股，小仓位高赔率（具体列表见 `config.py → smallcap_tickers`） |
| 杠杆ETF | 杠杆引擎 | 2x/3x 杠杆 ETF（具体列表见 `config.py`） |

> 另外自动加入参考指数 ETF 用于杠杆引擎的指数技术面分析，不参与打分。

### 2.2 持仓股票池

| 池子 | 引擎 | 标的 |
|------|------|------|
| 持仓个股 | 成长+趋势引擎 | 当前实际持有的个股（具体列表见 `config.py → portfolio_stock_tickers`） |
| 持仓杠杆ETF | 杠杆引擎 | 当前实际持有的杠杆 ETF（具体列表见 `config.py → portfolio_leverage_tickers`） |

- 持仓池与自选池**共享数据下载**（`all_tickers` 自动去重），但**独立打分**
- **持仓股自动加入自选池**：`portfolio_stock_tickers` 自动合并到成长池（`watchlist_growth_tickers`），`portfolio_leverage_tickers` 自动合并到杠杆池（`watchlist_leverage_tickers`），确保所有持仓股参与自选池的完整评分流程，无需手动重复维护
- 持仓池除计算量化分数外，还通过 LLM 生成每只股票的**投资建议**（加仓/持有/减仓/清仓）和**整体组合分析**（风险/调仓/关注点/总仓位建议）
- 输出保存至 `portfolio_advice_YYYYMMDD.csv` 和 `portfolio_overall_YYYYMMDD.txt`

### 2.3 moomoo (富途) 自动同步

支持通过 moomoo OpenAPI 自动同步持仓和自选股，无需手动维护股票列表：

- **持仓同步**：从 moomoo 账户读取美股/港股实际持仓 → 自动覆写 `portfolio_stock_tickers` + `portfolio_leverage_tickers`
- **自选股同步**：从 moomoo 自选股分组读取标的 → 自动覆写 `growth_tickers` / `leverage_etf_tickers` / `smallcap_tickers`
- 新出现在持仓中但不在自选池的股票会**自动加入观察池**（确保行情数据被下载）
- 代码格式自动转换：`US.NVDA` → `NVDA`，`HK.00700` → `0700.HK`
- 需要本地运行 FutuOpenD 网关并登录 moomoo 账户

使用方式：`python3 run_daily.py --llm --sync-moomoo`（详见第15节）

---

## 3. 投资风格：激进高弹性

系统针对激进投资风格做了全面优化：
- **趋势为王**：趋势权重35%，强势股获得明显加分，引入风险调整动量和收益率偏度
- **基本面加强**：基本面成长权重30% + 盈利质量8%（ROE + 负债率），兼顾增速与财务健康
- **量价信号**：新增量价背离、换手率因子，捕捉量价配合/背离信号
- **市场情绪**：引入VIX恐慌指数全局调节，恐慌期压分、贪婪期加分
- **社交情绪**：分析师评级量化（评级共识分 + 目标价上涨空间 + 近期升降级），个股级别加减分
- **主力资金**：机构持仓占比+做空压力+内部人交易，高机构持仓加分、高做空扣分
- **容忍高估值**：估值权重仅12%，PE/PS 阈值放宽，不因高估值误杀成长股
- **拥抱波动**：仓位分配弱化波动率反比惩罚，高弹性股票不被压低仓位
- **统一引擎**：所有个股（含大市值）统一使用成长+趋势引擎，不再区分蓝筹
- **放宽风控**：单票≤12%、杠杆ETF≤10%、杠杆ETF总计≤25%、小盘投机单票≤6%、小盘投机总计≤15%、单行业≤40%
- **惩罚宽松**：估值过热惩罚阈值提高（PE>150才扣分），软惩罚力度降低

---

## 4. 完整流水线

运行 `python3 run_daily.py [--llm] [--report] [--sync-moomoo]` 时，依次执行：

```
0. moomoo 同步（可选，--sync-moomoo）
   ├─ 连接本地 OpenD 网关
   ├─ 同步美股/港股实际持仓 → 覆写持仓池
   ├─ 同步自选股分组 → 覆写自选观察池
   └─ 代码格式自动转换（US.NVDA→NVDA, HK.00700→0700.HK）

1. 数据下载 (data.py)
   ├─ yf.download() 批量拉取行情（收盘价 + 成交量）
   ├─ 失败的 ticker 逐只用 yf.Ticker().history() 重试（最多2次）
   ├─ 逐只拉取基本面（PE/PS/ROE/营收增速/现金流/负债率等17个字段）
   ├─ 下载 VIX 恐慌指数（^VIX），与行情时间对齐
   └─ 当日缓存到 cache/ （按 ticker 列表 MD5 + 日期命名）

2. 技术特征计算 (scoring.py → compute_tech_features)
   ├─ 15个技术/量价因子（详见下文）
   └─ 输出 DataFrame，行=ticker，列=特征

3. 三引擎打分 + 引擎内独立拉伸 (scoring.py)
   ├─ 成长+趋势引擎 → score_growth_engine()（自动包含持仓个股）
   ├─ 小盘投机引擎 → score_smallcap_engine()
   ├─ 杠杆引擎 → score_leverage_engine()（自动包含持仓杠杆ETF）
   ├─ 每个引擎内部独立幂次拉伸 → merge_scores()
   └─ VIX 情绪因子全局调节（恐慌期压分/贪婪期加分）

3.5. 分析师评级量化 (sentiment.py)
   ├─ 从 yfinance 获取每只标的的分析师推荐汇总（strongBuy/buy/hold/sell/strongSell）
   ├─ 获取分析师目标价均值，计算隐含上涨空间
   ├─ 获取近30天升降级事件（upgrade/downgrade）
   ├─ 综合评分 analyst_score = 共识50% + 目标价30% + 升降级20%
   ├─ 个股级别调节 quant_score（最多 ±4 分）
   └─ 当日缓存到 cache/analyst_YYYYMMDD.pkl

3.6. 机构持仓与做空因子 (institutional.py)
   ├─ 从 yfinance 获取机构持仓占比（heldPercentInstitutions）
   ├─ 获取做空数据（做空占流通股比例 + 做空月度变化 + 空头回补天数）
   ├─ 获取近90天内部人交易（买入/卖出次数）
   ├─ 综合因子分 inst_score = 机构持仓40% + 做空压力35% + 做空变化15% + 内部人10%
   ├─ 个股级别调节 quant_score（最多 ±3 分）
   └─ 当日缓存到 cache/institutional_YYYYMMDD.pkl

3.7. DCF 估值 (dcf.py)
   ├─ 为非杠杆 ETF 个股计算 DCF 内在价值
   ├─ WACC 估算（CAPM: Rf + β × ERP + 债务成本加权）
   ├─ 5年 FCF 预测（历史 FCF × 衰减增长率）
   ├─ Bear / Base / Bull 三场景（scenario_mult = 0.6 / 1.0 / 1.4）
   ├─ 终值（永续增长法，年中惯例折现）
   ├─ WACC × 永续增长率 5×5 敏感性矩阵
   ├─ DCF 评分（上涨空间 → 0~100 分）→ 轻度调节 quant_score（±5分）
   └─ DCF 结果注入 LLM prompt（增强估值分析深度）

4. 新闻抓取 (news.py, 仅 --llm 启用时)
   ├─ 从 Google News RSS 抓取 6 类全局财经/地缘政治新闻
   ├─ 为每只股票/ETF 抓取相关新闻（普通股按名称、杠杆ETF按行业关键词）
   └─ 当日缓存，避免重复请求

5. LLM 辅助评分 (llm.py, 可选)
   ├─ 将实时新闻注入每只股票的 prompt
   ├─ 三模型并行调用：Kimi K2.5 + DeepSeek V3.2 + GLM-5（均通过腾讯云 lkeap /v3 接口）
   ├─ 交叉验证融合：多数方向一致→增强信心，分歧→取保守值
   └─ 返回 event_score [-1,1] + risk_flag

6. 分数融合 (engine.py → _apply_llm_fusion)
   └─ final_score = 0.8 × quant_score + 0.2 × LLM拉伸分

7. 信号生成 + 权重分配 (engine.py)
   ├─ BUY / HOLD / REDUCE 标记
   ├─ 目标权重计算（含风控约束）
   └─ 与昨日信号对比生成 diff

8. 组合收益跟踪 (tracker.py)
   ├─ 持仓快照 → 区间收益 → 累计收益
   └─ 基准指数对比（纳斯达克100 QQQ / 标普500 SPY）→ 超额收益计算

9. 持仓股票池独立评分 + 投资建议 + 整体分析 (engine.py + llm.py)
   ├─ 持仓个股 → 成长引擎打分，持仓杠杆ETF → 杠杆引擎打分
   ├─ 持仓个股 DCF 估值（补充主池未覆盖的标的）
   ├─ LLM event_score 融合（三模型交叉验证）
   ├─ LLM 生成投资建议（三模型交叉验证：加仓/持有/减仓/清仓 + 信心度 + 理由）
   ├─ LLM 生成整体组合分析（三模型各自独立分析，汇总为统一报告：组合总评/风险提示/调仓建议/关注要点/总仓位建议）
   └─ 输出 portfolio_advice_YYYYMMDD.csv + portfolio_overall_YYYYMMDD.txt

10. 研报生成（可选，--report）(report.py)
   ├─ 基础版：纯数据驱动的 Markdown 研报
   │   ├─ 投资摘要（池规模/买卖信号/前三大仓位/DCF低估高估标的/总仓位建议）
   │   ├─ 持仓个股分析（评分/信号/DCF/AI建议）
   │   ├─ DCF 估值汇总表（现价/Bear/Base/Bull/上涨空间/WACC）
   │   ├─ 自选池 Top 推荐（排名/评分/目标仓位/DCF上涨）
   │   ├─ 组合整体分析
   │   └─ 风险提示
   ├─ LLM增强版（--llm --report）：额外调用大模型为每只持仓生成 100-200 字深度分析
   └─ 输出 research_report_YYYYMMDD.md + dcf_valuation_YYYYMMDD.csv

11. 输出 CSV + 控制台展示
```

---

## 5. 技术特征计算

`compute_tech_features()` 基于每日收盘价、成交量和基本面数据，计算以下 15 个特征：

| 特征 | 计算方式 | 含义 |
|------|---------|------|
| `ret_5` | `pct_change(5)` | 5日涨跌幅 |
| `ret_20` | `pct_change(20)` | 20日涨跌幅（月线动量） |
| `ret_60` | `pct_change(60)` | 60日涨跌幅（季线动量） |
| `above_ma20` | `latest > MA20` | 是否站上20日均线（0/1） |
| `above_ma60` | `latest > MA60` | 是否站上60日均线（0/1） |
| `ma20_slope` | `MA20.pct_change(5)` | MA20近5日变化率（均线斜率） |
| `breakout_ma60` | 昨天低于MA60、今天站上 | MA60突破信号（0/1） |
| `dist_52h` | `latest / 52周高点 - 1` | 距52周高点距离（负值=回撤深度） |
| `drawdown_recovery` | `latest / 20日低点 - 1` | 从近20日低点反弹幅度 |
| `vol_20` | `daily_ret.rolling(20).std() × √252` | 20日年化波动率 |
| `vol_ratio` | `today_volume / MA20_volume` | 量比（放量程度） |
| `pv_divergence` | 价格与量配合/背离 | 量价背离（+1=放量新高, -1=顶背离, -0.5=恐慌抛售） |
| `risk_adj_mom` | `ret_20 / vol_20` | 波动率调整动量（类夏普比率，风险调整后的收益质量） |
| `ret_skew` | `daily_ret.rolling(20).skew()` | 收益率偏度（正偏=右尾爆发潜力，负偏=左尾风险） |
| `turnover` | `avg_vol_5 / shares_outstanding` | 日均换手率（成交量/流通股数，衡量交易活跃度） |

---

## 6. 三引擎评分逻辑

### 6.1 打分方法：线性插值

所有因子均采用 `_abs_score()` 函数打分——给定一组 `(阈值, 分值)` 锚点，使用 `np.interp` 在锚点间**线性插值**，而非传统阶梯函数。这确保分数平滑过渡、最大程度拉开差距。

示例：`ret_20` 的打分锚点为 `[(-0.15, 5), (-0.05, 18), (0.0, 38), (0.03, 58), (0.08, 80), (0.15, 100)]`
- 20日涨幅 -15% → 5分，0% → 38分，+8% → 80分，+15% → 100分
- 中间值如 +5% → 线性插值得 ≈67分

### 6.2 成长+趋势引擎 (`score_growth_engine`)

适用于所有个股（含大市值），统一以成长+趋势逻辑评估。

| 大类 | 权重 | 子因子及权重 |
|------|------|------------|
| **趋势动量** | **35%** | ret_5(18%) + ret_20(20%) + ret_60(15%) + MA20斜率(10%) + 站上MA20(7%) + 站上MA60(7%) + MA60突破(6%) + 风险调整动量(10%) + 收益率偏度(7%) |
| **基本面成长** | **30%** | 营收增速(35%) + 盈利增速(20%) + 毛利率(20%) + 自由现金流(25%) |
| **盈利质量** | **8%** | ROE(60%) + 负债率(40%) |
| **资金行为** | **15%** | 量比(25%) + 52周高点距离(25%) + 回撤修复(20%) + 量价背离(15%) + 换手率(15%) |
| **估值约束** | **12%** | Forward PE(40%) + Trailing PE(35%) + PS(25%) |

**额外机制：**
- **软惩罚**（降低力度）：营收增长<0 扣5分、毛利率<0 扣4分、PE>150 扣4分、PS>50 扣3分、负债率>300% 扣3分
- **硬过滤**：仅在营收<-30% **且** 毛利率<-10% 的极端情况排除（NaN不惩罚，`fillna(0)`）

### 6.3 小盘投机引擎 (`score_smallcap_engine`)

专为市值<10亿美金的小盘股设计，追求小仓位高赔率，完全拥抱风险。

| 大类 | 权重 | 子因子及权重 |
|------|------|------------|
| **趋势动量** | **40%** | ret_5(22%) + ret_20(20%) + ret_60(15%) + MA20斜率(10%) + 站上MA20(8%) + MA60突破(7%) + 风险调整动量(10%) + 收益率偏度(8%) |
| **爆发力** | **20%** | 5日爆发力(35%) + 量比异常放量(30%) + 20日低点反弹(35%) |
| **资金异动** | **20%** | 量比持续放量(22%) + 距52周新高(28%) + 回撤修复V反(20%) + 量价背离(15%) + 换手率(15%) |
| **基本面** | **20%** | 营收增速(65%) + 毛利率(35%)，不看盈利/现金流/估值 |

**与成长引擎的关键差异：**
- **趋势权重更高**（40% vs 35%）：投机趋势第一
- **新增爆发力维度**（20%）：短期暴涨、异常放量、V型反转是核心信号
- **基本面适度提权**（20% vs 30%）：但只看营收增速，不看盈利质量
- **完全去掉估值和盈利质量**（0% vs 20%）：投机不看PE/PS/ROE
- **波动率奖励**：高波不惩罚反而加分（+2~+12分），投机就要高波
- **极轻微惩罚**：仅营收<-30%或毛利<-30%时扣3分
- **无硬过滤**：投机股什么基本面都可能

### 6.4 杠杆ETF引擎 (`score_leverage_engine`)

根据 ETF 是否有底层股票映射或参考指数映射，采用不同权重组合：

| 类型 | 示例 | ETF趋势 | 底层技术面 | 底层基本面 | 参考指数技术面 | 波动率 |
|------|------|---------|-----------|-----------|--------------|--------|
| 有底层股票 | ETF→底层个股（见 `config.py` 映射表） | 45% | 20% | 20% | - | 15% |
| 有参考指数 | ETF→参考指数ETF（见 `config.py` 映射表） | 50% | - | - | 30% | 20% |
| 无参考 | （新增无映射的杠杆ETF） | 80% | - | - | - | 20% |

- **ETF自身趋势**：ret_5(30%) + ret_20(30%) + 站上MA20(20%) + MA20斜率(20%)
- **底层技术面**：底层股票的 ret_20/ret_60/MA60位置/MA20斜率
- **底层基本面**：底层股票的 营收增速/盈利增速/毛利率/Forward PE
- **参考指数技术面**：参考指数 ETF 的 ret_5/ret_20/ret_60/MA60位置/MA20斜率
- **波动率**：年化波动率 0.2→90分, 0.8→28分, 1.0→10分（激进风格，高波仅轻度惩罚）

---

## 7. 引擎内独立幂次拉伸（对抗分数收敛）

多因子加权平均天然会让分数向中间段收敛（大部分股票集中在50~70分），不利于决策。

### 7.1 问题：引擎间系统性偏差

三个引擎的原始分数分布有显著差异：
- **成长引擎**：因子值波动大（有的爆发增长、有的亏损），原始分分布广，15~85 都有
- **小盘引擎**：波动率奖励加分，加上爆发力因子，分数波动更大
- **杠杆引擎**：波动率高，原始分偏低

如果在三引擎合并后统一做幂次拉伸，某个引擎的系统性高分会被进一步放大，导致组合偏向某一类标的。

### 7.2 解决方案：引擎内独立拉伸

系统在 `merge_scores()` 中，**先对每个引擎的 `quant_score` 独立做幂次拉伸，再合并**：

```python
# merge_scores() 中的关键逻辑
stretched_dfs = []
for df in [growth_df, leverage_df]:
    d = df.copy()
    d["quant_score_raw"] = d["quant_score"].copy()       # 保留原始分用于诊断
    d["quant_score"] = _stretch_scores(d["quant_score"], power=1.8)  # 引擎内独立拉伸
    stretched_dfs.append(d)
merged = pd.concat(stretched_dfs, axis=0)                # 合并双引擎
```

### 7.3 `_stretch_scores` 拉伸函数

```
步骤：
1. 归一化到 [0, 1]：normed = (score - min) / (max - min)   ← 在本引擎内归一化
2. 幂次变换：stretched = normed ^ 1.8
3. 映射回 [0, 100]：final = stretched × 100
```

### 7.4 效果

- **每个引擎的第一名都能拿到接近 100 的高分**，不受其他引擎分布影响
- 成长引擎内的强势股和杠杆引擎内的强势标的站在同一起跑线
- power > 1 使得高分被拉高、低分被压低，分数分布从密集区展开
- 原始分数保留在 `quant_score_raw` 列用于诊断

---

## 8. 新闻抓取模块

`news.py` 从海外主流媒体自动抓取实时新闻，注入 LLM prompt，让大模型基于**真实时事**而非训练数据做出判断。

### 8.1 数据源

- **Google News RSS**（免费，无需 API Key）
- 聚合 Reuters、Bloomberg、CNBC、BBC、AP 等主流媒体
- 按关键词搜索，返回最新新闻标题

### 8.2 全局新闻

系统抓取 6 类全球宏观/地缘政治关键词的最新新闻：

| 关键词 | 覆盖方向 |
|--------|---------|
| `stock market today` | 全球股市动态 |
| `geopolitics war conflict` | 地缘政治/战争冲突 |
| `Federal Reserve interest rate` | 美联储利率政策 |
| `US China trade tariff` | 中美贸易/关税 |
| `oil price OPEC` | 油价/OPEC |
| `semiconductor chip AI` | 半导体/AI行业 |

去重后保留最多 30 条全局新闻标题。

### 8.3 个股新闻

针对每只股票/ETF 分别抓取相关新闻：

- **普通股票**：使用 `{公司名} stock` 搜索（如 `XXX Inc stock`）
- **杠杆ETF**：根据行业描述匹配关键词搜索：
  - 半导体类 → `semiconductor chip industry news`
  - 航空国防类 → `defense military geopolitics news`
  - 纳指类 → `Nasdaq 100 tech stocks news`
  - 恒指类 → `Hong Kong stock market news`
- 每只标的保留最多 6 条新闻标题

### 8.4 缓存策略

- 当日缓存到 `cache/news_cache.pkl`
- 同一天多次运行只抓取一次，避免触发频率限制
- 次日自动失效，确保新闻时效性

### 8.5 Prompt 注入格式

```
【近期全球财经/地缘政治要闻】
1. Fed holds interest rates steady amid tariff uncertainty
2. Israel strikes Iran nuclear facilities, oil prices surge
...

【XXXX 相关新闻】
1. Semiconductor stocks rally on AI demand outlook
2. TSMC reports record revenue as chip demand surges
...
```

---

## 9. LLM辅助评分（三模型交叉验证）

可选启用AI大模型（`--llm`），对每只股票独立生成事件评分。支持 **Kimi K2.5 + DeepSeek V3.2 + GLM-5 三模型交叉验证**，三个模型独立分析同一只股票，通过多数投票和交叉验证增强信号可靠性。

### 9.1 核心原则：正交互补 + 交叉验证

- **LLM 不接收任何量化分数**，完全基于公司背景、实时新闻、宏观环境独立判断
- 量化引擎擅长「价量趋势 + 财务数据」，LLM 擅长「新闻事件 + 宏观研判」，两者正交互补
- **新闻注入**：系统自动抓取最新新闻标题注入 prompt，解决大模型训练数据时效性不足的问题
- **三模型交叉验证**：Kimi K2.5、DeepSeek V3.2、GLM-5 各自独立分析，结果融合时根据一致性调整信心度

### 9.2 三模型架构

| 模型 | 接口 | 模型参数名 | 特点 |
|------|------|-----------|------|
| **Kimi K2.5** | 腾讯云 lkeap /v3 OpenAI 兼容接口 | `kimi-k2.5` | 256k 超长上下文，支持 thinking 模式 |
| **DeepSeek V3.2** | 腾讯云 lkeap /v3（或 /v1 / SDK） | `deepseek-v3.2` | 671B MoE 模型，支持联网搜索 |
| **GLM-5** | 腾讯云 lkeap /v3 OpenAI 兼容接口 | `glm-5` | 200k 上下文，支持 thinking + Function Calling |

**统一调用方式**：三个模型均通过腾讯云 lkeap `/v3` 端点 (`https://api.lkeap.cloud.tencent.com/v3`) 调用，共用同一个 `LKEAP_API_KEY`。DeepSeek 额外兼容 `/v1` 端点和 SDK 调用作为备用路径。

### 9.3 交叉验证策略

每只股票分别发送给三个模型，得到三组独立的 `event_score` 和 `risk_flag`，然后按以下策略融合：

| 场景 | 处理方式 |
|------|---------|
| **3/3 全体一致**（全看多或全看空） | 强力增强：均值 × 1.15，强信号额外 × 1.05 |
| **2/3 多数一致** | 温和增强：均值 × 1.05 |
| **三方分歧** | 保守处理：取绝对值最小的评分 × 0.70 |
| **风险标记** | 任一模型标记风险即标记（宁可错杀不可放过） |

支持三种融合模式（通过 `CROSS_VALIDATION_MODE` 环境变量或 `--cross-mode` 参数配置）：

| 模式 | 说明 |
|------|------|
| `cross`（默认） | 方向融合：多数一致增强、分歧保守 |
| `avg` | 简单平均三个模型的评分 |
| `primary` | 仅用第一个可用模型评分 |

### 9.4 Prompt 设计

**个股 prompt**（传入公司信息 + 实时新闻，不传量化分数）：
- 股票代码、名称、行业、市值、营收增速、毛利率、年化波动率
- 全局财经/地缘政治新闻标题（最多10条）
- 该股票相关新闻标题（最多5条）
- 要求结合真实新闻分析：行业影响、地缘政治、宏观环境、公司事件
- 输出：`{"event_score": -1~1, "risk_flag": 0/1, "reason": "..."}`

**杠杆ETF 定制 prompt**（传入 ETF 行业描述 + 实时新闻）：
- ETF 代码、产品描述（如 "3倍做多半导体指数ETF，受AI芯片需求、中美科技竞争影响"）
- 全局及行业相关新闻标题
- 要求结合真实新闻分析：地缘政治、宏观政策、行业周期、突发事件
- 特别关注杠杆ETF特有风险（波动率衰减、持仓成本）

### 9.5 融合公式

```
raw_llm ∈ [-1, 1] → 归一化 [0, 1] → 幂次拉伸 (power=2.0) → 映射 [0, 100]

final_score = 0.8 × quant_score + 0.2 × llm_stretched
若 risk_flag = 1，额外扣 10 分
```

LLM 的 event_score 也做了幂次拉伸（power=2.0），放大正面/负面评价之间的差距，避免 LLM 分数集中在小范围内对排序无影响。

### 9.6 调用方式

三个模型统一通过腾讯云 lkeap `/v3` OpenAI 兼容接口调用，只需一个 `LKEAP_API_KEY`：

1. **推荐方式（/v3 统一接口）**：在[腾讯云控制台 API Key 管理](https://hunyuan.cloud.tencent.com/#/app/apiKeyManage)创建 API Key，配置 `LKEAP_API_KEY` 即可同时驱动三个模型
2. **DeepSeek 备用路径1**：配置 `DEEPSEEK_API_KEY` 走 lkeap `/v1` 端点
3. **DeepSeek 备用路径2**：配置 `TENCENT_SECRET_ID` + `TENCENT_SECRET_KEY` 走 lkeap SDK

JSON 解析兼容 thinking 模型的 `<think>...</think>` 输出格式。

### 9.7 持仓投资建议 (`batch_portfolio_advice`)

针对持仓股票池中的每只标的，调用三模型生成具体操作建议并交叉验证。

**与事件评分的区别：**
- 事件评分 prompt **不包含**量化分数（正交互补原则）
- 投资建议 prompt **包含**量化趋势分/总分/最终得分/系统信号（综合判断）

**投资建议交叉验证策略：**

| 场景 | 处理方式 |
|------|---------|
| 三模型建议**完全一致** | 信心度增强15%（共识加成） |
| **多数一致**（2/3 相同） | 取多数意见，信心度取平均 |
| **三方分歧**（三个不同建议） | 取最保守建议，信心度大幅打折 |

**Prompt 输入：**
- 股票基本面：行业、市值、营收增速、毛利率、PE
- 量化分数：trend_score / quant_score / final_score / action
- 实时新闻：全局 + 个股相关新闻
- 杠杆ETF：额外提醒波动率衰减和持仓成本风险

**输出格式：**
```json
{"action": "持有", "confidence": 0.7, "reason": "详细中文理由（2-3句话）"}
```

- `action`：加仓 / 持有 / 减仓 / 清仓
- `confidence`：0~1 信心度
- `reason`：结合新闻 + 量化 + 基本面的综合理由

### 9.8 持仓整体组合分析 (`portfolio_overall_analysis`)

在逐只建议之外，对整个持仓组合进行全局分析。多模型启用时，三个模型**各自独立生成**组合分析报告，再由一个模型将所有分析**汇总整合为一份统一报告**。

**Prompt 输入：**
- 持仓概况：总标的数、引擎分布、行业分布、平均得分、平均波动率
- 逐只持仓摘要：ticker / 名称 / 引擎 / 行业 / 市值 / 营收增速 / 波动率 / 得分 / 信号 / 建议
- 近期相关新闻（前5只持仓的新闻）

**分析维度：**
- 📊 **组合总评**：整体健康度（健康/需调整/风险偏高）
- ⚠️ **风险提示**：行业集中度、杠杆敞口、波动率、相关性
- 💡 **调仓建议**：最值得加仓/减仓的标的及理由
- 🎯 **关注要点**：近期需重点关注的宏观/行业事件
- 💰 **总仓位建议**：基于宏观环境建议总仓位（100% / 75% / 50% / 25% / 空仓）

**LLM 未启用时回退：** 基于纯量化数据生成简要统计（持仓数/平均得分/BUY·REDUCE信号数/行业集中度/杠杆占比）。

**输出格式：**
```json
{"summary":"一句话总评","risk":"风险分析","rebalance":"调仓建议","watchlist":"关注要点","position":"75%","position_reason":"理由"}
```

---

## 10. DCF 估值模块

`dcf.py` 为每只非杠杆 ETF 个股计算折现现金流内在价值，方法论参考 Anthropic [financial-services-plugins](https://github.com/anthropics/financial-services-plugins) DCF 模型 Skill。

### 10.1 估值流程

```
1. WACC 估算
   ├─ CAPM: Ke = Rf(4.3%) + β × ERP(5.5%)
   ├─ 债务成本: Kd = (Rf + 1.5%) × (1 - 税率21%)
   └─ WACC = E/(E+D) × Ke + D/(E+D) × Kd（限制在 4%~25%）

2. FCF 预测（5年）
   ├─ 基于最新年度 FCF 外推
   ├─ 增长率逐年衰减: g_t = g_0 × 0.80^t
   ├─ 三场景: Bear(×0.6) / Base(×1.0) / Bull(×1.4)
   └─ 负 FCF 公司: 假设逐年改善

3. 终值
   └─ 永续增长法: TV = FCF_5 × (1+g) / (WACC-g)，g=2.5%

4. 折现（年中惯例）
   ├─ PV = Σ FCF_t / (1+WACC)^(t+0.5)
   └─ PV_terminal = TV / (1+WACC)^(N-0.5)

5. 每股价值
   └─ (EV - 净债务) / 稀释股数
```

### 10.2 DCF 评分映射

DCF 上涨空间（base 场景 vs 当前价）映射到 0~100 评分：

| 上涨空间 | -50% | -20% | 0% | +20% | +50% | +100% |
|----------|------|------|-----|------|------|-------|
| DCF 评分 | 5 | 25 | 50 | 65 | 80 | 95 |

DCF 评分作为 `quant_score` 的轻度调节因子：`adj = (dcf_score - 50) × 0.10`，最多 ±5 分。

### 10.3 敏感性矩阵

为每只股票生成 5×5 的 WACC × 永续增长率敏感性矩阵，显示不同假设下的内在价值变化。

### 10.4 DCF 与 LLM 结合

DCF 结果以 `【DCF 估值】WACC=X%, Bear=$X, Base=$X, Bull=$X, 上涨空间=+X%` 格式注入 LLM 研报分析 prompt，增强 AI 对估值的理解深度。

---

## 11. 格式化研报生成

`report.py` 生成结构化的投资研究报告，知识框架参考 Anthropic [financial-services-plugins](https://github.com/anthropics/financial-services-plugins) 中的 Initiating Coverage 和 Earnings Analysis Skill。

### 11.1 报告结构（6 章节）

| 章节 | 内容 |
|------|------|
| **一、投资摘要** | 池规模、买卖信号数、平均评分、前三大推荐仓位、DCF 低估/高估标的、总仓位建议 |
| **二、持仓个股分析** | 每只持仓：评级(🟢🟡🔴) + 引擎/行业/市值 + 量化评分 + DCF估值 + AI建议 + 分析理由 |
| **三、DCF 估值汇总** | Markdown 表格：现价 / Bear / Base / Bull / 上涨空间 / WACC / DCF评分 |
| **四、自选池 Top 推荐** | 排名 / 股票 / 引擎 / 最终评分 / 目标仓位 / DCF上涨空间 |
| **五、组合整体分析** | 来自 LLM 的组合总评 / 风险提示 / 调仓建议 / 关注要点 / 总仓位建议 |
| **六、风险提示** | 宏观 / 地缘政治 / 杠杆 / DCF局限 / 流动性 / 模型风险 + 免责声明 |

### 11.2 两种模式

| 模式 | 触发条件 | 个股分析 |
|------|---------|---------|
| **基础版** | `--report` | 纯数据展示（评分/DCF/信号/AI建议） |
| **LLM增强版** | `--llm --report` | 额外调用大模型为每只持仓生成 100-200 字深度分析（含评级/目标价区间/催化剂/风险） |

### 11.3 输出文件

- `research_report_YYYYMMDD.md` — 完整研报（Markdown 格式）
- `dcf_valuation_YYYYMMDD.csv` — DCF 估值汇总表（CSV）

---

## 12. 选股与权重分配

### 12.1 信号生成（`engine.py`）

按 `final_score` 从高到低排序后：
- **BUY**：排名前 `top_n_buy=12` 且 `final_score ≥ 55`
- **HOLD**：其余通过硬过滤且 `final_score ≥ 45`
- **REDUCE**：硬过滤不通过 或 `final_score < 45`

### 12.2 权重分配（`_build_target_weights`）

```python
raw = max(final_score - 40, 0)                # 超额分（基准线40）
vol_adj = clip(annual_vol, 0.15, 0.80)        # 波动率裁剪
raw = raw / (0.5 + 0.5 × vol_adj)             # 弱化波动率惩罚
weight = raw / sum(raw)                        # 归一化
```

**弱化波动率**：使用 `score / (0.5 + 0.5×vol)` 而非传统 `score / vol`，确保高弹性股票不被过度压低仓位。

### 12.3 风控约束

依次执行以下约束（超限则等比缩放）：

| 约束 | 上限 |
|------|------|
| 个股单票 | ≤ 12% |
| 杠杆ETF单票 | ≤ 10% |
| 小盘投机单票 | ≤ 6% |
| 杠杆ETF总计 | ≤ 25% |
| 小盘投机总计 | ≤ 15% |
| 单行业总计 | ≤ 40% |

最后再做一次归一化确保权重之和 = 100%。

---

## 13. 组合收益跟踪

`tracker.py` 内置组合收益跟踪模块，持续监控目标仓位表现，并与基准指数（纳斯达克100 QQQ / 标普500 SPY）对比。

### 工作流程

```
每次运行：
1. 读取 portfolio_history.csv 中上一期持仓快照
2. 用当前最新价格计算上一期每个持仓的区间收益
3. 加权汇总得到本期组合收益
4. 累计收益 = (1 + 上期累计) × (1 + 本期收益) - 1
5. 获取基准指数（QQQ/SPY）最新价格，计算同期及累计涨跌幅
6. 计算超额收益 = 组合收益 - 基准指数收益
7. 记录本期新持仓快照（ticker / 权重 / 入场价格）
8. 保存到 portfolio_history.csv + benchmark_history.csv
```

### 基准指数对比

| 基准 | Ticker | 说明 |
|------|--------|------|
| 纳斯达克100 | `QQQ` | Invesco QQQ Trust，跟踪纳斯达克100指数 |
| 标普500 | `SPY` | SPDR S&P 500 ETF Trust，跟踪标普500指数 |

- 基准指数与组合使用**相同时间区间**（同一 snapshot 间隔），确保对比公平
- 累计收益同样采用链式复利公式
- 自动计算**超额收益**（本期 & 累计），直观展示组合是否跑赢大盘
- 基准价格获取：优先从已下载行情中提取，否则单独通过 yfinance 获取近5天数据

### 持久化字段

**组合历史** (`portfolio_history.csv`)：

| 字段 | 说明 |
|------|------|
| `snapshot_date` | 快照日期 |
| `ticker` | 标的代码 |
| `weight` | 目标权重 |
| `entry_price` | 入场价格（当日最新收盘价） |
| `period_return` | 本期组合加权收益率 |
| `cumulative_return` | 从首次跟踪开始的累计收益率 |

**基准历史** (`benchmark_history.csv`)：

| 字段 | 说明 |
|------|------|
| `snapshot_date` | 快照日期 |
| `ticker` | 基准指数代码（QQQ / SPY） |
| `name` | 基准名称（纳斯达克100 / 标普500） |
| `entry_price` | 基准价格（当日最新收盘价） |
| `period_return` | 本期基准涨跌幅 |
| `cumulative_return` | 基准累计涨跌幅 |

### 换仓处理

目标仓位刷新时，自动以新权重 + 新入场价格记录新快照，旧持仓的收益已在本次计算中结算。

---

## 14. 数据来源与缓存

所有行情和基本面数据来自 **Yahoo Finance**（免费，无需API Key）。新闻来自 **Google News RSS**（免费，无需API Key）。VIX 恐慌指数来自 **Yahoo Finance**（`^VIX`）。分析师评级来自 **Yahoo Finance**（免费，无需API Key）。机构持仓/做空数据来自 **Yahoo Finance**（免费，无需API Key）。

### 行情数据
- 批量调用 `yf.download()` 拉取收盘价 + 成交量（约320个交易日）
- 批量下载失败的 ticker 会逐只用 `yf.Ticker().history()` 重试（最多2次，间隔1秒）
- 重试时自动处理时区（`tz_localize(None)`），确保与批量下载的 tz-naive 索引兼容

### 基本面数据
- 逐只调用 `yf.Ticker().info` 拉取 17 个字段：
  - 市值、PE(trailing/forward)、PS、PB
  - 毛利率、净利率、营收增速、盈利增速
  - ROE、自由现金流、经营现金流、负债率
  - Beta、平均成交量、行业、简称

### 新闻数据
- 从 Google News RSS 抓取（聚合 Reuters/Bloomberg/CNBC/BBC 等主流媒体）
- 6 类全局关键词 + 每只标的个性化搜索
- 当日缓存到 `cache/news_cache.pkl`，次日自动失效

### VIX 恐慌指数
- 下载 `^VIX` 收盘序列，与行情数据同时间范围对齐
- 用于全局市场情绪调节（VIX>30 恐慌扣分，VIX<15 贪婪加分，VIX急升额外扣分）
- 当日缓存到 `cache/vix_*.pkl`

### 分析师评级（yfinance）
- 通过 `yfinance.Ticker()` 获取三类数据：
  - **推荐汇总** (`recommendations`)：当月 strongBuy/buy/hold/sell/strongSell 人数，加权计算评级共识分 [-1, 1]
  - **目标价** (`analyst_price_targets`)：分析师目标价均值 vs 当前价格，计算隐含上涨空间
  - **升降级** (`upgrades_downgrades`)：近30天的 upgrade/downgrade 事件，净升级数作为动态信号
- 综合评分: `analyst_score = 共识50% + 目标价30% + 升降级20%`
- 覆盖分析师 < 3 人的标的不做调节
- 当日缓存到 `cache/analyst_YYYYMMDD.pkl`

### 机构持仓与做空数据（yfinance）
- 通过 `yfinance.Ticker().info` 和 `.insider_transactions` 获取三类数据：
  - **机构持仓占比** (`heldPercentInstitutions`)：机构投资者持仓占流通股比例
  - **做空数据**：做空占流通股比例(`shortPercentOfFloat`)、空头回补天数(`shortRatio`)、做空月度变化
  - **内部人交易** (`insider_transactions`)：近90天的 purchase/sale 事件
- 综合因子分: `inst_score = 机构持仓40% + 做空压力35% + 做空变化15% + 内部人10%`
- 高机构持仓加分，高做空比例扣分，做空增加扣分，内部人买入加分
- 当日缓存到 `cache/institutional_YYYYMMDD.pkl`

### 缓存策略
- 行情/基本面缓存路径：`cache/{label}_{md5(tickers)[:10]}_{YYYYMMDD}.pkl`
- 新闻缓存路径：`cache/news_cache.pkl`
- 分析师评级缓存路径：`cache/analyst_YYYYMMDD.pkl`
- 机构/做空数据缓存路径：`cache/institutional_YYYYMMDD.pkl`
- 同一天同一 ticker 列表只下载一次，重跑直接读缓存
- 股票池变更后 MD5 变化，自动触发重新下载
- 可手动删除 `cache/*.pkl` 强制刷新

### 数据过滤
- **自选观察池**：价格数据不足 `min_price_days=60` 天的标的被过滤（日志会输出被过滤的列表）
- **持仓股票池**：使用更宽松过滤（≥5天），确保所有实际持仓都出现在报告中
- 参考指数 ETF 仅用于杠杆引擎，不进入打分

---

## 15. 安装与运行

```bash
cd /path/to/project
python3 -m pip install -r requirements.txt

# 纯量化（不调用 LLM）
python3 run_daily.py

# 启用LLM辅助评分（Kimi K2.5 + DeepSeek V3.2 + GLM-5 三模型交叉验证）
# 配置 LKEAP_API_KEY 即可驱动全部三个模型
python3 run_daily.py --llm

# 禁用某个模型（如只用两个模型）
python3 run_daily.py --llm --no-kimi          # 禁用Kimi，仅DeepSeek+GLM
python3 run_daily.py --llm --no-deepseek      # 禁用DeepSeek，仅Kimi+GLM
python3 run_daily.py --llm --no-glm           # 禁用GLM，仅Kimi+DeepSeek
python3 run_daily.py --llm --no-kimi --no-glm # 仅DeepSeek单模型

# 指定交叉验证模式（默认 cross）
python3 run_daily.py --llm --cross-mode cross     # 方向融合：多数一致增强、分歧保守（默认）
python3 run_daily.py --llm --cross-mode avg        # 简单平均
python3 run_daily.py --llm --cross-mode primary    # 仅第一个可用模型

# 指定输出目录
python3 run_daily.py --out my_outputs

# 从 moomoo (富途) 同步持仓和自选股（需本地运行 OpenD 网关）
python3 run_daily.py --llm --sync-moomoo

# 仅同步 moomoo 持仓（不同步自选股）
python3 run_daily.py --llm --sync-moomoo-portfolio-only

# 生成格式化研报（基础版：纯数据驱动）
python3 run_daily.py --report

# 生成 AI 增强研报（LLM 为每只持仓生成深度分析）
python3 run_daily.py --llm --report
```

编辑 `.env` 配置LLM密钥（仅 `--llm` 功能需要）：

```env
# ── 腾讯云 lkeap 统一配置（三模型共用同一个 API Key）──
LLM_ENABLED=true
LKEAP_API_KEY=your_lkeap_api_key    # 在腾讯云控制台 API Key 管理页面创建
TENCENT_SECRET_ID=your_secret_id     # 可选，DeepSeek SDK 备用路径
TENCENT_SECRET_KEY=your_secret_key
TENCENT_REGION=ap-guangzhou

# ── Kimi K2.5 ──
KIMI_ENABLED=true
KIMI_MODEL=kimi-k2.5

# ── DeepSeek V3.2 ──
DEEPSEEK_ENABLED=true
DEEPSEEK_MODEL=deepseek-v3.2
DEEPSEEK_ENABLE_SEARCH=true    # 联网搜索（仅 SDK/v1 路径支持）
DEEPSEEK_TIMEOUT=90

# ── GLM-5 ──
GLM_ENABLED=true
GLM_MODEL=glm-5

# ── 交叉验证模式 ──
CROSS_VALIDATION_MODE=cross    # cross / avg / primary

# 可选：如有 lkeap 专用 DeepSeek API Key（不配置则走 /v3 统一接口或 SDK）
# DEEPSEEK_API_KEY=your_lkeap_api_key

# ── moomoo (富途) OpenAPI 配置（--sync-moomoo 时需要）──
# 需要本地安装并运行 FutuOpenD 网关，登录 moomoo 账户
# MOOMOO_HOST=127.0.0.1          # OpenD 地址
# MOOMOO_PORT=11111              # OpenD 端口
# MOOMOO_WATCHLIST_GROWTH=美股   # 自选股分组名 → 成长池
# MOOMOO_WATCHLIST_HK=港股       # 港股自选分组 → 成长池
# MOOMOO_WATCHLIST_LEVERAGE=     # 杠杆ETF分组（留空=不同步）
# MOOMOO_WATCHLIST_SMALLCAP=     # 小盘投机分组（留空=不同步）
```

> **moomoo 同步说明**：使用 `--sync-moomoo` 时，系统通过 futu-api SDK 连接本地 OpenD 网关，自动从 moomoo 账户读取实际持仓和自选股列表，覆写 `config.py` 中的默认值。需要先 `pip install futu-api` 并在本地运行 [FutuOpenD](https://openapi.futunn.com/futu-api-doc/)。

> **密钥说明**：三个模型（Kimi K2.5 / DeepSeek V3.2 / GLM-5）均通过腾讯云 lkeap `/v3` 端点调用，共用同一个 `LKEAP_API_KEY`（在[API Key 管理](https://hunyuan.cloud.tencent.com/#/app/apiKeyManage)页面创建）。DeepSeek 额外支持 `/v1` API Key 和 SDK 密钥作为备用路径。

---

## 16. 控制台输出

运行后控制台依次展示：

**📊 AI推荐组合（自选观察池）：**
1. **自选观察池评分表**：name / engine / quant_score / event_score / analyst_rating / final_score / action，按 final_score 降序
2. **AI推荐目标权重**：前15名持仓及其百分比权重
3. **信号变动**：与昨日信号对比（NEW / BUY→REDUCE 等）
4. **AI推荐组合收益跟踪**：本期收益 / 累计收益 / 基准指数对比（QQQ纳斯达克100 + SPY标普500 本期&累计涨跌幅 + 超额收益） / 各持仓明细（入场价、现价、个股收益率、加权贡献）

**💼 我的持仓（持仓股票池）：**
5. **我的持仓评分与建议**：name / engine / quant_score / final_score / action / advice_action / advice_confidence / advice_reason
6. **我的持仓整体组合分析**：组合总评 / 风险提示 / 调仓建议 / 关注要点 / 总仓位建议

**💰 DCF 估值：**
7. **DCF 估值汇总表**：ticker / current_price / dcf_bear / dcf_base / dcf_bull / upside_pct / dcf_score / wacc / fcf_M

**📝 研报（--report 时）：**
8. **研报文件路径**：`outputs/research_report_YYYYMMDD.md`

---

## 17. 输出文件

| 文件 | 说明 |
|------|------|
| `signal_table_YYYYMMDD.csv` | 自选观察池全量评分与AI推荐信号（按分数降序） |
| `target_weights_YYYYMMDD.csv` | AI推荐目标仓位权重 |
| `diff_report_YYYYMMDD.csv` | AI推荐信号与昨日对比变动 |
| `portfolio_advice_YYYYMMDD.csv` | 我的持仓评分 + LLM投资建议（加仓/持有/减仓/清仓 + 信心度 + 理由） |
| `portfolio_overall_YYYYMMDD.txt` | 我的持仓整体组合分析（组合总评/风险提示/调仓建议/关注要点/总仓位建议） |
| `dcf_valuation_YYYYMMDD.csv` | DCF 估值汇总（现价/Bear/Base/Bull/上涨空间/WACC/FCF/DCF评分） |
| `research_report_YYYYMMDD.md` | 格式化研报（--report 时生成，含投资摘要/个股分析/DCF汇总/Top推荐/组合分析/风险提示） |
| `portfolio_history.csv` | AI推荐组合持仓快照与收益跟踪历史（持久化） |
| `benchmark_history.csv` | 基准指数（QQQ/SPY）价格快照与收益跟踪历史（持久化） |

---

## 免责声明

本项目仅用于研究与学习，不构成任何投资建议。
