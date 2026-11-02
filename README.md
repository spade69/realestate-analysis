# 福田二手成交分析

深圳福田区二手房成交数据（行舟深房笔记公众号）的多期对比分析。

🌐 **在线报告：** https://spade69.github.io/realestate-analysis/

## 涵盖期次

- 2026-03 上 / 03 下
- 2026-04 上 / 04 下
- 2026-05 上 / 05 下
- 2026-06 上 / 06 下
- 2026-07 上 / 07 下
- 2026-08 上 / 08 下
- 2026-09 上 / 09 下
- 2026-10 上

共 1962 笔成交（1718 笔含谈价率），另有 27 笔法拍单独列出、不计入分析。

## 分析维度

1. **谈价率极值** — 各期 ≤-20%/-30%/-50% 跌幅笔数 + 楼盘明细
2. **谈价率分布** — 六桶分布演变
3. **片区涨跌幅** — 21 个片区跨期对比，自动分类「真信号 / 采样偏差 / 样本不足」
4. **户型维度** — 1/2/3/4+ 室分桶
5. **总价段维度 + 假设检验** — 验证「低总价 → 砍价更狠」（Pearson + Spearman）
6. **重点楼盘** — 多期 ≥2 笔成交的小区追踪
7. **同小区同户型跨期对比** — 消除品种偏移的真实信号
8. **学区溢价** — 名校 / 普通 / 无学位 三档；同片区单价中位数对比 + 回归（控制片区、期次、房龄、面积、户型）

## 项目结构

```
.
├── data/                # 转录的 CSV，每期一个文件
├── ref/school_map.csv   # 小区 → 对口小学/初中 + 学区档（名校/普通/无学位），手工维护
├── analyze.py           # 单文件分析脚本
├── plot_trends.py       # 趋势图生成
├── reports/             # 生成的 Markdown 报告
├── docs/                # 部署到 GitHub Pages 的静态站点
└── requirements.txt     # pandas / scipy / matplotlib
```

## 使用

```bash
pip install -r requirements.txt

# 加新一期数据：
# 1) 把新 CSV 放到 data/ (符合 schema；可选尾列 is_auction，1 = 法拍)
# 2) 全自动重新生成 docs/index.html + chart_data.json：
python3 render_site.py

# 可选：生成 Markdown 报告和 PNG 趋势图
python3 analyze.py        # → reports/*.md
python3 plot_trends.py    # → reports/*.png
```

## 模板与渲染

`template/index.html` 是带 `{{ PLACEHOLDER }}` 的模板，`render_site.py` 读 CSV → 计算指标 → 自动生成 KPI/表格/解读 → 替换占位符 → 写出 `docs/index.html`。所有"解读"性质文字（TLDR、片区观察、信号摘要、假设检验结论）都按规则自动生成，不需要手改。

## 学区映射

`ref/school_map.csv` 每行一个小区（按成交笔数降序），`tier` 留空表示未标注，不参与学区分析。
填写 `primary_school` / `middle_school` / `tier`，并在 `source` 注明依据（如“福田区2026小一地段代码图”）。
福田每年调整划片，映射以最新招生地段为准。

## 数据来源

行舟深房笔记（微信公众号）双周发布的福田区二手成交记录，手动转录为 CSV。
