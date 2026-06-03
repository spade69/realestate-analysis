# 福田二手成交分析

深圳福田区二手房成交数据（行舟深房笔记公众号）的多期对比分析。

🌐 **在线报告：** https://spade69.github.io/realestate-analysis/

## 涵盖期次

- 2026-03 上 / 03 下
- 2026-04 上 / 04 下
- 2026-05 上

共 749 笔成交（608 笔含谈价率）。

## 分析维度

1. **谈价率极值** — 各期 ≤-20%/-30%/-50% 跌幅笔数 + 楼盘明细
2. **谈价率分布** — 六桶分布演变
3. **片区涨跌幅** — 21 个片区跨期对比，自动分类「真信号 / 采样偏差 / 样本不足」
4. **户型维度** — 1/2/3/4+ 室分桶
5. **总价段维度 + 假设检验** — 验证「低总价 → 砍价更狠」（Pearson + Spearman）
6. **重点楼盘** — 多期 ≥2 笔成交的小区追踪
7. **同小区同户型跨期对比** — 消除品种偏移的真实信号

## 项目结构

```
.
├── data/                # 转录的 CSV，每期一个文件
├── analyze.py           # 单文件分析脚本
├── plot_trends.py       # 趋势图生成
├── reports/             # 生成的 Markdown 报告
├── docs/                # 部署到 GitHub Pages 的静态站点
└── requirements.txt     # pandas / scipy / matplotlib
```

## 使用

```bash
pip install -r requirements.txt
python3 analyze.py        # 生成 Markdown 报告 → reports/
python3 plot_trends.py    # 生成 4 张趋势图 → reports/
```

## 数据来源

行舟深房笔记（微信公众号）双周发布的福田区二手成交记录，手动转录为 CSV。
