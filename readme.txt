辅助文本复核观察服务。现有盲化抽检、人员分组、采纳修改指标和 SQLite 审计记录支持本地质量分析。

环境：Linux，Python 3.11.2 或更新的兼容版本。所有依赖来自 Python 标准库，无需安装第三方包。

医生分身对齐采样审计台
医疗AI质控团队对医生与分身的成对输出做盲化采样，计算采纳率、覆盖率、修改率和编辑距离并监测时间窗漂移。

测试：python3 -m unittest discover -s tests/avatar_alignment -p 'test_*.py'
构建：python3 -m compileall -q alignment_observatory tests/avatar_alignment
验收：python3 -m alignment_observatory.lab.audit_cli --fixture fixtures/demo.json --database /tmp/avatar_alignment-acceptance.sqlite3

目录：alignment_observatory 保存业务处理与存储模块；tests/avatar_alignment 保存业务回归；fixtures/demo.json 为原有业务样例，fixtures/analysis 为经营分析资料及字段说明。

原有验收命令每次使用新的 SQLite 路径，避免复用上次状态。新增输入完整性检查：
python3 -m unittest discover -s tests/analysis_inputs -v

每周监测（经营分析）
只比较工作人员对辅助文本的复核行为，不评价诊断正确性。命令：
python3 -m alignment_observatory.monitoring --input fixtures/analysis/input.json --database /tmp/monitoring.sqlite3 --output /tmp/monitoring.json

输出 JSON 含 series_id、estimates、alerts、design_checks、reference_version。
每条 estimate 带 week、version、metric、estimate、interval、status；每条 alert 带 alert_id、week、reason。

方法要点（详见 alignment_observatory/monitoring/__init__.py 与 engine.py 顶部说明）：
- 估计：按 inclusion_probability 做设计加权（Hajek 比率），方差按抽样层分层、按 operator 聚类；
  同一 sample_id 只取 as_of 当日已知的最高 revision，修订替代旧结论，不重复累计。
- 未完成复核不计入采纳/修改的分母，单独进入 coverage_rate；未完成不等于“未采纳”。
- 没有总体框（sampling_frame）或入样概率非法的层，estimate 为 null、status=unestimable，
  不填零，也不改用未加权结果。
- 参考期第一次运行即冻结；参考期或其数据变化会开新的 series，旧 series 与已发告警保留。
- 告警：需同时满足实际效应阈值、按查看次数/同周版本数/指标数做 Bonferroni 校正后的统计阈值、
  连续两周同向，才会给出 supported_shift。抽样结构变化标为 design_shift；
  数据不足标为 insufficient_evidence；人员/站点构成变化只作为解释信息，不称为模型退化。
- 同一输入重放、或同一周修订后重跑，不会新增告警；修订会 supersede 旧告警并保留记录。

样例：fixtures/analysis/sampling_shift_only.json 只改变抽样比例（40/40 -> 10/70），
潜在复核行为不变，用于验证加权估计不动、且不会报成质量下降。

原有盲化抽检入口保持不变：python3 -m alignment_observatory.lab.audit_cli ...

容器运行可使用已经预装的 ccusr/prepared:3f007a95fcee-4441968e5ff6 镜像；将当前仓库只读挂载至 /source，在容器内复制到 /tmp/product 后运行上述命令。依赖准备和正式运行分离，运行过程中不需要网络访问。
