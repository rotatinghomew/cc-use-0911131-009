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

容器运行可使用已经预装的 ccusr/prepared:3f007a95fcee-4441968e5ff6 镜像；将当前仓库只读挂载至 /source，在容器内复制到 /tmp/product 后运行上述命令。依赖准备和正式运行分离，运行过程中不需要网络访问。
