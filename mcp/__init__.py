"""
HealthMind 自研 MCP 风格工具框架（本地包）。

注意：本项目同时安装了官方 mcp SDK（供 mcp_server.py 导出标准协议）。
本目录必须是正规包（有 __init__.py），才能凭借 sys.path 优先级
压过 site-packages 中的 SDK 同名包，保证项目内 `from mcp.xxx import` 指向这里。
"""
