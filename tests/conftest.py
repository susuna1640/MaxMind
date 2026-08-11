import pathlib
import sys

# 保证从任意目录运行 pytest 时都能导入项目根包
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
