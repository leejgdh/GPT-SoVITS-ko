"""GPT_SoVITS 내부 임포트를 위한 sys.path 설정. 단일 소스.

서버(context.py)와 스크립트(_bootstrap.py) 양쪽에서 이 모듈을 임포트하여
동일한 경로 설정을 공유한다.
"""
import os
import sys


def setup_gpt_sovits_paths() -> str:
    """프로젝트 루트를 기준으로 GPT_SoVITS 관련 경로를 sys.path에 추가한다.

    추가되는 경로:
      - 프로젝트 루트 (src/, tools/ 등 import용)
      - GPT_SoVITS/ (AR/, module/, text/ 등 내부 모듈 import용)
      - GPT_SoVITS/eres2net/ (ERes2NetV2 직접 import용)

    Returns:
        프로젝트 루트 절대 경로.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    for p in [
        root,
        os.path.join(root, "GPT_SoVITS"),
        os.path.join(root, "GPT_SoVITS", "eres2net"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)
    return root
