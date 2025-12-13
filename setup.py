from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.is_file():
        return []
    lines = req_file.read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


setup(
    name="aalap",
    version="0.1.0",
    author="Moniruzzaman Akash",
    author_email="akash.moniruzzaman@gmail.com",
    description="Voice assistant dialogue manager with faster-whisper ASR, Piper TTS, and advanced wake-word support.",
    packages=find_packages(include=["aalap", "aalap.*"]),
    include_package_data=True,
    package_data={"aalap": ["resources/models/*"]},
    python_requires=">=3.9",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "aalap=aalap.dialogue_manager:cli",
        ],
    },
)
