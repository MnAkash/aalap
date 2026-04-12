from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT / "README.md"


def read_requirements() -> list[str]:
    req_file = ROOT / "requirements.txt"
    if not req_file.is_file():
        return []
    lines = req_file.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


setup(
    name="aalap",
    version="0.1.0",
    author="Moniruzzaman Akash",
    author_email="akashmoniruzzaman@gmail.com",
    description="Voice assistant dialogue manager with faster-whisper ASR, Piper TTS, and wake-word support.",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/MnAkash/aalap",
    project_urls={
        "Homepage": "https://github.com/MnAkash/aalap",
        "Repository": "https://github.com/MnAkash/aalap",
        "Issues": "https://github.com/MnAkash/aalap/issues",
    },
    license="Apache-2.0",
    packages=find_packages(include=["aalap", "aalap.*"]),
    python_requires=">=3.9",
    install_requires=read_requirements(),
    keywords=[
        "voice assistant",
        "speech recognition",
        "whisper",
        "wake word",
        "tts",
        "vad",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "aalap=aalap.dialogue_manager:cli",
        ],
    },
)
