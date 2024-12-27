from setuptools import setup, find_packages

setup(
    name="dataloader_param_helper",  # 패키지 이름
    version="0.1.0",  # 초기 버전
    author="Soo Hwan Cho",
    author_email="soohwancho@korea.ac.kr",
    description="Finding the optimal parameters for a Dataloader.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/splendidz/dataloader_param_helper",  # GitHub URL
    packages=find_packages(),  # 패키지 자동 탐색
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
