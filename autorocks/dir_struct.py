from pathlib import Path

RootDir = Path(__file__).parent
PackageRootDir = RootDir.parent
LocalResultDir = PackageRootDir / "local_execution"
ProcessedDataDir = PackageRootDir / "ProcessedData"
