import yaml
from pathlib import Path
import ipsuite as ips
from apax.nodes import Apax, ApaxBatchPrediction
from copy import deepcopy

# def create_apax_config_files(file: Path = Path("config/apax.yaml")):
#     reference = yaml.safe_load(file.read_text())
#     cutoffs = [2, 3, 4, 5, 6]
#     nns = [(16, 16), (32, 32), (64, 64), (256, 256), (512,512)]
#     n_basis = [4, 8, 16]
#     n_radial = [3, 4, 6, 7]

reference = yaml.safe_load(Path("config/apax.yaml").read_text())

project = ips.Project()
with project:
    train = ips.AddDataH5MD(file="data/cosmo_water_train.h5")
    val = ips.AddDataH5MD(file="data/cosmo_water_val.h5")
    test = ips.AddDataH5MD(file="data/cosmo_water_test.h5")

for r_max in [2, 3, 4, 5, 6]:
    with project.group("r_max", f"{r_max}"):
        config = deepcopy(reference)
        config["model"]["basis"]["r_max"] = r_max
        file = Path(f"config/apax-r_max-{r_max}.yaml")
        file.write_text(yaml.dump(config))

        model = Apax(
            config=file.as_posix(),
            data=train.frames,
            validation_data=val.frames,
        )
        test_eval = ApaxBatchPrediction(data=test.frames, model=model)
        ips.PredictionMetrics(
            x=test_eval.frames, y=test.frames
        )

for nn in [(16, 16), (32, 32), (64, 64), (128, 128)]:
    with project.group("nn", f"{nn[0]}-{nn[1]}"):
        config = deepcopy(reference)
        config["model"]["nn"] = list(nn)
        file = Path(f"config/apax-nn-{nn[0]}-{nn[1]}.yaml")
        file.write_text(yaml.dump(config))

        model = Apax(
            config=file.as_posix(),
            data=train.frames,
            validation_data=val.frames,
        )
        test_eval = ApaxBatchPrediction(data=test.frames, model=model)
        ips.PredictionMetrics(
            x=test_eval.frames, y=test.frames
        )

for n_basis in [4, 8, 16]:
    with project.group("n_basis", f"{n_basis}"):
        config = deepcopy(reference)
        config["model"]["basis"]["n_basis"] = n_basis
        file = Path(f"config/apax-n_basis-{n_basis}.yaml")
        file.write_text(yaml.dump(config))

        model = Apax(
            config=file.as_posix(),
            data=train.frames,
            validation_data=val.frames,
        )
        test_eval = ApaxBatchPrediction(data=test.frames, model=model)
        ips.PredictionMetrics(
            x=test_eval.frames, y=test.frames
        )

for n_radial in [5, 6, 7]:
    with project.group("n_radial", f"{n_radial}"):
        config = deepcopy(reference)
        config["model"]["basis"]["n_radial"] = n_radial
        file = Path(f"config/apax-n_radial-{n_radial}.yaml")
        file.write_text(yaml.dump(config))

        model = Apax(
            config=file.as_posix(),
            data=train.frames,
            validation_data=val.frames,
        )
        test_eval = ApaxBatchPrediction(data=test.frames, model=model)
        ips.PredictionMetrics(
            x=test_eval.frames, y=test.frames
        )

if __name__ == "__main__":
    project.build()
    # running on CIP5
