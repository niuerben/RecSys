import argparse
import ast
import csv
import importlib.util
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RECSYS_DIR = ROOT / "RecSys"
DATA_DIR = RECSYS_DIR / "data" / "ml-1m"
MODEL_DIR = RECSYS_DIR / "model"
OUTPUT_DIR = RECSYS_DIR / "outputs"


def load_module(model_name):
    model_name = model_name.lower()
    module_path = MODEL_DIR / model_name / f"{model_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {module_path}")
    spec = importlib.util.spec_from_file_location(model_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(config_path):
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    config = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not value:
                continue
            config[key] = parse_config_value(value)
    return config


def parse_config_value(value):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def config_get(config, *keys, default=None):
    for key in keys:
        if key in config:
            return config[key]
    return default


def merge(moviesfile, ratingsfile, usersfile, outputfile):
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)

    movies = {}
    with open(moviesfile, "r", encoding="latin-1") as f:
        for line in f:
            movie_id, title, genres = line.rstrip("\n").split("::")
            movies[movie_id] = (title, genres)

    users = {}
    with open(usersfile, "r", encoding="latin-1") as f:
        for line in f:
            user_id, gender, age, occupation, zipcode = line.rstrip("\n").split("::")
            users[user_id] = (gender, age, occupation, zipcode)

    with open(ratingsfile, "r", encoding="latin-1") as rf, open(
        outputfile, "w", encoding="utf-8", newline=""
    ) as wf:
        writer = csv.writer(wf)
        writer.writerow(
            [
                "user_id",
                "movie_id",
                "rating",
                "timestamp",
                "gender",
                "age",
                "occupation",
                "zipcode",
                "title",
                "genres",
            ]
        )
        for line in rf:
            user_id, movie_id, rating, timestamp = line.rstrip("\n").split("::")
            gender, age, occupation, zipcode = users[user_id]
            title, genres = movies[movie_id]
            writer.writerow(
                [user_id, movie_id, rating, timestamp, gender, age, occupation, zipcode, title, genres]
            )
    return outputfile


def run_ncf(module, config, paths):
    mlp_hidden = config_get(config, "mlp_hidden", "mlp_hidden_size", default=(32, 16, 8))
    if isinstance(mlp_hidden, str):
        mlp_hidden = module.parse_hidden_layers(mlp_hidden)

    model = module.NCF(
        recommendation_topn=config_get(config, "recommendation_topn", default=100),
        embedding_size=config_get(config, "embedding_size", default=64),
        mlp_hidden=tuple(mlp_hidden),
        dropout=config_get(config, "dropout", "dropout_prob", default=0.0),
        epochs=config_get(config, "epochs", default=500),
        batch_size=config_get(config, "batch_size", "train_batch_size", default=4096),
        learning_rate=config_get(config, "learning_rate", default=1e-4),
        num_neg=config_get(config, "num_neg", default=4),
        seed=config_get(config, "seed", default=2020),
        valid_interval=config_get(config, "valid_interval", default=1),
        early_stop_patience=config_get(config, "early_stop_patience", default=10),
        min_delta=config_get(config, "min_delta", default=1e-6),
        save_epoch_recommendations=False,
    )
    model.generate_dataset(str(paths["ratings"]), usersfile=str(paths["users"]))
    model.calc_movie_sim()
    model.evaluate()
    model.generate_recommendation(
        filepath=str(paths["output"]),
        topn=config_get(config, "recommendation_topn", default=100),
    )


def run_xsimgcl(module, config, paths):
    model = module.XSimGCL(
        topn=config_get(config, "topn", default=10),
        recommendation_topn=config_get(config, "recommendation_topn", default=100),
        embedding_dim=config_get(config, "embedding_dim", "embedding_size", default=64),
        n_layers=config_get(config, "n_layers", default=2),
        epochs=config_get(config, "epochs", default=500),
        batch_size=config_get(config, "batch_size", "train_batch_size", default=4096),
        learning_rate=config_get(config, "learning_rate", default=0.002),
        reg_weight=config_get(config, "reg_weight", default=1e-4),
        ssl_weight=config_get(config, "ssl_weight", "lambda", default=0.1),
        temperature=config_get(config, "temperature", default=0.2),
        eps=config_get(config, "eps", default=0.2),
        layer_cl=config_get(config, "layer_cl", default=1),
        seed=config_get(config, "seed", default=2020),
        valid_interval=config_get(config, "valid_interval", default=1),
        early_stop_patience=config_get(config, "early_stop_patience", default=10),
        min_delta=config_get(config, "min_delta", default=1e-6),
        save_epoch_recommendations=False,
    )
    model.generate_dataset(str(paths["ratings"]), usersfile=str(paths["users"]))
    model.calc_movie_sim()
    model.evaluate()
    model.generate_recommendation(filepath=str(paths["output"]), mask_valid=True)


def run_dcn(module, config, paths):
    merged_file = paths["merged"]
    if not merged_file.exists():
        merge(str(paths["movies"]), str(paths["ratings"]), str(paths["users"]), str(merged_file))

    model = module.DCN(
        topn=config_get(config, "topn", default=10),
        rating_threshold=config_get(config, "rating_threshold", default=3),
        epochs=config_get(config, "epochs", default=5),
        batch_size=config_get(config, "batch_size", "train_batch_size", default=8192),
        learning_rate=config_get(config, "learning_rate", default=1e-3),
        weight_decay=config_get(config, "weight_decay", default=1e-6),
        train_neg_per_positive=config_get(config, "train_neg_per_positive", default=2),
        seed=config_get(config, "seed", default=0),
        recommendation_topn=config_get(config, "recommendation_topn", default=100),
        valid_interval=config_get(config, "valid_interval", default=1),
        early_stop_patience=config_get(config, "early_stop_patience", default=10),
        min_delta=config_get(config, "min_delta", default=1e-6),
        save_epoch_recommendations=False,
    )
    model.generate_dataset(str(merged_file))
    model.calc_movie_sim()
    model.evaluate()
    model.generate_recommendation(
        filepath=str(paths["output"]),
        topn=config_get(config, "recommendation_topn", default=100),
    )


def run_user_cf(module, config, paths):
    model = module.UserCF(
        rating_threshold=config_get(config, "rating_threshold", default=3),
        recommendation_topn=config_get(config, "recommendation_topn", default=100),
    )
    model.generate_dataset(str(paths["ratings"]), usersfile=str(paths["users"]))
    model.calc_user_sim()
    model.evaluate()
    model.generate_recommendation(
        filepath=str(paths["output"]),
        topn=config_get(config, "recommendation_topn", default=100),
    )


def run_item_cf(module, config, paths):
    model = module.ItemCF(
        rating_threshold=config_get(config, "rating_threshold", default=3),
        recommendation_topn=config_get(config, "recommendation_topn", default=100),
    )
    model.generate_dataset(str(paths["ratings"]), usersfile=str(paths["users"]))
    model.calc_movie_sim()
    model.evaluate()
    model.generate_recommendation(
        filepath=str(paths["output"]),
        topn=config_get(config, "recommendation_topn", default=100),
    )


RUNNERS = {
    "ncf": run_ncf,
    "xsimgcl": run_xsimgcl,
    "dcn": run_dcn,
    "user_cf": run_user_cf,
    "item_cf": run_item_cf,
}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="RecSys unified runner.")
    parser.add_argument("--model", required=True, help="模型名，例如 ncf、xsimgcl、dcn、user_cf、item_cf")
    parser.add_argument("--config", default=None, help="配置文件路径，例如 RecSys/config/ncf.yaml")
    return parser


def main():
    args = build_arg_parser().parse_args()
    model_name = args.model.lower()
    if model_name not in RUNNERS:
        supported = ", ".join(sorted(RUNNERS))
        raise ValueError(f"暂不支持模型 {model_name}，当前支持: {supported}")

    config = load_config(args.config)
    module = load_module(model_name)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "ratings": DATA_DIR / "ratings.dat",
        "users": DATA_DIR / "users.dat",
        "movies": DATA_DIR / "movies.dat",
        "merged": DATA_DIR / "merged.dat",
        "output": OUTPUT_DIR / f"{model_name}_recommendation.csv",
    }
    start = time.time()
    RUNNERS[model_name](module, config, paths)
    print("%s finished in %.2fs, output: %s" % (model_name, time.time() - start, paths["output"]))


if __name__ == "__main__":
    main()
