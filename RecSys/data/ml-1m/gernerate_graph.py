from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


DATA_DIR = Path(__file__).resolve().parent
MOVIES_FILE = DATA_DIR / "movies.dat"
RATINGS_FILE = DATA_DIR / "ratings.dat"


def setup_chinese_font():
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}

    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break

    plt.rcParams["axes.unicode_minus"] = False


def load_movies(path):
    movies = {}
    genre_counter = Counter()

    with path.open("r", encoding="latin-1") as file:
        for line in file:
            parts = line.strip().split("::")
            if len(parts) != 3:
                continue

            movie_id, title, genres = parts
            movies[movie_id] = title

            for genre in genres.split("|"):
                genre_counter[genre] += 1

    return movies, genre_counter


def load_ratings(path):
    rating_counter = Counter()
    user_rating_counter = Counter()
    movie_rating_counter = Counter()

    with path.open("r", encoding="latin-1") as file:
        for line in file:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue

            user_id, movie_id, rating, _timestamp = parts
            rating_counter[int(rating)] += 1
            user_rating_counter[user_id] += 1
            movie_rating_counter[movie_id] += 1

    return rating_counter, user_rating_counter, movie_rating_counter


def save_rating_distribution(rating_counter):
    ratings = [1, 2, 3, 4, 5]
    counts = [rating_counter[rating] for rating in ratings]

    plt.figure(figsize=(8, 5))
    plt.bar(ratings, counts, color="#4C78A8")
    plt.title("评分分布图")
    plt.xlabel("评分")
    plt.ylabel("评分数量")
    plt.xticks(ratings)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "评分分布图.png", dpi=160)
    plt.close()


def save_genre_distribution(genre_counter):
    genres = [genre for genre, _count in genre_counter.most_common()]
    counts = [count for _genre, count in genre_counter.most_common()]

    plt.figure(figsize=(11, 6))
    plt.bar(genres, counts, color="#59A14F")
    plt.title("电影类型分布图")
    plt.xlabel("电影类型")
    plt.ylabel("电影数量")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "电影类型分布图.png", dpi=160)
    plt.close()


def save_user_rating_count_distribution(user_rating_counter):
    counts = list(user_rating_counter.values())

    plt.figure(figsize=(9, 5))
    plt.hist(counts, bins=50, color="#F28E2B", edgecolor="white")
    plt.title("用户评分数量分布图")
    plt.xlabel("每个用户的评分数量")
    plt.ylabel("用户数量")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "用户评分数量分布图.png", dpi=160)
    plt.close()


def save_top10_most_rated_movies(movie_rating_counter, movies):
    top10 = movie_rating_counter.most_common(10)
    labels = [movies.get(movie_id, f"Movie {movie_id}") for movie_id, _count in top10]
    counts = [count for _movie_id, count in top10]

    labels = [label if len(label) <= 32 else label[:29] + "..." for label in labels]

    plt.figure(figsize=(11, 6))
    plt.barh(labels[::-1], counts[::-1], color="#E15759")
    plt.title("评分最多电影 Top 10")
    plt.xlabel("评分数量")
    plt.ylabel("电影名称")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "评分最多电影Top10.png", dpi=160)
    plt.close()


def main():
    setup_chinese_font()

    movies, genre_counter = load_movies(MOVIES_FILE)
    rating_counter, user_rating_counter, movie_rating_counter = load_ratings(RATINGS_FILE)

    save_rating_distribution(rating_counter)
    save_genre_distribution(genre_counter)
    save_user_rating_count_distribution(user_rating_counter)
    save_top10_most_rated_movies(movie_rating_counter, movies)

    print("中文图表已生成：")
    print(DATA_DIR / "评分分布图.png")
    print(DATA_DIR / "电影类型分布图.png")
    print(DATA_DIR / "用户评分数量分布图.png")
    print(DATA_DIR / "评分最多电影Top10.png")


if __name__ == "__main__":
    main()
