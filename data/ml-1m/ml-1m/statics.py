from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
USERS_FILE = DATA_DIR / "users.dat"
MOVIES_FILE = DATA_DIR / "movies.dat"
RATINGS_FILE = DATA_DIR / "ratings.dat"


def count_lines(path):
    count = 0
    with path.open("r", encoding="latin-1") as file:
        for _ in file:
            count += 1
    return count


def analyze_ratings(path):
    rating_count = 0
    rating_sum = 0.0
    rated_users = set()
    rated_movies = set()

    with path.open("r", encoding="latin-1") as file:
        for line in file:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue

            user_id, movie_id, rating, _timestamp = parts
            rated_users.add(user_id)
            rated_movies.add(movie_id)
            rating_sum += float(rating)
            rating_count += 1

    average_rating = rating_sum / rating_count if rating_count else 0.0
    return rating_count, average_rating, rated_users, rated_movies


def main():
    user_count = count_lines(USERS_FILE)
    movie_count = count_lines(MOVIES_FILE)
    rating_count, average_rating, rated_users, rated_movies = analyze_ratings(RATINGS_FILE)

    total_possible_ratings = user_count * movie_count
    density = rating_count / total_possible_ratings if total_possible_ratings else 0.0
    sparsity = 1 - density

    print("MovieLens 1M 数据集统计")
    print("=" * 28)
    print(f"用户数量: {user_count}")
    print(f"电影数量: {movie_count}")
    print(f"评分数量: {rating_count}")
    print(f"平均评分: {average_rating:.4f}")
    print(f"评分矩阵密度: {density:.6f}")
    print(f"评分矩阵稀疏度: {sparsity:.6f}")
    print()
    print("补充检查")
    print("=" * 28)
    print(f"ratings.dat 中出现过的用户数量: {len(rated_users)}")
    print(f"ratings.dat 中出现过的电影数量: {len(rated_movies)}")


if __name__ == "__main__":
    main()
