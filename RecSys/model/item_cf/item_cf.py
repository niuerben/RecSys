# -*- coding: utf-8 -*-
"""
基于物品的协同过滤
"""


import random
import sys
import math
import os
import csv
from pathlib import Path
from operator import itemgetter
random.seed(0)


class ItemCF(object):
    def __init__(self, rating_threshold=3, recommendation_topn=100):
        self.trainset = {}
        self.testset = {}
        self.train_ratedset = {}
        self.user_ids = []

        self.n_sim_movie = 20
        self.n_rec_movie = recommendation_topn
        self.rating_threshold = rating_threshold

        self.movie_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

    # 先过滤rating<3，再按8:1:1切分，验证集当前不参与训练和评估
    def generate_dataset(self, filename, usersfile=None, pivot=0.8, valid_ratio=0.1):
        print("使用设备:cpu")
        print("加载 ItemCF 数据...")
        trainset_len = 0
        testset_len = 0
        train_rated_len = 0
        observed_users = set()
        observed_movies = set()
        positive_interactions = 0
        with open(filename, 'r') as fp:
            for line in fp:
                user, movie, rating, _ = line.split('::')
                observed_users.add(user)
                rating = int(rating)
                if rating < self.rating_threshold:
                    continue
                observed_movies.add(movie)
                positive_interactions += 1

                rand_value = random.random()
                if rand_value < pivot:
                    self.train_ratedset.setdefault(user, set())
                    self.train_ratedset[user].add(movie)
                    train_rated_len += 1
                    self.trainset.setdefault(user, {})
                    self.trainset[user][movie] = rating
                    trainset_len += 1
                elif rand_value >= pivot + valid_ratio:
                    self.testset.setdefault(user, {})
                    self.testset[user][movie] = rating
                    testset_len += 1

        if usersfile and os.path.exists(usersfile):
            with open(usersfile, 'r', encoding='latin-1') as f:
                self.user_ids = [line.rstrip('\n').split('::')[0] for line in f if line.strip()]
        else:
            self.user_ids = sorted(observed_users, key=lambda value: int(value))

        print(
            "用户数:%d，电影数:%d，交互数:%d，训练:%d，Top%d 推荐列: %d"
            % (
                len(self.user_ids),
                len(observed_movies),
                positive_interactions,
                trainset_len,
                self.n_rec_movie,
                len(self.user_ids) * self.n_rec_movie,
            )
        )

    def calc_movie_sim(self):
        print("加载模型 ItemCF...")
        for user, movies in self.trainset.items():
            for movie in movies:
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1
        print('count movies number and pipularity succ', file=sys.stderr)

        self.movie_count = len(self.movie_popular)
        print('total movie number = %d' % self.movie_count, file=sys.stderr)

        itemsim_mat = self.movie_sim_mat
        print('building co-rated users matrix', file=sys.stderr)
        for user, movies in self.trainset.items():
            for m1 in movies:
                for m2 in movies:
                    if m1 == m2:
                        continue
                    itemsim_mat.setdefault(m1, {})
                    itemsim_mat[m1].setdefault(m2, 0)
                    itemsim_mat[m1][m2] += 1

        print('build co-rated users matrix succ', file=sys.stderr)
        print('calculating movie similarity matrix', file=sys.stderr)

        simfactor_count = 0
        PRINT_STEP = 2000000

        for m1, related_movies in itemsim_mat.items():
            for m2, count in related_movies.items():
                itemsim_mat[m1][m2] = count / math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print('calcu movie similarity factor(%d)' % simfactor_count, file=sys.stderr)
        print('calcu similiarity succ', file=sys.stderr)

    def recommend(self, user):
        K = self.n_sim_movie
        N = self.n_rec_movie
        rank = {}
        # watched_movies表示用户user看过的电影和评分
        watched_movies = self.train_ratedset.get(user, set())

        for movie, rating in self.trainset.get(user, {}).items():
            for related_movie, similarity_factor in sorted(self.movie_sim_mat.get(movie, {}).items(), key=itemgetter(1),
                                                           reverse=True)[0:K]:
                if related_movie in watched_movies:
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += similarity_factor * rating
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        print('evaluation start', file=sys.stderr)

        N = self.n_rec_movie

        precision_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0
        mrr_sum = 0.0
        hit_user_count = 0
        eval_user_count = 0

        for i, user in enumerate(self.user_ids or self.trainset):
            if i % 500 == 0:
                print('recommend for %d users ' % i, file=sys.stderr)
            test_movies = self.testset.get(user, {})
            if not test_movies:
                continue
            rec_movies = self.recommend(user)

            dcg = 0
            user_hit = 0
            reciprocal_rank = 0.0
            for rank, (movie, _) in enumerate(rec_movies, start=1):
                if movie in test_movies:
                    user_hit += 1
                    dcg += 1 / math.log2(rank + 1)
                    if reciprocal_rank == 0.0:
                        reciprocal_rank = 1 / rank

            ideal_hits = min(len(test_movies), N)
            idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
            precision_sum += user_hit / N
            recall_sum += user_hit / len(test_movies)
            ndcg_sum += dcg / idcg if idcg else 0
            mrr_sum += reciprocal_rank
            if user_hit > 0:
                hit_user_count += 1
            eval_user_count += 1

        metrics = {
            "recall": recall_sum / eval_user_count if eval_user_count else 0,
            "mrr": mrr_sum / eval_user_count if eval_user_count else 0,
            "ndcg": ndcg_sum / eval_user_count if eval_user_count else 0,
            "hit": hit_user_count / eval_user_count if eval_user_count else 0,
            "precision": precision_sum / eval_user_count if eval_user_count else 0,
        }
        print(
            "测试集 Test RECALL@%d : %.4f    MRR@%d : %.4f    NDCG@%d : %.4f    HIT@%d : %.4f    PRECISION@%d : %.4f"
            % (N, metrics["recall"], N, metrics["mrr"], N, metrics["ndcg"], N, metrics["hit"], N, metrics["precision"])
        )
        return metrics

    def generate_recommendation(self, filepath='./RecSys/outputs/item_cf_recommendation.csv', topn=None):
        ''' 输出推荐结果 '''
        topn = topn or self.n_rec_movie
        old_topn = self.n_rec_movie
        self.n_rec_movie = topn
        users = self.user_ids or sorted(self.trainset, key=lambda value: int(value))

        print('generating ItemCF recommendation result: %s' % filepath)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id'] + ['rec%d' % i for i in range(1, topn + 1)])
            for i, user in enumerate(users):
                if i % 500 == 0:
                    print('generate recommendation for %d users' % i, file=sys.stderr)
                rec_movies = self.recommend(user)
                movie_ids = [movie for movie, _ in rec_movies[:topn]]
                movie_ids.extend([''] * (topn - len(movie_ids)))
                writer.writerow([user] + movie_ids)
        self.n_rec_movie = old_topn
        print('generate recommendation result succ', file=sys.stderr)

if __name__ == '__main__':
    recsys_root = Path(__file__).resolve().parents[2]
    ratingfile = recsys_root / "data" / "ml-1m" / "ratings.dat"
    usersfile = recsys_root / "data" / "ml-1m" / "users.dat"
    outputfile = recsys_root / "outputs" / "item_cf_recommendation.csv"
    item_cf = ItemCF()
    item_cf.generate_dataset(str(ratingfile), usersfile=str(usersfile))
    item_cf.calc_movie_sim()
    item_cf.generate_recommendation(filepath=str(outputfile))
