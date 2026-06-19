# -*- coding: utf-8 -*-
"""
基于用户的协同过滤
"""

import sys
import random
import math
import os
import csv
from pathlib import Path
from operator import itemgetter


random.seed(0)


class UserCF(object):
    ''' TopN recommendation - User Based Collaborative Filtering '''

    def __init__(self, rating_threshold=3, recommendation_topn=100):
        self.trainset = {}
        self.testset = {}
        self.train_ratedset = {}
        self.user_ids = []

        self.n_sim_user = 20
        self.n_rec_movie = recommendation_topn
        self.rating_threshold = rating_threshold
        
        
        self.user_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

    # 先过滤rating<3，再按8:1:1切分，验证集当前不参与训练和评估
    def generate_dataset(self, filename, usersfile=None, pivot=0.8, valid_ratio=0.1):
        ''' load rating data and split it to training set and test set '''
        print("使用设备:cpu")
        print("加载 UserCF 数据...")
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

    def calc_user_sim(self):
        ''' calculate user similarity matrix '''
        print("加载模型 UserCF...")
        # 建立电影到用户的倒排表
        # key=movieID, value=list of userIDs who have seen this movie
        print ('building movie-users inverse table...', file=sys.stderr)
        movie2users = dict()

        for user, movies in self.trainset.items():
            for movie in movies:
                # inverse table for item-users
                if movie not in movie2users:
                    movie2users[movie] = set()
                movie2users[movie].add(user)
                # count item popularity at the same time
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1
        print ('build movie-users inverse table succ', file=sys.stderr)

        # save the total movie number, which will be used in evaluation
        self.movie_count = len(movie2users)
        print ('total movie number = %d' % self.movie_count, file=sys.stderr)

        # 计算用户相似度矩阵
        usersim_mat = self.user_sim_mat
        print ('building user co-rated movies matrix...', file=sys.stderr)

        for movie, users in movie2users.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    usersim_mat.setdefault(u, {})
                    usersim_mat[u].setdefault(v, 0)
                    usersim_mat[u][v] += 1
        print ('build user co-rated movies matrix succ', file=sys.stderr)

        print ('calculating user similarity matrix...', file=sys.stderr)
        simfactor_count = 0
        PRINT_STEP = 2000000

        for u, related_users in usersim_mat.items():
            for v, count in related_users.items():
                usersim_mat[u][v] = count / math.sqrt(
                    len(self.trainset[u]) * len(self.trainset[v]))
                simfactor_count += 1
                if simfactor_count % PRINT_STEP == 0:
                    print ('calculating user similarity factor(%d)' %
                           simfactor_count, file=sys.stderr)

        print ('calculate user similarity matrix(similarity factor) succ',
               file=sys.stderr)
        print ('Total similarity factor number = %d' %
               simfactor_count, file=sys.stderr)

    def recommend(self, user):
        ''' 根据K个相似用户推荐N个该用户没看过的电影. '''
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = dict()
        watched_movies = self.train_ratedset.get(user, set())

        for similar_user, similarity_factor in sorted(self.user_sim_mat.get(user, {}).items(),
                                                      key=itemgetter(1), reverse=True)[0:K]:
            for movie, rating in self.trainset[similar_user].items():
                if movie in watched_movies:
                    continue
                # predict the user's "interest" for each movie
                rank.setdefault(movie, 0)
                rank[movie] += similarity_factor*rating
        # return the N best movies
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        ''' print evaluation result: precision@K, ndcg@K and map@K '''
        print ('Evaluation start...', file=sys.stderr)

        N = self.n_rec_movie
        precision_sum = 0.0
        recall_sum = 0.0
        ndcg_sum = 0
        mrr_sum = 0.0
        hit_user_count = 0
        eval_user_count = 0

        for i, user in enumerate(self.user_ids or self.trainset):
            if i % 500 == 0:
                print ('recommended for %d users' % i, file=sys.stderr)
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

    def generate_recommendation(self, filepath='./RecSys/outputs/user_cf_recommendation.csv', topn=None):
        ''' 输出推荐结果 '''
        topn = topn or self.n_rec_movie
        old_topn = self.n_rec_movie
        self.n_rec_movie = topn
        users = self.user_ids or sorted(self.trainset, key=lambda value: int(value))

        print ('generating UserCF recommendation result: %s' % filepath)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id'] + ['rec%d' % i for i in range(1, topn + 1)])
            for i, user in enumerate(users):
                if i % 500 == 0:
                    print ('generate recommendation for %d users' % i, file=sys.stderr)
                rec_movies = self.recommend(user)
                movie_ids = [movie for movie, _ in rec_movies[:topn]]
                movie_ids.extend([''] * (topn - len(movie_ids)))
                writer.writerow([user] + movie_ids)
        self.n_rec_movie = old_topn
        print ('generate recommendation result succ', file=sys.stderr)

if __name__ == '__main__':
    recsys_root = Path(__file__).resolve().parents[2]
    ratingfile = recsys_root / "data" / "ml-1m" / "ratings.dat"
    usersfile = recsys_root / "data" / "ml-1m" / "users.dat"
    outputfile = recsys_root / "outputs" / "user_cf_recommendation.csv"
    usercf = UserCF()
    usercf.generate_dataset(str(ratingfile), usersfile=str(usersfile))
    usercf.calc_user_sim()
    usercf.generate_recommendation(filepath=str(outputfile))
