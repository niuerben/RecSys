# -*- coding: utf-8 -*-
"""
基于物品的协同过滤
"""


import random
import sys
import math
from operator import itemgetter
random.seed(0)


class ItemCF(object):
    def __init__(self):
        self.trainset = {}
        self.testset = {}

        self.n_sim_movie = 20
        self.n_rec_movie = 10

        self.movie_sim_mat = {}
        self.movie_popular = {}
        self.movie_count = 0

        print('Similar movie number = %d' % self.n_sim_movie, file=sys.stderr)
        print('Recommendend movie number = %d' % self.n_rec_movie, file=sys.stderr)
    # 训练比为0.7
    def generate_dataset(self, filename, pivot=0.7):
        trainset_len = 0
        testset_len = 0
        fp = open(filename, 'r')
        for line in fp:
            user, movie, rating, _ = line.split('::')
            if random.random() < pivot:
                self.trainset.setdefault(user, {})
                self.trainset[user][movie] = int(rating)
                trainset_len += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                testset_len += 1

        print('split succ , trainset is %d , testset is %d' % (trainset_len, testset_len), file=sys.stderr)

    def calc_movie_sim(self):
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
        watched_movies = self.trainset[user]

        for movie, rating in watched_movies.items():
            for related_movie, similarity_factor in sorted(self.movie_sim_mat[movie].items(), key=itemgetter(1),
                                                           reverse=True)[0:K]:
                if related_movie in watched_movies:
                    continue
                rank.setdefault(related_movie, 0)
                rank[related_movie] += similarity_factor * rating
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N]

    def evaluate(self):
        print('evaluation start', file=sys.stderr)

        N = self.n_rec_movie

        hit = 0
        rec_count = 0
        test_count = 0
        all_rec_movies = set()
        popular_sum = 0

        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print('recommend for %d users ' % i, file=sys.stderr)
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)

            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.movie_popular[movie])

            rec_count += N
            test_count += len(test_movies)

            precision = hit / (1.0 * rec_count)
            recall = hit / (1.0 * test_count)
            coverage = len(all_rec_movies) / (1.0 * self.movie_count)
            popularity = popular_sum / (1.0 * rec_count)

            with open('./outputs/metrics.csv', 'a') as f:
                f.write('"ItemCF",%.4f,%.4f,%.4f,%.4f\n' % (precision, recall, coverage, popularity))

    def generate_recommendation(self):
        ''' 输出推荐结果 '''
        print('generating recommendation result...', file=sys.stderr)
        with open('./outputs/itemcf_recommendation.csv', 'w') as f:
            for i, user in enumerate(self.trainset):
                if i % 500 == 0:
                    print('generate recommendation for %d users' % i, file=sys.stderr)
                rec_movies = self.recommend(user)
                f.write('%s,%s\n' % (user, ','.join([movie for movie, _ in rec_movies])))
        print('generate recommendation result succ', file=sys.stderr)

if __name__ == '__main__':
    ratingfile = "./data/ml-1m/ml-1m/ratings.dat"
    item_cf = ItemCF()
    item_cf.generate_dataset(ratingfile)
    item_cf.calc_movie_sim()
    item_cf.evaluate()
    item_cf.generate_recommendation()

