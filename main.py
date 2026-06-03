import time
from py3x import user_cf
from py3x import item_cf

if __name__ == '__main__':
    ratingfile = "./data/ml-1m/ml-1m/ratings.dat"
    user_cf_obj = user_cf.UserCF()
    t1=time.time()
    user_cf_obj.generate_dataset(ratingfile)
    user_cf_obj.calc_user_sim()
    user_cf_obj.evaluate()
    user_cf_obj.generate_recommendation()
    t2=time.time()
    print('user_cf算法耗时: %.2f' % (t2-t1))
    item_cf_obj = item_cf.ItemCF()
    t1=time.time()
    item_cf_obj.generate_dataset(ratingfile)
    item_cf_obj.calc_movie_sim()
    item_cf_obj.evaluate()
    item_cf_obj.generate_recommendation()
    t2=time.time()
    print('item_cf算法耗时: %.2f' % (t2-t1))