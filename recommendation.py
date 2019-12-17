import pprint

import numpy as np
from collections import defaultdict
import tkinter as tk
from geopy.geocoders import Nominatim
from tkinter import *
import pickle, time
from sklearn.externals import joblib

from lib.UserBasedCF import UserBasedCF
from lib.FriendBasedCF import FriendBasedCF
from lib.PowerLaw import PowerLaw

from lib.metrics import precisionk, recallk

def read_friend_data():
    social_data = open(social_file, 'r').readlines()
    social_relations = defaultdict(list)
    for eachline in social_data:
        uid1, uid2 = eachline.strip().split()
        uid1, uid2 = int(uid1), int(uid2)
        social_relations[uid1].append(uid2)
        social_relations[uid2].append(uid1)
    for uid in social_relations:
        social_relations[uid] = set(social_relations[uid])
    return social_relations


def read_poi_coos():
    poi_coos = {}
    poi_data = open(poi_file, 'r').readlines()
    for eachline in poi_data:
        lid, lat, lng = eachline.strip().split()
        lid, lat, lng = int(lid), float(lat), float(lng)
        poi_coos[lid] = (lat, lng)
    return poi_coos

def read_training_data():
    train_data = open(train_file, 'r').readlines()
    training_matrix = np.zeros((user_num, poi_num))
    for eachline in train_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        training_matrix[uid, lid] = 1.0
    return training_matrix


def read_ground_truth():
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth


def normalize(scores):
    max_score = max(scores)
    if not max_score == 0:
        scores = [s / max_score for s in scores]
    return scores

def predict_user(check_ins,friends):
    X = np.zeros((poi_num,),dtype='float32')
    X[check_ins] = 1.0

    U_score = U.predict_user(X.reshape(1,-1)).reshape(-1)
    #s1 = time.perf_counter()
    S_Score = np.array([S.predict_user(friends,X,lj) for lj in range(poi_num)])
    #s2 = time.perf_counter()
    G_Score = np.array([G.predict_user(X,lj) for lj in range(poi_num)])
    #s3 = time.perf_counter()

    overall_scores = (1.0 - alpha - beta) * U_score + alpha * S_Score + beta * G_Score
    predicted = list(reversed(overall_scores.argsort()))[:top_k]
    return predicted

OBJS = {}


def train(evaluate=False):
    global OBJS
    training_matrix = read_training_data()
    social_relations = read_friend_data()
    ground_truth = read_ground_truth()
    poi_coos = read_poi_coos()

    U.pre_compute_rec_scores(training_matrix)
    S.compute_friend_sim(social_relations, training_matrix)
    G.fit_distance_distribution(training_matrix, poi_coos)

    if evaluate==False:
        return 0

    result_out = open("result/predictions_top_" + str(top_k) + ".txt", 'w')

    all_uids = list(range(user_num))
    all_lids = list(range(poi_num))
    np.random.shuffle(all_uids)

    precision, recall = [], []
    for cnt, uid in enumerate(all_uids):
        if uid in ground_truth:
            U_scores = normalize([U.predict(uid, lid)
                                  if training_matrix[uid, lid] == 0 else -1
                                  for lid in all_lids])
            S_scores = normalize([S.predict(uid, lid)
                                  if training_matrix[uid, lid] == 0 else -1
                                  for lid in all_lids])
            G_scores = normalize([G.predict(uid, lid)
                                  if training_matrix[uid, lid] == 0 else -1
                                  for lid in all_lids])

            U_scores = np.array(U_scores)
            S_scores = np.array(S_scores)
            G_scores = np.array(G_scores)

            overall_scores = (1.0 - alpha - beta) * U_scores + alpha * S_scores + beta * G_scores

            predicted = list(reversed(overall_scores.argsort()))[:top_k]
            actual = ground_truth[uid]

            precision.append(precisionk(actual, predicted[:top_k]))
            recall.append(recallk(actual, predicted[:top_k]))

            print(cnt, uid, "pre@10:", np.mean(precision), "rec@10:", np.mean(recall))
            result_out.write('\t'.join([
                str(cnt),
                str(uid),
                ','.join([str(lid) for lid in predicted])
            ]) + '\n')


if __name__ == '__main__':
    data_dir = "../Gowalla_processed/"
    result_out = open("result/predictions_top_" + str(100) + ".txt", 'w')

    size_file = data_dir + "Gowalla_data_size.txt"
    check_in_file = data_dir + "Gowalla_checkins.txt"
    train_file = data_dir + "Gowalla_train.txt"
    tune_file = data_dir + "Gowalla_tune.txt"
    test_file = data_dir + "Gowalla_test.txt"
    social_file = data_dir + "Gowalla_social_relations.txt"
    poi_file = data_dir + "Gowalla_poi_coos.txt"

    user_num, poi_num = open(size_file, 'r').readlines()[0].strip('\n').split()
    user_num, poi_num = int(user_num), int(poi_num)
    print("No of user, poi:",user_num,poi_num)

    top_k = 5
    alpha = 0.5
    beta = 0

    U = UserBasedCF()
    S = FriendBasedCF(eta=0.05)
    G = PowerLaw()

    #st = time.perf_counter()
    train(evaluate=False)
    #print("Whole Training took:",time.perf_counter()-st)

    training_matrix = read_training_data()
    social_relations = read_friend_data()
    #check_ins = training_matrix[0].nonzero()[0]
    # check_ins = [1,2,6,7,12]
    # print(check_ins)
    # friends = social_relations[0]
    # friends = [1,2,3,4,5,6]
    # print(friends)
    # preds = predict_user(check_ins,friends)
    # print("predictions:", preds)

    check_ins = [0,0,0,0,0]
    friends = [0,0,0,0,0]

    poi_coos = read_poi_coos()

    def getPlaceName(id):
        #result = rg.search(poi_coos[id])
        # result is a list containing ordered dictionary.
        #pprint.pprint(result)
        geolocator = Nominatim(user_agent="wheretogo")
        location = geolocator.reverse(poi_coos[id])
        return location.address


    def show_entry_fields():
        #print("First Name: %s\nLast Name: %s" % (e1.get(), e2.get()))
        check_ins[0] = int(e1.get())
        check_ins[1] = int(e2.get())
        check_ins[2] = int(e3.get())
        check_ins[3] = int(e4.get())
        check_ins[4] = int(e5.get())
        friends[0] = int(e6.get())
        friends[1] = int(e7.get())
        friends[2] = int(e8.get())
        friends[3] = int(e9.get())
        friends[4] = int(e10.get())
        print(check_ins)
        print(friends)
        preds = predict_user(check_ins,friends)


        print("predictions:", preds)
        #preds = [13, 121, 123, 4124, 41]

        tk.Label(master,
                 text="Top 5 Recommended Places:", fg="blue", font="Verdana 11 bold").place(x=520, y=320)
        tk.Label(master,
                 text="Pid: "+str(preds[0]) +" : " + getPlaceName(preds[0])+ "\n" ,
                 fg="black", font="Verdana 8 bold").place(x=30, y=360)
        tk.Label(master,
                 text="Pid: "+str(preds[1]) + " : " + getPlaceName(preds[1])+ "\n",
                 fg="black", font="Verdana 8 bold").place(x=30, y=390)
        tk.Label(master,
                 text="Pid: "+str(preds[2]) + " : " + getPlaceName(preds[2])+ "\n",
                 fg="black", font="Verdana 8 bold").place(x=30, y=420)
        tk.Label(master,
                 text="Pid: "+str(preds[3]) + " : " + getPlaceName(preds[3])+ "\n",
                 fg="black", font="Verdana 8 bold").place(x=30, y=450)
        tk.Label(master,
                 text="Pid: "+str(preds[4]) + " : " +getPlaceName(preds[4])+ "\n",
                 fg="black", font="Verdana 8 bold").place(x=30, y=480)

    master = tk.Tk()
    master.geometry("1250x550")
    master.title("Where To Go")
    tk.Label(master, text="Where to Go!", fg="black", bg="grey", font="Times 18 bold").place(x=610, y=10)
    tk.Label(master, text="Enter your checkins and friends id!",  fg = "blue",font = "Verdana 11 bold").place(x=520, y=60)
    tk.Label(master,
             text="Check In ID-1:").place(x=430, y=90)
    tk.Label(master,
             text="Check In ID-2:").place(x=430, y=120)
    tk.Label(master,
             text="Check In ID-3:").place(x=430, y=150)
    tk.Label(master,
             text="Check In ID-4:").place(x=430, y=180)
    tk.Label(master,
             text="Check In ID-5:").place(x=430, y=210)

    tk.Label(master,
             text="Friend ID-1:").place(x=660, y=90)
    tk.Label(master,
             text="Friend ID-2:").place(x=660, y=120)
    tk.Label(master,
             text="Friend ID-3:").place(x=660, y=150)
    tk.Label(master,
             text="Friend ID-4:").place(x=660, y=180)
    tk.Label(master,
             text="Friend ID-5:").place(x=660, y=210)

    e1 = tk.Entry(master)
    e2 = tk.Entry(master)
    e3 = tk.Entry(master)
    e4 = tk.Entry(master)
    e5 = tk.Entry(master)
    e6 = tk.Entry(master)
    e7 = tk.Entry(master)
    e8 = tk.Entry(master)
    e9 = tk.Entry(master)
    e10 = tk.Entry(master)

    e1.place(x=515,y=90)
    e2.place(x=515,y=120)
    e3.place(x=515,y=150)
    e4.place(x=515,y=180)
    e5.place(x=515,y=210)
    e6.place(x=730,y=90)
    e7.place(x=730,y=120)
    e8.place(x=730,y=150)
    e9.place(x=730,y=180)
    e10.place(x=730,y=210)

    tk.Button(master,
              text='Quit Recommendation', bg="white",
              command=master.quit).place(x=700,y=270)
    tk.Button(master,
              text='Show Recommendation', bg="green", command=show_entry_fields).place(x=500,y=270)

    tk.mainloop()


    #check_ins = [1,2,6,7,12]
    #print(check_ins)
    #friends = social_relations[0]
    #friends = [1,2,3,4,5,6]
    #print(friends)
    #preds = predict_user(check_ins,friends)
    #print("predictions:", preds)



















