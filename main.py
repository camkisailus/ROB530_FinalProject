import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import pandas as pd
import utils

def objective(mu, cov, learning_rate, decay_rate, m, particles, origin, resolution):

    cov_norm = np.sqrt(float(cov[0,0])**2+float(cov[1,1])**2+float(cov[5,5])**2)
    learning_rate = (learning_rate)/(1+decay_rate)
    num_particles = len(particles)
    f=learning_rate /((cov_norm * overlap(particles,m,origin,resolution)))
    # print("cov: {}, overlap: {}".format(cov, overlap(particles,m,origin,resolution)))
    return f, learning_rate

def overlap(particles,m,origin,resolution):
    count=0.0
    for i in range(particles.shape[0]):
        x,y=float(particles[i][0]),float(particles[i][1])

        index_vert,index_hori=xy_tomap(x,y,origin,resolution)
        if m[index_vert,index_hori] == 0 or m[index_vert,index_hori] == 205:
            count+=1.0

    return (count+1.0)/((particles.shape[0])+1.0)

def xy_tomap(x,y,origin,resolution):
    return int(2047-(y+51.225)/0.05), int((x+51.225)/0.05)

def run(environment, maps, save_data):
    map_names=["bedroom","dining_room","garage","obstacle_world","study","turtle_world"]
    learning_rates =[1 for map_name in map_names]
    decay=0
    obj={environment +"_"+map_name: 1/len(map_names) for map_name in map_names}
    beliefs_dict = {environment+"_"+map_name: [1/len(map_names)] for map_name in map_names}
    k = 0
    while True:
        k+=1
        sum = 0
        for i, map_name in enumerate(map_names):
            dir="data/"+environment+"/"+environment+"_"+map_name
            file_cov = "/covariance_"+str(k)+".csv"
            file_pose = "/pose_"+str(k)+".csv"
            file_particles = "/particles_"+str(k)+".csv"
            try:
                rows=[]
                with open (dir+file_cov,"r") as f:
                    csvreader = csv.reader(f)
                    for row in csvreader:
                        rows.append(row)
                cov=np.array(rows)

                rows=[]
                with open (dir+file_pose,"r") as f:
                    csvreader = csv.reader(f)
                    for row in csvreader:
                        rows.append(row)
                poses=np.array(rows)

                rows=[]
                with open (dir+file_particles,"r") as f:
                    csvreader = csv.reader(f)
                    for row in csvreader:
                        rows.append(row)
                particles=np.array(rows)
            except:
                print("Ran out of data before reaching 95% belief in any map")
                raise(RuntimeError)


            m=maps[map_name].map
            origin=maps[map_name].origin
            resolution=maps[map_name].resolution
            score, learning_rates[i] =objective(poses, cov, learning_rates[i], decay, m, particles, origin, resolution)

            obj[environment+"_"+map_name] = obj[environment+"_"+map_name]*score
            sum += obj[environment+"_"+map_name]
        for map_name in map_names:
            obj[environment+"_"+map_name] = obj[environment+"_"+map_name]/sum
            beliefs_dict[environment+"_"+map_name].append(obj[environment+"_"+map_name])
        beliefs = obj.values()
        max_belief = max(beliefs)
        print("Max belief at iter {} is map {}: {}".format(k,  max_belief, max(obj, key=obj.get)))
        if max_belief >= 0.95:
            print("Reached greater than 95% confidence in a map.")
            sorted_beliefs = sorted(beliefs, reverse=True)
            for idx, b in enumerate(sorted_beliefs):
                for map_name, belief in obj.items():
                    if belief == b:
                        print("Rank {}: {} with belief: {}".format(idx+1, map_name, belief))
            break
    if save_data:
        df = pd.DataFrame(data=beliefs_dict)
        out = "data/"+environment+"/lrcov_norm_ablation_results.csv"
        df.to_csv(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', required=True, help='True environment the robot is in.')
    parser.add_argument('--save', action='store_true', default=False, help="Store beliefs at every iteration to csv")
    args = parser.parse_args()
    maps = utils.load_maps()
    run(args.environment, maps, args.save)
    
