import numpy as np
import pickle


token_file = "music_tokens/new_songs_game100"

with open(token_file, "rb") as file:
    t = pickle.load(file)
    # print(t)
np_dimension = (len(t), 2500, 6)
res = -np.ones(np_dimension)

for i in range(len(t)):
    for j in range(len(t[i])):
        print(t[i][j])
        if t[i][j] != "start":
            temp = t[i][j].split("|")

            res[i][j] = list(map(lambda x: float(x), temp))

        else:
            res[i][j] = [-1,-1.0,-1.0,-1.0,-1.0,-1]
            print("ok")

output_file = "music_tokens/output/new_songs_game100.npy"
np.save(output_file, res)


