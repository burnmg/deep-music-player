from data_processing_helpers import from_four_to_three_vec, from_three_to_four_vec
import numpy as np


a = np.array([[76, 90, 0.0, 1.25],
              [90, 110, 1.25, 4.25],])
# b = from_four_to_three_vec(a)
# print(b)
# c = from_three_to_four_vec(b)


t = np.array([a, a])
np.save("res")
print(t)
# notes { pitch: 76 velocity: 90 start_time: 1.0 end_time: 1.25 }