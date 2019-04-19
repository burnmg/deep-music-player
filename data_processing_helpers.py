import re
import numpy as np


def from_four_to_three_vec(input):
    """
    convert 4-tuple np arrays to 3-tuple np arrays
    from (pitch, velocity, start_time, end_time)
    to (pitch, velocity, interval_time)
    :param input: np array
    :return: np aray
    """

    return np.apply_along_axis(lambda row:[row[0], row[1], row[3]-row[2]], 1, input)


def from_three_to_four_vec(input):
    """
    convert 3-tuple np arrays to 4-tuple np arrays
    from (pitch, velocity, interval_time)
    to (pitch, velocity, start_time, end_time)
    :param input: np array
    :return: np aray
    """
    res = []
    cur_time = 0.0
    for row in input:
        res.append([row[0], row[1], cur_time, cur_time + row[2]])
        cur_time += row[2]

    return np.array(res)


def clean_music(music_score):
    """
    clean data's illegal characters.

    :param music_score: string
    :return: cleaned_score: string
    """
    p = re.compile(r"\(3(\w)\w(\w)")
    res = p.sub(r"\1\2", music_score)
    res = res.replace("!", "")
    return res

