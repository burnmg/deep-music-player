import re

def clean_music(music_score):
    """

    :param music_score: string
    :return: cleaned_score,
    """
    p = re.compile(r"\(3(\w)\w(\w)")
    res = p.sub(r"\1\2", music_score)
    res = res.replace("!", "")
    return res


with open("test_data", "r") as input:
    str = input.read()
    print(
        clean_music(str)
          )
