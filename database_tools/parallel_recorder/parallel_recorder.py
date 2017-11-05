import os
from pprint import pprint

from database_tools.get_all_files_in_tree import get_file_and_path_list
from params.params import get_params

params = get_params()


bruce_willis_file_and_path_list = get_file_and_path_list(os.path.join(params["project_base_path"],"data/bruce_willis/Studio"))
pierre_sendorek_file_and_path_list = [(os.path.join(params["project_base_path"], "data/pierre_sendorek/Studio"),
                                       filename[1]) for filename in bruce_willis_file_and_path_list]

pprint([v[1] for v in bruce_willis_file_and_path_list])

pprint(pierre_sendorek_file_and_path_list)

assemble_name = lambda el: os.path.join(el[0], el[1])

i = 0
while i < len(bruce_willis_file_and_path_list):
    bruce_willis_wav_file = assemble_name(bruce_willis_file_and_path_list[i])
    pierre_sendorek_wav_file = assemble_name(pierre_sendorek_file_and_path_list[i])
    os.system("play " + bruce_willis_wav_file)
    input("press enter when ready")
    os.system("rec -c1 -r44100 " + pierre_sendorek_wav_file)
    print("to listen and record again type a")
    print("to go forward type f")
    print("to go to previous one type p")
    c = input("then press enter\n")
    if c == "f":
        i += 1
    if c == "a":
        i += 0
    if c == "p":
        i -= 1

#retvalue = os.system("rec -c1 -r 441000 test.wav")
#print(retvalue)