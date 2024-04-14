import os
from constants import *

users = [user for user in os.listdir(PATH)]

for user in users:
    print('GETTING DATA FROM USER:', user)

    # Get the list of files in the user's directory
    files = [file for file in os.listdir(PATH + user)]
    print('FILES:', files)
