import os



# Prints all of the directories under the root directory
def tree_directory_printer(root):
    '''Root should be formatted with forward slashes or double back slashes'''
    for root, dirs, files in os.walk(root):
        for d in dirs:
            print (os.path.join(root, d))    

# Prints all of the files under the root directory
def tree_file_printer(root):
    '''Root should be formatted with forward slashes or double back slashes'''
    for root, dirs, files in os.walk(root):
        for f in files:
            print (os.path.join(root, f))

# tree_directory_printer('C:/Users/Ross/OneDrive/Documents/Thesis')