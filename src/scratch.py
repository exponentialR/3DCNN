import configparser

configfile = configparser.ConfigParser()
configfile.read('config.ini')
configfile_head = configfile['hyperparameter']
classes = configfile_head['classes_to_use']
classes_list = eval(classes)
for i in classes_list:
    print(i)
