import pickle



def u_save(filename, user):
    file = open(filename + '.usr', 'wb')
    file.write(pickle.dumps(user.__dict__))
    file.close()


def u_load(filename):
    file = open(filename + '.usr', 'rb')
    dataPickle = file.read()
    file.close()
    return pickle.loads(dataPickle)
