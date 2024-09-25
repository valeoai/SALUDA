class ToDict(object):

    def __call__(self, data):

        d = {}
        for key in data.keys():
            d[key] = data[key]
        return d

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)