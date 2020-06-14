class Cell:
    def __init__(self, id_):
        self._id = id_

    def __str__(self):
        return self.id

    @property
    def id(self):
        return self._id
