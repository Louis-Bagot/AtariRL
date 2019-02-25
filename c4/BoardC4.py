class BoardC4:
  """ Connect 4 Board"""
  nodes = {}

  def __init__(self,other=None):
    self.playerJustMoved = -1
    self.trans = {1:[1,0], -1:[0,1], 0:[0,0]}
    self.width = 7
    self.height = 6
    self.fields = {}
    for y in range(self.height):
      for x in range(self.width):
        self.fields[x,y] = 0

  def switch_player():
    self.playerJustMoved *= -1

  def move(self,x):
    switch_player()
    for y in range(self.height):
      if self.fields[x,y] == 0:
        self.fields[x,y] = self.player
        break
    return self

  def legal_moves(self):
    return [i for i in range(self.width) \
            if (self.fields[i, self.height-1] == 0)]

  def fields2channels(self):
    """Preprocessing : turning the fields into 2 channels using trans dict"""
    return np.array([[self.trans.get(self.fields[x,y]) for x in range(self.width)]\
                                                       for y in range(self.height)])

  def dico2array(self):
    return np.array([[self.fields[x,y] for x in range(self.width)] \
                                       for y in range(self.height)])

  def win_check(self):
    # horizontal
    for y in range(self.height):
      winning = []
      for x in range(self.width):
        if self.fields[x,y] == self.playerJustMoved:
          winning.append((x,y))
          if len(winning) == 4:
            return winning
        else:
          winning = []
    # vertical
    for x in range(self.width):
      winning = []
      for y in range(self.height):
        if self.fields[x,y] == self.playerJustMoved:
          winning.append((x,y))
          if len(winning) == 4:
            return winning
        else:
          winning = []
    # diagonal
    winning = []
    for cx in range(self.width-1):
      sx,sy = max(cx-2,0),abs(min(cx-2,0))
      winning = []
      for cy in range(self.height):
        x,y = sx+cy,sy+cy
        if x<0 or y<0 or x>=self.width or y>=self.height:
          continue
        if self.fields[x,y] == self.playerJustMoved:
          winning.append((x,y))
          if len(winning) == 4:
            return winning
        else:
          winning = []
    # other diagonal
    winning = []
    for cx in range(self.width-1):
      sx,sy = self.width-1-max(cx-2,0),abs(min(cx-2,0))
      winning = []
      for cy in range(self.height):
        x,y = sx-cy,sy+cy
        if x<0 or y<0 or x>=self.width or y>=self.height:
          continue
        if self.fields[x,y] == self.playerJustMoved:
          winning.append((x,y))
          if len(winning) == 4:
            return winning
        else:
          winning = []
    # default
    return None
